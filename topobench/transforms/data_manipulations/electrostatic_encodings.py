"""Electrostatic Positional Encoding (ElectrostaticPE) Transform."""

import time

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    remove_self_loops,
)


class ElectrostaticPE(BaseTransform):
    r"""
    Electrostatic Positional Encoding (ElectrostaticPE) transform.

    Parameters
    ----------
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features.
        Default is True.
    eps : float, optional
        Small value to avoid division by zero.
        Default is 1e-6.
    method : str, optional
        Computation method: "numpy" (CPU NumPy) or "gpu" (GPU PyTorch).
        Default is "gpu".
    debug : bool, optional
        If True, runs both methods and compares outputs.
        Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        method: str = "numpy",
        debug: bool = False,
        **kwargs,
    ):
        self.concat_to_x = concat_to_x
        self.eps = eps
        self.method = method
        self.debug = debug
        self.pe_dim = 7

        if method not in ["numpy", "gpu"]:
            raise ValueError("Method must be 'numpy' or 'gpu'.")

    def forward(self, data: Data) -> Data:
        """Compute the electrostatic positional encodings for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with electrostatic positional encodings added.
        """
        if self.debug:
            print("\n--- ElectrostaticPE Debug Report ---")
            print(f"Data device:        {data.edge_index.device}")
            # Exact Method (Original CPU NumPy)
            t0 = time.time()
            pe_numpy = self._compute_numpy(data.edge_index, data.num_nodes)
            t_numpy = time.time() - t0
            print(f"Exact compute time:  {t_numpy:.4f}s")

            # Fast Method (Pure PyTorch GPU)
            t0 = time.time()
            pe_gpu = self._compute_gpu(data.edge_index, data.num_nodes)
            t_gpu = time.time() - t0
            print(f"Fast compute time:   {t_gpu:.4f}s")

            # Compare (Only if non-zero)
            diff = torch.abs(pe_numpy - pe_gpu)
            speedup = (t_numpy / t_gpu) if t_gpu > 0 else float("inf")
            print(f"Speedup Factor:      {speedup:.2f}x")
            print(f"Mean Abs Error:      {diff.mean().item():.6e}")
            print("------------------------------------\n")

            pe = pe_numpy if self.method == "numpy" else pe_gpu
        else:
            if self.method == "numpy":
                pe = self._compute_numpy(data.edge_index, data.num_nodes)
            else:
                pe = self._compute_gpu(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.ElectrostaticPE = pe

        return data

    def _compute_gpu(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute ElectrostaticPE using optimized pure-PyTorch implementation.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Electrostatic positional encodings of shape ``[num_nodes, 7]``.
        """
        device = edge_index.device
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.pe_dim, device=device)

        # 1. Get Normalized Laplacian and make it dense immediately on device
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight.float(),
            (num_nodes, num_nodes),
            device=device,
        ).to_dense()

        # 2. Efficiently compute DinvA without deepcopy or torch.eye
        diag_L = L.diagonal()
        Dinv_vec = 1.0 / (diag_L + 1e-6)

        A = L.abs()
        A.fill_diagonal_(0)
        # Broadcasting [N, 1] * [N, N] applies the row-wise scalar multiplication identical to Dinv @ A
        DinvA = Dinv_vec.unsqueeze(1) * A

        # 3. Hardware-accelerated eigendecomposition
        evals, evecs = torch.linalg.eigh(L)

        # 4. Filter eigenvalues
        mask = evals >= self.eps
        if not mask.any():
            return torch.zeros(
                num_nodes, self.pe_dim, dtype=torch.float32, device=device
            )

        evals_filtered = evals[mask]
        evecs_filtered = evecs[:, mask]

        # 5. Reconstruct Pseudo-Inverse (Electrostatic matrix)
        # evecs @ diag(1/evals) @ evecs.T
        electrostatic = (evecs_filtered / evals_filtered) @ evecs_filtered.T

        # Broadcast subtraction of the diagonal
        electrostatic = electrostatic - electrostatic.diag()

        # 6. Compute statistics
        # Note: dim=0 is operations along columns, dim=1 is operations along rows
        electrostatic_encoding = torch.stack(
            [
                electrostatic.min(dim=0)[0],
                electrostatic.mean(dim=0),
                electrostatic.std(dim=0),
                electrostatic.min(dim=1)[0],
                electrostatic.std(dim=1),
                (DinvA * electrostatic).sum(dim=0),
                (DinvA * electrostatic).sum(dim=1),
            ],
            dim=1,
        )

        # Corner case check
        if (
            torch.all(electrostatic_encoding == 0)
            and num_nodes > 2
            and list(remove_self_loops(edge_index)[0].cpu().shape) != [2, 0]
        ):
            raise ValueError("ElectrostaticPE is all zeros")

        if torch.any(torch.isnan(electrostatic_encoding)):
            raise ValueError("ElectrostaticPE contains NaNs")

        return electrostatic_encoding

    def _compute_numpy(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute ElectrostaticPE using the original CPU NumPy implementation.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Electrostatic positional encodings of shape ``[num_nodes, 7]``.
        """
        from copy import deepcopy

        import numpy as np
        from torch_geometric.utils import to_scipy_sparse_matrix

        device = edge_index.device
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.pe_dim, device=device)

        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = (
            to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
            .astype(np.float64)
            .todense()
        )

        L = torch.as_tensor(L)
        Dinv = torch.eye(L.shape[0]) * ((L.diag() + 1e-6) ** -1)
        A = deepcopy(L).abs()
        A.fill_diagonal_(0)
        DinvA = Dinv.matmul(A)

        evals, evecs = np.linalg.eigh(L.numpy())
        evals = torch.from_numpy(evals)
        evecs = torch.from_numpy(evecs)

        offset = (evals < self.eps).sum().item()
        if offset == num_nodes:
            return torch.zeros(num_nodes, 7, dtype=torch.float32)

        electrostatic = (
            evecs[:, offset:] / evals[offset:] @ evecs[:, offset:].T
        )
        electrostatic = electrostatic - electrostatic.diag()
        electrostatic_encoding = torch.stack(
            [
                electrostatic.min(dim=0)[0],
                electrostatic.mean(dim=0),
                electrostatic.std(dim=0),
                electrostatic.min(dim=1)[0],
                electrostatic.std(dim=1),
                (DinvA * electrostatic).sum(dim=0),
                (DinvA * electrostatic).sum(dim=1),
            ],
            dim=1,
        )

        if (
            torch.all(electrostatic_encoding == 0)
            and num_nodes > 2
            and list(remove_self_loops(edge_index)[0].cpu().shape) != [2, 0]
        ):
            raise ValueError("ElectrostaticPE is all zeros")

        return electrostatic_encoding.float().to(device)
