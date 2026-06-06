"""Heat Kernel Diagonal Structural Encoding (HKdiagSE) Transform."""

import time

import omegaconf
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    remove_self_loops,
)


class HKdiagSE(BaseTransform):
    r"""
    Heat Kernel Diagonal Structural Encoding (HKdiagSE) transform.

    Parameters
    ----------
    kernel_param_HKdiagSE : tuple of int
        Tuple specifying the start and end diffusion times for the heat kernel.
    space_dim : int, optional
        Estimated dimensionality of the space. Used to correct the diffusion
        diagonal by a factor `t^(space_dim/2)`. Default is 0 (no correction).
    include_eigenvalues : bool, optional
        If True, concatenates eigenvalues alongside eigenvectors.
        Default is False.
    include_first : bool, optional
        If False, removes eigenvectors corresponding to (near-)zero eigenvalues.
        Default is False.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features.
        Default is True.
    method : str, optional
        Computation method: "exact" (CPU NumPy + loop) or "fast" (GPU PyTorch + vectorized).
        Default is "fast".
    debug : bool, optional
        If True, runs both methods and prints error/timing metrics.
        Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        kernel_param_HKdiagSE: tuple,
        space_dim: int = 0,
        include_eigenvalues: bool = False,
        include_first: bool = False,
        concat_to_x: bool = True,
        method: str = "fast",
        debug: bool = False,
        **kwargs,
    ):
        self.kernel_param_HKdiagSE = kernel_param_HKdiagSE
        self.space_dim = space_dim
        self.include_eigenvalues = include_eigenvalues
        self.include_first = include_first
        self.concat_to_x = concat_to_x
        self.method = method
        self.debug = debug
        self.pe_dim = (
            kernel_param_HKdiagSE[1] - kernel_param_HKdiagSE[0]
            if type(kernel_param_HKdiagSE) is omegaconf.listconfig.ListConfig
            else kernel_param_HKdiagSE
        )

        if method not in ["exact", "fast"]:
            raise ValueError("Method must be 'exact' or 'fast'.")

    def forward(self, data: Data) -> Data:
        """Compute the Heat Kernel Diagonal Structural Encodings for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with HKdiagSE positional encodings added.
        """
        if self.debug:
            print("\n--- HKdiagSE Debug Report ---")
            print(f"Data device:        {data.edge_index.device}")
            # Exact Method (CPU)
            t0 = time.time()
            pe_exact = self._compute_exact(data.edge_index, data.num_nodes)
            t_exact = time.time() - t0
            print(f"Exact compute time:  {t_exact:.4f}s")

            # Fast Method (GPU Vectorized)
            t0 = time.time()
            pe_fast = self._compute_fast(data.edge_index, data.num_nodes)
            t_fast = time.time() - t0
            print(f"Fast compute time:   {t_fast:.4f}s")

            # Compare
            diff = torch.abs(pe_exact - pe_fast)
            speedup = (t_exact / t_fast) if t_fast > 0 else float("inf")
            print(f"Speedup Factor:      {speedup:.2f}x")
            print(f"Mean Abs Error:      {diff.mean().item():.6e}")
            print("---------------------------\n")

            pe = pe_exact if self.method == "exact" else pe_fast
        else:
            if self.method == "exact":
                pe = self._compute_exact(data.edge_index, data.num_nodes)
            else:
                pe = self._compute_fast(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.HKdiagSE = pe

        return data

    def _compute_fast(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute HKdiagSE using an optimized pure-PyTorch implementation with a vectorized time loop.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Heat Kernel Diagonal Structural Encodings of shape ``[num_nodes, pe_dim]``.
        """
        device = edge_index.device

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.pe_dim, device=device)

        # 1. Create dense Laplacian directly on GPU
        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight.float(),
            (num_nodes, num_nodes),
            device=device,
        ).to_dense()

        # 2. Hardware-accelerated eigendecomposition
        evals, evects = torch.linalg.eigh(L)
        evects = F.normalize(evects, p=2.0, dim=0)

        # 3. Filter out zero eigenvalues
        mask = evals >= 1e-8
        evals = evals[mask]
        evects = evects[:, mask]

        start, end = (
            self.kernel_param_HKdiagSE[0],
            self.kernel_param_HKdiagSE[1],
        )

        # 4. Vectorize the time loop
        # t_tensor shape: [T]
        t_tensor = torch.arange(start, end, dtype=torch.float32, device=device)

        if len(t_tensor) == 0:
            raise ValueError("Diffusion times are required for heat kernel")

        # Exponent matrix: shape [T, E] (Time steps by Eigenvalues)
        exp_term = torch.exp(-t_tensor.unsqueeze(1) * evals.unsqueeze(0))

        # Squared eigenvectors (phi^2): shape [E, N]
        eigvec_mul = (evects**2).T

        # Single matrix multiplication replaces the entire loop: [T, E] @ [E, N] -> [T, N]
        hk_diag = exp_term @ eigvec_mul

        # Apply spatial correction factor: t^(space_dim/2)
        if self.space_dim != 0:
            correction = t_tensor ** (self.space_dim / 2.0)
            hk_diag = hk_diag * correction.unsqueeze(1)

        # Transpose to return [N, T]
        hk_diag = hk_diag.transpose(0, 1)

        # Corner case checking
        if (
            (torch.all(hk_diag == 0))
            and (num_nodes > 2)
            and list(remove_self_loops(edge_index)[0].cpu().shape) != [2, 0]
        ):
            raise ValueError("HKdiagSE is all zeros")

        if torch.any(torch.isnan(hk_diag)):
            raise ValueError("HKdiagSE contains NaNs")

        return hk_diag

    def _compute_exact(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute HKdiagSE using the original, un-optimized CPU NumPy implementation.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Heat Kernel Diagonal Structural Encodings of shape ``[num_nodes, pe_dim]``.
        """
        import numpy as np
        from torch_geometric.utils import to_scipy_sparse_matrix

        device = edge_index.device
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.pe_dim, device=device)

        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(
            edge_index_lap, edge_weight, num_nodes
        ).astype(np.float64)

        evals, evects = np.linalg.eigh(L.toarray())
        evals = torch.from_numpy(evals).to(device)
        evects = torch.from_numpy(evects).to(device)

        start, end = (
            self.kernel_param_HKdiagSE[0],
            self.kernel_param_HKdiagSE[1],
        )
        kernel_times = range(start, end)

        if len(kernel_times) == 0:
            raise ValueError("Diffusion times are required for heat kernel")

        hk_diag = []
        evects = F.normalize(evects, p=2.0, dim=0)

        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        evals = evals.unsqueeze(-1)
        evects = evects.transpose(0, 1)

        eigvec_mul = evects**2
        for t in kernel_times:
            this_kernel = torch.sum(
                torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False
            )
            hk_diag.append(this_kernel * (t ** (self.space_dim / 2)))

        hk_diag = torch.stack(hk_diag, dim=0).transpose(0, 1)

        if (
            (torch.all(hk_diag == 0))
            and (num_nodes > 2)
            and list(remove_self_loops(edge_index)[0].cpu().shape) != [2, 0]
        ):
            raise ValueError("HKdiagSE is all zeros")

        return hk_diag.float().to(device)
