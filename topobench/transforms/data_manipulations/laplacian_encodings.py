"""Laplacian Positional Encoding (LapPE) Transform."""

import time

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, get_laplacian, to_scipy_sparse_matrix


class LapPE(BaseTransform):
    r"""
    Laplacian Positional Encoding (LapPE) transform.

    Parameters
    ----------
    max_pe_dim : int
        Maximum number of eigenvectors to use (dimensionality of the encoding).
    include_eigenvalues : bool, optional
        If True, concatenates eigenvalues alongside eigenvectors.
        Default is False.
    include_first : bool, optional
        If False, removes eigenvectors corresponding to (near-)zero eigenvalues.
        Default is False.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features.
        Default is True.
    eps : float, optional
        Small value to avoid division by zero.
        Default is 1e-6.
    tolerance : float, optional
        Tolerance for the eigenvalue solver.
        Default is 0.001.
    method : str, optional
        Computation method: "exact" (SciPy CPU) or "gpu" (PyTorch GPU).
        Default is "gpu".
    debug : bool, optional
        If True, runs both methods and prints error/timing metrics.
        Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        max_pe_dim: int,
        include_eigenvalues: bool = False,
        include_first: bool = False,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        tolerance: float = 0.001,
        method: str = "gpu",
        debug: bool = False,
        **kwargs,
    ):
        self.max_pe_dim = max_pe_dim
        self.include_eigenvalues = include_eigenvalues
        self.include_first = include_first
        self.concat_to_x = concat_to_x
        self.eps = eps
        self.tolerance = tolerance
        self.debug = debug

        if method not in ["exact", "gpu"]:
            raise ValueError("Method must be 'exact' or 'gpu'.")
        self.method = method

    def forward(self, data: Data) -> Data:
        """Compute the Laplacian positional encodings for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with Laplacian positional encodings added.
        """
        if self.debug:
            print("\n--- LapPE Debug Report ---")
            print(f"Data device:        {data.edge_index.device}")
            # Exact Method (SciPy CPU)
            t0 = time.time()
            pe_exact = self._compute_exact(data.edge_index, data.num_nodes)
            t_exact = time.time() - t0
            print(f"Exact compute time:  {t_exact:.4f}s")

            # Fast Method (PyTorch GPU)
            t0 = time.time()
            pe_gpu = self._compute_gpu(data.edge_index, data.num_nodes)
            t_gpu = time.time() - t0
            print(f"Fast compute time:   {t_gpu:.4f}s")

            # Compare Tensors
            diff = torch.abs(pe_exact - pe_gpu)
            speedup = (t_exact / t_gpu) if t_gpu > 0 else float("inf")
            print(f"Speedup Factor:      {speedup:.2f}x")
            print(f"Mean Abs Error:      {diff.mean().item():.6e}")
            print("--------------------------\n")

            pe = pe_exact if self.method == "exact" else pe_gpu
        else:
            if self.method == "exact":
                pe = self._compute_exact(data.edge_index, data.num_nodes)
            else:
                pe = self._compute_gpu(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.LapPE = pe

        return data

    def _fix_sign_ambiguity(self, evecs: torch.Tensor) -> torch.Tensor:
        """Standardize eigenvector signs so the max absolute value is positive.

        Parameters
        ----------
        evecs : torch.Tensor
            Eigenvectors tensor of shape ``[num_nodes, max_pe_dim]``.

        Returns
        -------
        torch.Tensor
            Sign-corrected eigenvectors tensor.
        """
        max_idxs = torch.argmax(torch.abs(evecs), dim=0)
        signs = torch.sign(evecs[max_idxs, torch.arange(evecs.shape[1])])
        # Replace 0 signs with 1 to avoid zeroing out vectors
        signs[signs == 0] = 1
        return evecs * signs

    def _pad_and_concat(
        self,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pad to max_pe_dim and optionally concatenate eigenvalues.

        Parameters
        ----------
        evals : torch.Tensor
            Eigenvalues tensor of shape ``[max_pe_dim]``.
        evecs : torch.Tensor
            Eigenvectors tensor of shape ``[num_nodes, max_pe_dim]``.
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            The device to place the resulting tensor on.

        Returns
        -------
        torch.Tensor
            The padded and optionally concatenated positional encoding tensor.
        """
        # Pad if fewer than max_pe_dim
        pad_width = self.max_pe_dim - evecs.shape[1]
        if pad_width > 0:
            evecs = torch.nn.functional.pad(
                evecs, (0, pad_width), mode="constant", value=0
            )
            evals = torch.nn.functional.pad(
                evals, (0, pad_width), mode="constant", value=0
            )

        pe = evecs
        if self.include_eigenvalues:
            eigvals_broadcast = evals.unsqueeze(0).repeat(num_nodes, 1)
            pe = torch.cat([pe, eigvals_broadcast], dim=-1)

        return pe

    def _compute_exact(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute LapPE using original SciPy CPU Implementation.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Exact Laplacian positional encodings.
        """
        device = edge_index.device
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(
                num_nodes,
                self.max_pe_dim * (2 if self.include_eigenvalues else 1),
                device=device,
            )

        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(
            edge_index_lap, edge_weight, num_nodes
        ).astype(np.float64)

        k = min(
            self.max_pe_dim + (0 if self.include_first else 1), num_nodes - 1
        )
        k = max(1, k)

        try:
            evals, evecs = eigsh(L, k=k, which="SM", tol=self.tolerance)
        except Exception:
            evals, evecs = np.linalg.eigh(L.toarray())

        if not self.include_first:
            mask = evals > self.eps
            evals, evecs = evals[mask], evecs[:, mask]

        evals, evecs = evals[: self.max_pe_dim], evecs[:, : self.max_pe_dim]

        # Convert to PyTorch for sign fixing and padding
        evals = torch.from_numpy(evals).float().to(device)
        evecs = torch.from_numpy(evecs).float().to(device)

        evecs = self._fix_sign_ambiguity(evecs)
        return self._pad_and_concat(evals, evecs, num_nodes, device)

    def _compute_gpu(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute LapPE using gpu PyTorch GPU Implementation with Shift Trick.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Fast approximation of Laplacian positional encodings.
        """
        device = edge_index.device
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(
                num_nodes,
                self.max_pe_dim * (2 if self.include_eigenvalues else 1),
                device=device,
            )

        # We need k + 1 if we are dropping the first eigenvalue
        k_compute = min(
            self.max_pe_dim + (0 if self.include_first else 1), num_nodes
        )

        # 1. Get exact Laplacian edge weights
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )

        # 2. Decide solver based on graph size
        if num_nodes < 128 or k_compute >= num_nodes:
            # For small graphs, dense PyTorch GPU is instantaneously gpu and mathematically perfectly stable
            L_dense = torch.sparse_coo_tensor(
                edge_index_lap,
                edge_weight_lap.float(),
                (num_nodes, num_nodes),
                device=device,
            ).to_dense()
            evals, evecs = torch.linalg.eigh(L_dense)
        else:
            try:
                # 3. The Shift Trick: Find LARGEST eigenvalues of Adjacency (A = I - L)
                # We extract the Adjacency matrix by negating the off-diagonal Laplacian elements
                row, col = edge_index
                deg = degree(col, num_nodes, dtype=torch.float32)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
                a_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

                A_sym = torch.sparse_coo_tensor(
                    edge_index, a_weight, (num_nodes, num_nodes), device=device
                ).coalesce()

                # Provide initial guess X to speed up convergence
                X = torch.randn(
                    num_nodes, k_compute, dtype=torch.float32, device=device
                )

                # lobpcg computes largest eigenvalues natively
                evals_A, evecs = torch.lobpcg(
                    A=A_sym, X=X, largest=True, tol=self.tolerance
                )

                # Convert back to Laplacian eigenvalues (L = I - A)
                evals = 1.0 - evals_A

                # lobpcg returns descending order; we need ascending order (smallest first)
                evals, indices = torch.sort(evals, descending=False)
                evecs = evecs[:, indices]

            except Exception:
                # If the sparse graph is highly ill-conditioned and lobpcg fails, fallback to dense GPU
                L_dense = torch.sparse_coo_tensor(
                    edge_index_lap,
                    edge_weight_lap.float(),
                    (num_nodes, num_nodes),
                    device=device,
                ).to_dense()
                evals, evecs = torch.linalg.eigh(L_dense)

        # 4. Mask, Slice, and Format
        if not self.include_first:
            mask = evals > self.eps
            evals, evecs = evals[mask], evecs[:, mask]

        evals = evals[: self.max_pe_dim]
        evecs = evecs[:, : self.max_pe_dim]

        evecs = self._fix_sign_ambiguity(evecs)
        return self._pad_and_concat(evals, evecs, num_nodes, device)
