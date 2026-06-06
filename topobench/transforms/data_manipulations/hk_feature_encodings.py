"""Heat Kernel feature Encoding (HKFE) Transform (Debug Version)."""

import time

import numpy as np
import omegaconf
import torch
from scipy.sparse.linalg import expm_multiply
from scipy.special import iv
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)


class HKFE(BaseTransform):
    r"""
    Heat Kernel Feature Encodings (HKFE) transform.

    Parameters
    ----------
    kernel_param_HKFE : tuple of int
        Tuple specifying the start and end diffusion times for the heat kernel.
    concat_to_x : bool, optional
        If True, concatenates encodings with existing node features in ``data.x``.
        Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min". Default is "mean".
    method : str, optional
        Computation method: "exact" or "approx". Default is "approx".
    cheb_order : int, optional
        The order of the Chebyshev polynomial. Default is 10.
    debug : bool, optional
        If True, runs both exact and approx methods, compares their outputs,
        and prints the timing and error metrics. Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        kernel_param_HKFE: tuple,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        method: str = "approx",
        cheb_order: int = 10,
        debug: bool = False,
        **kwargs,
    ):
        self.kernel_param_HKFE = kernel_param_HKFE
        self.concat_to_x = concat_to_x

        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(f"Unknown aggregation '{aggregation}'.")
        self.aggregation = aggregation

        if method not in ["exact", "approx"]:
            raise ValueError("Method must be 'exact' or 'approx'.")
        self.method = method
        self.cheb_order = cheb_order
        self.debug = debug

        if (
            isinstance(kernel_param_HKFE, (list, tuple))
            or type(kernel_param_HKFE) is omegaconf.listconfig.ListConfig
        ):
            self.fe_dim = kernel_param_HKFE[1] - kernel_param_HKFE[0]
        else:
            self.fe_dim = kernel_param_HKFE

    def forward(self, data: Data) -> Data:
        """Compute the HKFE for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with HKFE added to ``data.x`` or ``data.HKFE``.
        """
        if data.x is None:
            raise ValueError(
                "HKFE requires node features (data.x cannot be None)"
            )

        fe = self._compute_hkfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = fe
            else:
                data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.HKFE = fe

        return data

    def _compute_hkfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute heat kernel feature encodings.

        Computes heat kernel diffusion at multiple time scales and aggregates
        over input features to produce a fixed-dimension output.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            Heat Kernel feature encodings of shape ``[num_nodes, fe_dim]``.
        """
        device = edge_index.device
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.fe_dim, device=device)

        start, end = self.kernel_param_HKFE[0], self.kernel_param_HKFE[1]
        kernel_times = np.geomspace(start, end, self.fe_dim)

        if len(kernel_times) == 0:
            raise ValueError("Diffusion times are required for heat kernel")

        edge_index_lap, edge_weight = get_laplacian(
            edge_index, normalization="sym", num_nodes=num_nodes
        )

        if self.debug:
            print("\n--- HKFE Debug Report ---")
            print(f"Data device:        {edge_index.device}")
            # 1. Run Exact Method
            t0 = time.time()
            hk_fe_exact = self._compute_exact(
                x, edge_index_lap, edge_weight, num_nodes, kernel_times, device
            )
            t_exact = time.time() - t0
            print(f"Exact compute time:  {t_exact:.4f}s")

            # 2. Run Approx Method
            t0 = time.time()
            hk_fe_approx = self._compute_approx(
                x, edge_index_lap, edge_weight, num_nodes, kernel_times, device
            )
            t_approx = time.time() - t0
            print(
                f"Approx compute time: {t_approx:.4f}s (Cheb Order: {self.cheb_order})"
            )

            # 3. Compare Tensors (Before aggregation to see pure mathematical difference)
            diff = torch.abs(hk_fe_exact - hk_fe_approx)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            reldiff = diff / (torch.abs(hk_fe_exact) + 1e-8)
            mean_reldiff = reldiff.mean().item()

            print(f"Speedup Factor:      {t_exact / t_approx:.2f}x")
            print(f"Mean Abs Error:      {mean_diff:.6e}")
            print(f"Max Abs Error:       {max_diff:.6e}")
            print(f"Mean Rel Error:      {mean_reldiff:.6e}")
            print("-------------------------\n")

            # Proceed with the method the user actually requested
            hk_fe_raw = hk_fe_exact if self.method == "exact" else hk_fe_approx
        else:
            if self.method == "exact":
                hk_fe_raw = self._compute_exact(
                    x,
                    edge_index_lap,
                    edge_weight,
                    num_nodes,
                    kernel_times,
                    device,
                )
            else:
                hk_fe_raw = self._compute_approx(
                    x,
                    edge_index_lap,
                    edge_weight,
                    num_nodes,
                    kernel_times,
                    device,
                )

        # Aggregate over features
        agg_fn = getattr(hk_fe_raw, self._AGG_FN_MAP[self.aggregation])
        hk_fe = agg_fn(dim=-1)

        if torch.any(torch.isnan(hk_fe)):
            raise ValueError("HKFE contains NaNs")
        return hk_fe.float()

    def _compute_exact(
        self,
        x: torch.Tensor,
        edge_index_lap: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        kernel_times: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute HKFE using original SciPy-based exact matrix exponential.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index_lap : torch.Tensor
            Laplacian edge indices.
        edge_weight : torch.Tensor
            Laplacian edge weights.
        num_nodes : int
            Number of nodes in the graph.
        kernel_times : numpy.ndarray
            Array of diffusion times.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        torch.Tensor
            Exact heat kernel feature encodings.
        """
        L = to_scipy_sparse_matrix(
            edge_index_lap, edge_weight, num_nodes
        ).astype(np.float64)
        hk_fe = []
        x_np = x.detach().cpu().numpy().astype(np.float64)
        for t in kernel_times:
            x_t = expm_multiply((-float(t)) * L, x_np)
            hk_fe.append(torch.from_numpy(x_t).float().to(device))
        return torch.stack(hk_fe, dim=1)

    def _compute_approx(
        self,
        x: torch.Tensor,
        edge_index_lap: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        kernel_times: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute HKFE using fast Chebyshev polynomial approximation on GPU.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index_lap : torch.Tensor
            Laplacian edge indices.
        edge_weight : torch.Tensor
            Laplacian edge weights.
        num_nodes : int
            Number of nodes in the graph.
        kernel_times : numpy.ndarray
            Array of diffusion times.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        torch.Tensor
            Approximated heat kernel feature encodings.
        """
        L = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight.float(),
            (num_nodes, num_nodes),
            device=device,
        ).coalesce()

        def apply_L_tilde(v):
            """Apply the normalized Laplacian to a vector v.

            Parameters
            ----------
            v : torch.Tensor
                Input vector of shape [num_nodes, feature_dim].

            Returns
            -------
            torch.Tensor
                Result of applying the normalized Laplacian to v.
            """
            return torch.sparse.mm(L, v) - v

        T_x = [x]
        if self.cheb_order > 0:
            T_x.append(apply_L_tilde(x))

        for _ in range(2, self.cheb_order + 1):
            T_k = 2 * apply_L_tilde(T_x[-1]) - T_x[-2]
            T_x.append(T_k)

        T_x = torch.stack(T_x, dim=0)

        # 1. Vectorize the CPU computation
        # t_np shape: (T,)
        t_np = kernel_times
        # k_np shape: (K, 1) where K is cheb_order + 1
        k_np = np.arange(self.cheb_order + 1)[:, None]

        # SciPy's iv broadcasts automatically to shape (K, T)
        bessel = iv(k_np, t_np[None, :])

        # Calculate all coefficients at once: shape (K, T)
        coeffs = 2 * np.exp(-t_np) * ((-1) ** k_np) * bessel
        coeffs[0, :] /= 2  # The k=0 term doesn't have the 2x multiplier

        # 2. Single transfer to GPU
        # Transpose to shape (T, K) for easier einsum matching
        coeffs_tensor = torch.tensor(
            coeffs.T, dtype=torch.float32, device=device
        )

        # 3. Single operation on GPU!
        # coeffs_tensor is [T, K]
        # T_x is [K, N, F]
        # We want output [N, T, F] to match your original torch.stack(..., dim=1)
        hk_fe = torch.einsum("tk, knf -> ntf", coeffs_tensor, T_x)

        return hk_fe
