"""Personalized Page Rank Feature Encoding (PPRFE) Transform."""

import time

import numpy as np
import omegaconf
import torch
from scipy.linalg import inv
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops, degree, to_dense_adj


class PPRFE(BaseTransform):
    r"""
    Personalized Page Rank Feature Encodings (PPRFE) transform.

    Parameters
    ----------
    alpha_param_PPRFE : tuple of float
        Tuple specifying the start and end teleport probabilities (alpha values).
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features.
        Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min".
        Default is "mean".
    self_loop : bool, optional
        If True, adds self-loops to the adjacency matrix.
        Default is True.
    method : str, optional
        Computation method: "exact" or "approx".
        Default is "approx".
    appnp_K : int, optional
        Number of polynomial expansion terms (propagation steps) for the approx method.
        Higher means more global information but slower.
        Default is 20.
    debug : bool, optional
        If True, runs both methods and prints error/timing metrics.
        Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        alpha_param_PPRFE: tuple,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        self_loop: bool = True,
        method: str = "approx",
        appnp_K: int = 20,
        debug: bool = False,
        **kwargs,
    ):
        self.alpha_param_PPRFE = alpha_param_PPRFE
        self.concat_to_x = concat_to_x
        self.self_loop = self_loop

        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(f"Unknown aggregation '{aggregation}'.")
        self.aggregation = aggregation

        if method not in ["exact", "approx"]:
            raise ValueError("Method must be 'exact' or 'approx'.")
        self.method = method
        self.appnp_K = appnp_K
        self.debug = debug

        if (
            isinstance(alpha_param_PPRFE, (list, tuple))
            or type(alpha_param_PPRFE) is omegaconf.listconfig.ListConfig
        ):
            self.fe_dim = alpha_param_PPRFE[1]
        else:
            self.fe_dim = alpha_param_PPRFE

    def forward(self, data: Data) -> Data:
        """Compute the PPR feature encodings for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with PPR feature encodings added.
        """
        if data.x is None:
            raise ValueError(
                "PPRFE requires node features (data.x cannot be None)"
            )

        fe = self._compute_pprfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.PPRFE = fe

        return data

    def _compute_pprfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute PPR feature encodings.

        Computes PPR diffusion at multiple alpha values and aggregates
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
            PPR feature encodings of shape ``[num_nodes, fe_dim]``.
        """
        device = edge_index.device

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.fe_dim, device=device)

        start, num_alphas = (
            self.alpha_param_PPRFE[0],
            self.alpha_param_PPRFE[1],
        )
        alpha_values = np.linspace(start, 0.9, num_alphas)

        if self.debug:
            print("\n--- PPRFE Debug Report ---")
            print(f"Data device:        {device}")
            # Exact
            t0 = time.time()
            fe_exact = self._compute_exact(
                x, edge_index, num_nodes, alpha_values, device
            )
            t_exact = time.time() - t0
            print(f"Exact compute time:  {t_exact:.4f}s")

            # Approx
            t0 = time.time()
            fe_approx = self._compute_approx(
                x, edge_index, num_nodes, alpha_values, device
            )
            t_approx = time.time() - t0
            print(
                f"Approx compute time: {t_approx:.4f}s (Polynomial Order K: {self.appnp_K})"
            )

            # Compare
            diff = torch.abs(fe_exact - fe_approx)
            reldiff = diff / (torch.abs(fe_exact) + 1e-8)
            mean_reldiff = reldiff.mean().item()

            print(f"Speedup Factor:      {t_exact / t_approx:.2f}x")
            print(f"Mean Abs Error:      {diff.mean().item():.6e}")
            print(f"Max Abs Error:       {diff.max().item():.6e}")
            print(f"Mean Rel Error:      {mean_reldiff:.6e}")
            print("--------------------------\n")

            fe_raw = fe_exact if self.method == "exact" else fe_approx
        else:
            if self.method == "exact":
                fe_raw = self._compute_exact(
                    x, edge_index, num_nodes, alpha_values, device
                )
            else:
                fe_raw = self._compute_approx(
                    x, edge_index, num_nodes, alpha_values, device
                )

        # Aggregate over features
        agg_fn = getattr(fe_raw, self._AGG_FN_MAP[self.aggregation])
        fe_agg = agg_fn(dim=-1)

        if torch.any(torch.isnan(fe_agg)):
            raise ValueError("PPRFE contains NaNs")

        return fe_agg.float()

    def _compute_exact(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        alpha_values: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute exact O(N^3) dense matrix inversion method for PPR.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.
        alpha_values : numpy.ndarray
            Array of teleport probabilities.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        torch.Tensor
            Exact PPR feature encodings.
        """
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        adj_np = adj.cpu().numpy().astype(np.float64)

        if self.self_loop:
            adj_np = adj_np + np.eye(num_nodes)

        deg = np.sum(adj_np, axis=1)
        deg_safe = np.where(deg > 0, deg, 1.0)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
        adj_norm = deg_inv_sqrt @ adj_np @ deg_inv_sqrt

        x_np = x.detach().cpu().numpy().astype(np.float64)
        ppr_fe = []
        identity = np.eye(num_nodes)

        for alpha in alpha_values:
            ppr_matrix = alpha * inv(identity - (1 - alpha) * adj_norm)
            x_ppr = ppr_matrix @ x_np
            ppr_fe.append(torch.from_numpy(x_ppr).float().to(device))

        return torch.stack(ppr_fe, dim=1)

    def _compute_approx(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        alpha_values: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute fast APPNP polynomial approximation using sparse matrix multiplication.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.
        alpha_values : numpy.ndarray
            Array of teleport probabilities.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        torch.Tensor
            Approximated PPR feature encodings.
        """
        # 1. Add self loops if needed
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=device
        )
        if self.self_loop:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
            )

        # 2. Compute symmetric degree normalization: D^{-1/2} A D^{-1/2}
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # 3. Create sparse adjacency tensor
        A_tilde = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        # 4. Precompute the graph diffusions T_k = A_tilde^k * X
        T_x = [x]
        for _ in range(self.appnp_K):
            T_x.append(torch.sparse.mm(A_tilde, T_x[-1]))

        # Stack to shape [K+1, N, F]
        T_x_tensor = torch.stack(T_x, dim=0)

        # 5. Vectorized alpha coefficients
        # Shape alpha_tensor: [num_alphas]
        alpha_tensor = torch.tensor(
            alpha_values, dtype=torch.float32, device=device
        )
        # Shape k_tensor: [K+1]
        k_tensor = torch.arange(
            self.appnp_K + 1, dtype=torch.float32, device=device
        )

        # Calculate coefficients: alpha * (1 - alpha)^k
        # Expand dims to calculate outer product-like matrix. Result shape: [num_alphas, K+1]
        alphas_expanded = alpha_tensor.unsqueeze(1)
        ks_expanded = k_tensor.unsqueeze(0)
        coeffs = alphas_expanded * (1.0 - alphas_expanded) ** ks_expanded

        # 6. Apply coefficients to diffusions via Einsum
        # coeffs: [num_alphas, K+1] -> 'ak'
        # T_x_tensor: [K+1, N, F] -> 'knf'
        # Output: [N, num_alphas, F] -> 'nak'
        ppr_fe = torch.einsum("ak, knf -> naf", coeffs, T_x_tensor)

        return ppr_fe
