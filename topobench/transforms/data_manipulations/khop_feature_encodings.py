"""K-hop feature Encoding (KFE) for Hasse graphs Transform."""

import time

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_dense_adj


class KHopFE(BaseTransform):
    r"""
    K-hop Feature Encodings (KHopFE) transform.

    Parameters
    ----------
    max_hop : int
        The maximum hop neighbourhood.
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features in
        ``data.x``. If ``data.x`` is None, creates it.
        Default is True.
    aggregation : str, optional
        Aggregation function to reduce over the feature dimension.
        Options: "mean", "sum", "max", "min".
        Default is "mean".
    method : str, optional
        Computation method: "dense" or "sparse".
        Default is "sparse".
    debug : bool, optional
        If True, runs both methods and prints error/timing metrics.
        Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    _AGG_FN_MAP = {"mean": "mean", "sum": "sum", "max": "amax", "min": "amin"}

    def __init__(
        self,
        max_hop: int,
        concat_to_x: bool = True,
        aggregation: str = "mean",
        method: str = "sparse",
        debug: bool = False,
        **kwargs,
    ):
        self.concat_to_x = concat_to_x
        self.max_hop = (
            max_hop - 1
        )  # The 0-th hop is always the features themselves
        self.method = method
        self.debug = debug

        if aggregation not in self._AGG_FN_MAP:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(self._AGG_FN_MAP.keys())}"
            )
        self.aggregation = aggregation

        if method not in ["dense", "sparse"]:
            raise ValueError("Method must be 'dense' or 'sparse'.")

    def forward(self, data: Data) -> Data:
        """Compute the K-hop feature encodings for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with K-hop feature encodings added.
        """
        if data.x is None:
            raise ValueError(
                "KHopFE requires node features (data.x cannot be None)"
            )

        fe = self._compute_khopfe(data.x, data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = fe
            else:
                data.x = torch.cat([data.x, fe], dim=-1)
        else:
            data.KHopFE = fe

        return data

    def _compute_khopfe(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Internal method to compute K-hop feature encodings.

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
            K-hop feature encodings of shape [num_nodes, max_hop, feature_dim].
        """
        device = edge_index.device
        x = x.to(device)

        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_hop, device=device)

        if self.debug:
            print("\n--- KHopFE Debug Report ---")
            is_cuda = device.type == "cuda"
            print(f"Data device:        {device}")

            # Helper function to track both time and peak GPU memory
            def _track_execution(func):
                if is_cuda:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    mem_start = torch.cuda.memory_allocated(device)

                t0 = time.time()
                result = func(x, edge_index, num_nodes, device)

                if is_cuda:
                    torch.cuda.synchronize(device)
                    t_elapsed = time.time() - t0
                    # Calculate peak memory used during the function call
                    mem_peak = torch.cuda.max_memory_allocated(device)
                    mem_used = mem_peak - mem_start
                else:
                    t_elapsed = time.time() - t0
                    mem_used = 0

                return result, t_elapsed, mem_used

            # Exact (Dense)
            fe_dense, t_dense, mem_dense = _track_execution(
                self._compute_dense
            )
            print(f"Dense compute time:  {t_dense:.4f}s")
            if is_cuda:
                print(f"Dense peak memory:   {mem_dense / (1024**2):.2f} MB")

            # Approx (Sparse)
            fe_sparse, t_sparse, mem_sparse = _track_execution(
                self._compute_sparse
            )
            print(f"Sparse compute time: {t_sparse:.4f}s")
            if is_cuda:
                print(f"Sparse peak memory:  {mem_sparse / (1024**2):.2f} MB")

            # Compare
            diff = torch.abs(fe_dense - fe_sparse)
            speedup = (t_dense / t_sparse) if t_sparse > 0 else float("inf")
            print(f"\nSpeedup Factor (Time): {speedup:.2f}x")

            if is_cuda and mem_sparse > 0:
                mem_ratio = mem_dense / mem_sparse
                print(
                    f"Memory Factor (VRAM):  {mem_ratio:.2f}x (Dense uses {mem_ratio:.1f}x more memory)"
                )

            print(f"Mean Abs Error:        {diff.mean().item():.6e}")
            print(f"Max Abs Error:         {diff.max().item():.6e}")
            print("---------------------------\n")

            fe_raw = fe_dense if self.method == "dense" else fe_sparse
        else:
            if self.method == "dense":
                fe_raw = self._compute_dense(x, edge_index, num_nodes, device)
            else:
                fe_raw = self._compute_sparse(x, edge_index, num_nodes, device)

        # Aggregate over features
        agg_fn = getattr(fe_raw, self._AGG_FN_MAP[self.aggregation])
        khop_fe = agg_fn(dim=-1)

        if torch.any(torch.isnan(khop_fe)):
            raise ValueError("KHopFE contains NaNs")

        return khop_fe.float()

    def _compute_dense(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute KHopFE using original dense adjacency matrices.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        torch.Tensor
            Dense computation of K-hop feature encodings.
        """
        khop_fe = []
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)

        # Symmetric norm adjacency matrix
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.diagflat(torch.pow(deg + 1e-8, -0.5))
        A_norm = deg_inv_sqrt @ A @ deg_inv_sqrt

        curr_x = x
        for _ in range(self.max_hop):
            curr_x = A_norm @ curr_x
            khop_fe.append(curr_x)

        return torch.stack(khop_fe, dim=1)

    def _compute_sparse(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute KHopFE using optimized pure PyTorch sparse tensors.

        Parameters
        ----------
        x : torch.Tensor
            Node features of the graph.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        torch.Tensor
            Sparse computation of K-hop feature encodings.
        """
        khop_fe = []
        row, col = edge_index

        # 1. Compute node degrees using the row indices (out-degree)
        deg = degree(row, num_nodes, dtype=torch.float32)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

        # 2. Compute symmetric normalized edge weights: (D^-0.5)[i] * (D^-0.5)[j]
        # Since A[i,j] is 1 for existing edges, the weight is just the product of the inverse sqrts
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 3. Create the sparse symmetric normalized adjacency matrix
        A_sparse = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        # 4. Iteratively propagate features via Sparse Matrix-Matrix multiplication
        curr_x = x
        for _ in range(self.max_hop):
            curr_x = torch.sparse.mm(A_sparse, curr_x)
            khop_fe.append(curr_x)

        return torch.stack(khop_fe, dim=1)
