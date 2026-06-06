"""Random Walk Structural Encodings (RWSE) Transform."""

import time

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class RWSE(BaseTransform):
    r"""Random Walk Structural Encoding (RWSE) transform.

    Parameters
    ----------
    max_pe_dim : int
        Maximum walk length (number of RWSE dimensions).
    concat_to_x : bool, optional
        If True, concatenates the encodings with existing node features.
        Default is True.
    method : str, optional
        Computation method: "dense", "sparse", or "batched".
        "dense" uses standard matrix multiplication (Memory intensive).
        "sparse" uses pure sparse matrix multiplication (Fastest, moderate memory).
        "batched" uses indicator diffusion (Memory-bounded, slightly slower).
        Default is "sparse".
    batch_size : int, optional
        Number of nodes to process simultaneously when using the "batched" method.
        Lower values use less memory but take slightly longer. Default is 2048.
    debug : bool, optional
        If True, runs all methods, catches OOM errors, and prints a detailed
        timing and peak VRAM memory footprint report. Default is False.
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        max_pe_dim: int,
        concat_to_x: bool = True,
        method: str = "batched",
        batch_size: int = 128,
        debug: bool = False,
        **kwargs,
    ):
        self.max_pe_dim = max_pe_dim
        self.concat_to_x = concat_to_x
        self.batch_size = batch_size
        self.debug = debug

        if method not in ["dense", "sparse", "batched"]:
            raise ValueError("Method must be 'dense', 'sparse', or 'batched'.")
        self.method = method

    def forward(self, data: Data) -> Data:
        """Compute the RWSE for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with RWSE added to ``data.x`` or ``data.RWSE``.
        """
        if self.debug:
            print("\n--- RWSE Debug Report ---")
            print(f"Data device:        {data.edge_index.device}")
            # 1. Dense Method
            try:
                t0 = time.time()
                pe_dense, t_dense, mem_dense = self._profile_method(
                    self._compute_dense, data.edge_index, data.num_nodes
                )
                t_dense_total = time.time() - t0
                dense_status = f"{t_dense_total:.4f}s | {mem_dense:.2f} MB"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pe_dense = None
                    dense_status = "OOM (Out Of Memory) 💥"
                else:
                    raise e

            # 2. Sparse Method
            try:
                t0 = time.time()
                pe_sparse, t_sparse, mem_sparse = self._profile_method(
                    self._compute_sparse, data.edge_index, data.num_nodes
                )
                t_sparse_total = time.time() - t0
                sparse_status = f"{t_sparse_total:.4f}s | {mem_sparse:.2f} MB"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pe_sparse = None
                    sparse_status = "OOM (Out Of Memory) 💥"
                else:
                    raise e

            # 3. Batched Method
            t0 = time.time()
            pe_batched, t_batched, mem_batched = self._profile_method(
                self._compute_batched, data.edge_index, data.num_nodes
            )
            t_batched_total = time.time() - t0
            batched_status = f"{t_batched_total:.4f}s | {mem_batched:.2f} MB"

            # Print Report
            print(f"{'Method':<10} | {'Status (Time | Peak VRAM)':<30}")
            print("-" * 45)
            print(f"{'Dense':<10} | {dense_status:<30}")
            print(f"{'Sparse':<10} | {sparse_status:<30}")
            print(f"{'Batched':<10} | {batched_status:<30}")
            print("-" * 45)

            # Comparisons
            if pe_dense is not None and pe_sparse is not None:
                diff_ds = torch.abs(pe_dense - pe_sparse).max().item()
                print(f"Max Abs Error (Dense vs Sparse):   {diff_ds:.6e}")
            if pe_sparse is not None and pe_batched is not None:
                diff_sb = torch.abs(pe_sparse - pe_batched).max().item()
                print(f"Max Abs Error (Sparse vs Batched): {diff_sb:.6e}")
            print("-" * 45 + "\n")

            # Select output based on requested method
            if self.method == "dense" and pe_dense is not None:
                pe = pe_dense
            elif self.method == "batched":
                pe = pe_batched
            else:
                pe = pe_sparse
        else:
            if self.method == "dense":
                pe = self._compute_dense(data.edge_index, data.num_nodes)
            elif self.method == "batched":
                pe = self._compute_batched(data.edge_index, data.num_nodes)
            else:
                pe = self._compute_sparse(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.RWSE = pe

        return data

    def _profile_method(self, func, edge_index: torch.Tensor, num_nodes: int):
        """Helper method to profile execution time and memory (CPU or GPU).

        Parameters
        ----------
        func : callable
            The computation method to profile.
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        tuple
            A tuple containing (result_tensor, time_elapsed_seconds, peak_memory_mb).
        """
        import tracemalloc

        device = edge_index.device
        is_cuda = device.type == "cuda"

        if is_cuda:
            # --- GPU Memory Tracking ---
            torch.cuda.synchronize(device)
            # Record the baseline memory before the function starts
            start_mem = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

            t0 = time.time()
            pe = func(edge_index, num_nodes)
            torch.cuda.synchronize(device)
            t_elapsed = time.time() - t0

            # Get the peak memory reached during the function
            peak_mem = torch.cuda.max_memory_allocated(device)

            # The actual footprint of this method is the Peak minus the Baseline
            mem_mb = (peak_mem - start_mem) / (1024 * 1024)
        else:
            # --- CPU Memory Tracking ---
            tracemalloc.start()

            t0 = time.time()
            pe = func(edge_index, num_nodes)
            t_elapsed = time.time() - t0

            # tracemalloc returns (current_memory, peak_memory) in bytes
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            mem_mb = peak_bytes / (1024 * 1024)

        return pe, t_elapsed, mem_mb

    def _compute_dense(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute RWSE using original dense matrix multiplication.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph of shape ``[2, num_edges]``.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            RWSE return probabilities of shape ``[num_nodes, max_pe_dim]``.
        """
        device = edge_index.device
        if edge_index.numel() == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        deg = degree(edge_index[0], num_nodes=num_nodes).float().to(device)
        deg = torch.where(deg == 0, torch.ones_like(deg), deg)

        adj = torch.zeros(num_nodes, num_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0

        P = adj / deg.unsqueeze(-1)
        rwse = torch.zeros(num_nodes, self.max_pe_dim, device=device)
        P_power = torch.eye(num_nodes, device=device)

        for k in range(1, self.max_pe_dim + 1):
            P_power = P_power @ P
            rwse[:, k - 1] = P_power.diag()

        return rwse.float()

    def _compute_sparse(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute RWSE using optimized PyTorch sparse matrix multiplication.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph of shape ``[2, num_edges]``.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            RWSE return probabilities of shape ``[num_nodes, max_pe_dim]``.
        """
        device = edge_index.device
        if edge_index.numel() == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        row, col = edge_index

        # 1. Compute Out-Degree
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
        deg_inv = 1.0 / deg.clamp_(min=1.0)

        # 2. Transition probabilities: P_{i,j} = 1 / deg(i)
        edge_weight = deg_inv[row]

        # 3. Create Sparse Transition Matrix P
        P = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        rwse = []
        Pk = P

        # Pre-allocate a zero tensor to avoid re-allocating memory inside the loop
        pe_k = torch.zeros(num_nodes, device=device)

        for _ in range(self.max_pe_dim):
            # 1. Grab coordinates and values
            row, col = Pk.indices()
            val = Pk.values()

            # 2. Find the diagonal elements (where row index == col index)
            mask = row == col

            # 3. Drop them into the pre-allocated zero tensor and save
            pe_k.zero_()  # Reset the tensor inplace
            pe_k.scatter_(0, row[mask], val[mask])
            rwse.append(pe_k.clone())  # Clone to save this step's state

            # 4. Advance the random walk
            Pk = torch.sparse.mm(Pk, P)

        return torch.stack(rwse, dim=1)

    def _compute_batched(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute RWSE using memory-bounded batched indicator diffusion.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph of shape ``[2, num_edges]``.
        num_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            RWSE return probabilities of shape ``[num_nodes, max_pe_dim]``.
        """
        device = edge_index.device
        if edge_index.numel() == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        row, col = edge_index

        # 1. Compute Out-Degree and Edge Weights
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
        deg_inv = 1.0 / deg.clamp_(min=1.0)
        edge_weight = deg_inv[row]

        # 2. Create Sparse Transition Matrix P
        P = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        rwse = torch.zeros(num_nodes, self.max_pe_dim, device=device)

        # 3. Process nodes in strict memory-bounded batches
        for start_idx in range(0, num_nodes, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_nodes)
            current_batch_size = end_idx - start_idx

            # Create an indicator matrix for this specific batch: Shape [N, B]
            X = torch.zeros(num_nodes, current_batch_size, device=device)
            batch_nodes = torch.arange(start_idx, end_idx, device=device)
            batch_indices = torch.arange(current_batch_size, device=device)
            X[batch_nodes, batch_indices] = 1.0

            # Diffuse the features K times
            for k in range(self.max_pe_dim):
                # Matrix-Vector multiplication: [N, N] @ [N, B] -> [N, B]
                X = torch.sparse.mm(P, X)

                # Extract the diagonal equivalent for this batch
                return_probs = X[batch_nodes, batch_indices]
                rwse[start_idx:end_idx, k] = return_probs

        return rwse
