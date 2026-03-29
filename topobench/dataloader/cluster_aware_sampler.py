"""Cluster-aware node sampling for community-preserving mini-batch training.

This module provides samplers that preserve graph community structure during
mini-batch training, enabling dense neighborhoods and better message passing
while maintaining memory efficiency.
"""

import random

import torch
from torch_geometric.data import Data


class ClusterAwareNodeSampler:
    """Node sampler that preserves community structure.

    Instead of random sampling, samples entire clusters/communities to keep
    dense neighborhoods intact. This improves message passing quality and
    preserves community-level patterns during training.

    Parameters
    ----------
    graph_data : Data
        The full graph data with edge_index.
    batch_size : int
        Target number of nodes per batch.
    clustering_method : str, optional
        Clustering algorithm to use. Options:
        - "louvain": Fast community detection (default)
        - "metis": METIS graph partitioning
        - "leiden": Leiden algorithm (improved Louvain)
        - "label_propagation": Fast label propagation
        - "random": Random partitioning (for baseline)
    num_clusters : int, optional
        Target number of clusters. If None, determined automatically.
    shuffle : bool, optional
        Whether to shuffle cluster order each epoch (default: True).
    mask : torch.Tensor, optional
        Boolean mask for train/val/test split. Only nodes with mask=True
        are included in sampling.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> # Basic usage with Louvain clustering
    >>> sampler = ClusterAwareNodeSampler(
    ...     graph_data=data,
    ...     batch_size=1024,
    ...     clustering_method="louvain",
    ...     shuffle=True,
    ...     mask=data.train_mask
    ... )
    >>>
    >>> # Use with OnDiskTransductiveCollate
    >>> collate_fn = OnDiskTransductiveCollate(preprocessor)
    >>> for batch_nodes in sampler:
    ...     batch = collate_fn([batch_nodes])
    ...     # Batch has dense community structure + complete topology!
    """

    def __init__(
        self,
        graph_data: Data,
        batch_size: int,
        clustering_method: str = "louvain",
        num_clusters: int | None = None,
        shuffle: bool = True,
        mask: torch.Tensor | None = None,
        seed: int | None = None,
    ):
        self.graph_data = graph_data
        self.batch_size = batch_size
        self.clustering_method = clustering_method
        self.num_clusters = num_clusters
        self.shuffle = shuffle
        self.mask = mask
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Compute clusters (one-time cost)
        print(f"Computing {clustering_method} clustering...")
        self.clusters = self._compute_clusters()
        print(f"Created {len(self.clusters)} clusters")

        # Filter clusters by mask if provided
        if mask is not None:
            self.clusters = self._filter_clusters_by_mask()
            print(f"Filtered to {len(self.clusters)} clusters with mask")

        # Pre-compute cluster sizes
        self.cluster_sizes = [len(c) for c in self.clusters]

        # Statistics
        total_nodes = sum(self.cluster_sizes)
        avg_size = total_nodes / len(self.clusters) if self.clusters else 0
        print(f"Total nodes: {total_nodes}, Avg cluster size: {avg_size:.1f}")

    def _compute_clusters(self) -> list[list[int]]:
        """Compute graph clustering using specified method.

        Returns
        -------
        list[list[int]]
            List of clusters, each cluster is a list of node IDs.
        """
        method = self.clustering_method.lower()

        if method == "louvain":
            return self._louvain_clustering()
        elif method == "metis":
            return self._metis_clustering()
        elif method == "leiden":
            return self._leiden_clustering()
        elif method == "label_propagation":
            return self._label_propagation_clustering()
        elif method == "random":
            return self._random_clustering()
        else:
            raise ValueError(
                f"Unknown clustering method: {method}. "
                f"Choose from: louvain, metis, leiden, label_propagation, random"
            )

    def _louvain_clustering(self) -> list[list[int]]:
        """Louvain community detection algorithm.

        Fast and effective for most graphs. Good default choice.

        Returns
        -------
        list[list[int]]
            List of clusters detected by Louvain algorithm.
        """
        try:
            import networkx as nx
            from networkx.algorithms import community
        except ImportError as err:
            raise ImportError(
                "NetworkX required for Louvain clustering. "
                "Install with: pip install networkx"
            ) from err

        # Convert to NetworkX graph
        edge_index = self.graph_data.edge_index
        G = nx.Graph()
        G.add_nodes_from(range(self.graph_data.num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Run Louvain
        communities = community.louvain_communities(G, seed=self.seed)

        # Convert to list of lists
        clusters = [list(comm) for comm in communities]
        return clusters

    def _metis_clustering(self) -> list[list[int]]:
        """METIS graph partitioning.

        Same algorithm used by Cluster-GCN. Guarantees balanced partitions
        and connected components.

        Returns
        -------
        list[list[int]]
            List of partitions created by METIS.
        """
        try:
            import networkx as nx
            import nxmetis
        except ImportError as err:
            raise ImportError(
                "nxmetis required for METIS clustering. "
                "Install with: pip install networkx-metis"
            ) from err

        # Convert to NetworkX graph
        edge_index = self.graph_data.edge_index
        G = nx.Graph()
        G.add_nodes_from(range(self.graph_data.num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Determine number of partitions
        if self.num_clusters is None:
            # Heuristic: aim for ~500-1000 nodes per cluster
            self.num_clusters = max(2, self.graph_data.num_nodes // 750)

        # Run METIS
        (_edgecuts, parts) = nxmetis.partition(G, self.num_clusters)

        # Convert partition assignment to clusters
        clusters = [[] for _ in range(self.num_clusters)]
        for node_id, partition_id in enumerate(parts):
            clusters[partition_id].append(node_id)

        # Remove empty clusters
        clusters = [c for c in clusters if len(c) > 0]
        return clusters

    def _leiden_clustering(self) -> list[list[int]]:
        """Leiden community detection algorithm.

        Improved version of Louvain with better quality guarantees.

        Returns
        -------
        list[list[int]]
            List of clusters detected by Leiden algorithm.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError as err:
            raise ImportError(
                "igraph and leidenalg required for Leiden clustering. "
                "Install with: pip install igraph leidenalg"
            ) from err

        # Convert to igraph
        edge_index = self.graph_data.edge_index
        edges = edge_index.t().tolist()
        g = ig.Graph(n=self.graph_data.num_nodes, edges=edges, directed=False)

        # Run Leiden
        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, seed=self.seed
        )

        # Convert to list of lists
        clusters = [list(comm) for comm in partition]
        return clusters

    def _label_propagation_clustering(self) -> list[list[int]]:
        """Fast label propagation algorithm.

        Very fast but may produce lower quality communities than Louvain.
        Good for quick experiments or very large graphs.

        Returns
        -------
        list[list[int]]
            List of clusters detected by label propagation.
        """
        try:
            import networkx as nx
        except ImportError as err:
            raise ImportError(
                "NetworkX required for label propagation. "
                "Install with: pip install networkx"
            ) from err

        # Convert to NetworkX graph
        edge_index = self.graph_data.edge_index
        G = nx.Graph()
        G.add_nodes_from(range(self.graph_data.num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Run label propagation
        communities = nx.algorithms.community.label_propagation_communities(G)

        # Convert to list of lists
        clusters = [list(comm) for comm in communities]
        return clusters

    def _random_clustering(self) -> list[list[int]]:
        """Random partitioning (baseline).

        Returns
        -------
        list[list[int]]
            List of random partitions.
        """
        # Determine number of clusters
        if self.num_clusters is None:
            self.num_clusters = max(2, self.graph_data.num_nodes // 750)

        # Randomly assign nodes to clusters
        nodes = list(range(self.graph_data.num_nodes))
        random.shuffle(nodes)

        cluster_size = len(nodes) // self.num_clusters
        clusters = []
        for i in range(self.num_clusters):
            start = i * cluster_size
            end = (
                start + cluster_size
                if i < self.num_clusters - 1
                else len(nodes)
            )
            clusters.append(nodes[start:end])

        return clusters

    def _filter_clusters_by_mask(self) -> list[list[int]]:
        """Filter clusters to only include nodes in mask.

        Returns
        -------
        list[list[int]]
            List of filtered clusters.
        """
        filtered_clusters = []
        for cluster in self.clusters:
            filtered = [node for node in cluster if self.mask[node]]
            if len(filtered) > 0:
                filtered_clusters.append(filtered)
        return filtered_clusters

    def __iter__(self):
        """Iterate over batches by sampling clusters.

        Yields batches by selecting clusters and combining them until
        batch_size is reached. Preserves community structure within batches.
        """
        # Get cluster order
        cluster_indices = list(range(len(self.clusters)))
        if self.shuffle:
            random.shuffle(cluster_indices)

        current_batch = []

        for cluster_idx in cluster_indices:
            cluster_nodes = self.clusters[cluster_idx].copy()

            # Shuffle nodes within cluster (optional, for variation)
            if self.shuffle:
                random.shuffle(cluster_nodes)

            current_batch.extend(cluster_nodes)

            # Yield batch when we reach batch_size
            while len(current_batch) >= self.batch_size:
                batch = current_batch[: self.batch_size]
                yield batch
                current_batch = current_batch[self.batch_size :]

        # Yield remaining nodes as final batch
        if len(current_batch) > 0:
            yield current_batch

    def __len__(self):
        """Number of batches per epoch.

        Returns
        -------
        int
            Number of batches.
        """
        total_nodes = sum(self.cluster_sizes)
        return (total_nodes + self.batch_size - 1) // self.batch_size


class HybridNodeSampler:
    """Flexible sampler supporting multiple strategies.

    Combines random sampling (for exploration) with cluster sampling
    (for community preservation) in a single interface.

    Parameters
    ----------
    graph_data : Data
        The full graph data with edge_index.
    batch_size : int
        Target number of nodes per batch.
    strategy : str, optional
        Sampling strategy. Options:
        - "cluster": Only cluster-based sampling
        - "random": Only random sampling
        - "hybrid": Mix of both (default)
    cluster_ratio : float, optional
        For hybrid strategy: fraction of batches that use cluster sampling.
        Default: 0.7 (70% cluster, 30% random).
    clustering_method : str, optional
        Clustering algorithm (for cluster/hybrid strategies).
    **kwargs : dict
        Additional arguments passed to ClusterAwareNodeSampler.
    """

    def __init__(
        self,
        graph_data: Data,
        batch_size: int,
        strategy: str = "hybrid",
        cluster_ratio: float = 0.7,
        clustering_method: str = "louvain",
        **kwargs,
    ):
        from .ondisk_transductive_collate import NodeBatchSampler

        self.strategy = strategy.lower()
        self.cluster_ratio = cluster_ratio

        if self.strategy not in ["cluster", "random", "hybrid"]:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Choose from: cluster, random, hybrid"
            )

        # Initialize samplers based on strategy
        if self.strategy in ["cluster", "hybrid"]:
            self.cluster_sampler = ClusterAwareNodeSampler(
                graph_data=graph_data,
                batch_size=batch_size,
                clustering_method=clustering_method,
                **kwargs,
            )

        if self.strategy in ["random", "hybrid"]:
            self.random_sampler = NodeBatchSampler(
                num_nodes=graph_data.num_nodes,
                batch_size=batch_size,
                shuffle=kwargs.get("shuffle", True),
                mask=kwargs.get("mask"),
            )

    def __iter__(self):
        """Iterate based on selected strategy."""
        if self.strategy == "random":
            yield from self.random_sampler
        elif self.strategy == "cluster":
            yield from self.cluster_sampler
        elif self.strategy == "hybrid":
            # Interleave cluster and random batches
            cluster_iter = iter(self.cluster_sampler)
            random_iter = iter(self.random_sampler)

            # Determine order: cluster_ratio of batches use cluster sampling
            total_batches = len(self)
            cluster_count = int(total_batches * self.cluster_ratio)
            random_count = total_batches - cluster_count

            # Create schedule: mix of 'C' (cluster) and 'R' (random)
            schedule = ["C"] * cluster_count + ["R"] * random_count
            random.shuffle(schedule)

            for batch_type in schedule:
                try:
                    if batch_type == "C":
                        yield next(cluster_iter)
                    else:
                        yield next(random_iter)
                except StopIteration:
                    break

    def __len__(self):
        """Number of batches per epoch.

        Returns
        -------
        int
            Number of batches.
        """
        if self.strategy == "cluster":
            return len(self.cluster_sampler)
        elif self.strategy == "random":
            return len(self.random_sampler)
        else:  # hybrid
            # Use average of both
            return (len(self.cluster_sampler) + len(self.random_sampler)) // 2
