"""Custom collate function for on-disk transductive learning.

This module provides a collate function that enables mini-batch training on
large transductive graphs by querying structures on-demand from disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobench.transforms.data_transform import DataTransform

if TYPE_CHECKING:
    from topobench.data.preprocessor.ondisk_transductive import (
        OnDiskTransductivePreprocessor,
    )


class OnDiskTransductiveCollate:
    """Collate function for on-disk transductive mini-batch training.

    This collate function queries topological structures on-demand from
    an OnDiskTransductiveDataset and applies transforms to create proper
    simplicial complex structures, enabling memory-efficient mini-batch
    training on large graphs with arbitrary topological liftings.

    Parameters
    ----------
    ondisk_dataset : OnDiskTransductiveDataset
        The on-disk transductive dataset containing the graph and index.
        If ondisk_dataset.transforms_config is provided, transforms will be
        applied to each batch to create simplicial complex structures as
        individual attributes (x_0, x_1, x_2, hodge_laplacian_0, incidence_1, etc.).
        Use a backbone wrapper (e.g., SCCNNWrapper) to convert these to tuple format
        (x_all, laplacian_all, incidence_all) if needed by your model.
    fully_contained : bool, optional
        If True, only return structures where ALL nodes are in the batch.
        If False, return structures with ANY node in the batch.
        Default: True (recommended for correctness).

    Notes
    -----
    - Memory usage is O(B × D^k) where B is batch size, not O(N × D^k) for full graph
    - Structures are queried from disk index, not computed on-the-fly
    - Compatible with any node sampling strategy
    - For transductive learning, train/val/test splits are handled via node masks

    See Also
    --------
    OnDiskTransductiveDataset : Dataset for on-disk transductive learning.
    NodeBatchSampler : Helper for creating node batches.
    """

    def __init__(
        self,
        ondisk_dataset: OnDiskTransductivePreprocessor,
        fully_contained: bool = True,
    ) -> None:
        """Initialize collate function.

        Parameters
        ----------
        ondisk_dataset : OnDiskTransductiveDataset
            Dataset containing graph and structure index.
        fully_contained : bool, optional
            Whether to require all nodes in structures to be in batch.
            Default: True.
        """
        self.ondisk_dataset = ondisk_dataset
        self.fully_contained = fully_contained
        self.graph_data = ondisk_dataset.graph_data

        # Initialize transform if transforms_config is provided
        self.transform = None
        if ondisk_dataset.transforms_config is not None:
            self.transform = self._instantiate_transform(
                ondisk_dataset.transforms_config
            )

    def __call__(self, batch_items: list[Any]) -> Data:
        """Collate a batch of node indices into a Data object.

        Parameters
        ----------
        batch_items : list
            List of batch items. Each item should be either:
            - A list/tensor of node indices
            - A dict with 'node_ids' key
            - An integer (single node)

        Returns
        -------
        Data
            PyTorch Geometric Data object containing:
            - Node features for sampled nodes
            - Edges connecting sampled nodes
            - Topological structures (queried from index)
            - Original node indices for mapping back

        Raises
        ------
        ValueError
            If batch_items format is not recognized.
        """
        # Extract node indices from batch items
        node_ids = self._extract_node_ids(batch_items)

        # Query structures for this batch from disk
        structures = self.ondisk_dataset.query_batch(
            node_ids, fully_contained=self.fully_contained
        )

        # Build batch Data object
        batch_data = self._build_batch(node_ids, structures)

        return batch_data

    def _extract_node_ids(self, batch_items: list[Any]) -> list[int]:
        """Extract node IDs from batch items.

        Parameters
        ----------
        batch_items : list
            Batch items from dataloader.

        Returns
        -------
        list[int]
            List of node IDs.

        Raises
        ------
        ValueError
            If batch format is not recognized.
        """
        if len(batch_items) == 0:
            return []

        # Handle different batch formats
        first_item = batch_items[0]

        # Format 1: List of node ID lists [[1,2,3], [4,5,6]]
        if isinstance(first_item, (list, tuple)):
            # Flatten if needed
            node_ids = []
            for item in batch_items:
                if isinstance(item, (list, tuple)):
                    node_ids.extend(item)
                else:
                    node_ids.append(int(item))
            return node_ids

        # Format 2: List of tensors [tensor([1,2,3]), tensor([4,5,6])]
        if isinstance(first_item, torch.Tensor):
            node_ids = []
            for item in batch_items:
                if item.numel() > 1:
                    node_ids.extend(item.tolist())
                else:
                    node_ids.append(item.item())
            return node_ids

        # Format 3: List of dicts [{'node_ids': [1,2,3]}, ...]
        if isinstance(first_item, dict) and "node_ids" in first_item:
            node_ids = []
            for item in batch_items:
                ids = item["node_ids"]
                if isinstance(ids, torch.Tensor):
                    node_ids.extend(ids.tolist())
                else:
                    node_ids.extend(ids)
            return node_ids

        # Format 4: List of integers [1, 2, 3, 4]
        if isinstance(first_item, int):
            return [int(x) for x in batch_items]

        # Format 5: Single tensor with node IDs
        if len(batch_items) == 1 and isinstance(batch_items[0], torch.Tensor):
            return batch_items[0].tolist()

        raise ValueError(
            f"Unrecognized batch format. First item type: {type(first_item)}"
        )

    def _build_batch(
        self, node_ids: list[int], structures: list[tuple[int, list[int]]]
    ) -> Data:
        """Build a mini-batch Data object for sampled nodes.

        Parameters
        ----------
        node_ids : list[int]
            Node IDs in this batch.
        structures : list[tuple[int, list[int]]]
            Topological structures from query engine.
            Each tuple is (structure_id, node_list).

        Returns
        -------
        Data
            Mini-batch containing features, edges, and structures.
        """
        # Create node ID to local index mapping
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        num_nodes_batch = len(node_ids)

        # Extract node features for batch
        x = self.graph_data.x[node_ids]

        # Extract edges connecting nodes in batch
        edge_index = self._extract_batch_edges(node_ids, node_to_idx)

        # Extract labels for nodes in batch
        y = None
        if hasattr(self.graph_data, "y") and self.graph_data.y is not None:
            y = self.graph_data.y[node_ids]

        # Extract masks for transductive splits
        train_mask = None
        val_mask = None
        test_mask = None

        if hasattr(self.graph_data, "train_mask"):
            train_mask = self.graph_data.train_mask[node_ids]
        if hasattr(self.graph_data, "val_mask"):
            val_mask = self.graph_data.val_mask[node_ids]
        if hasattr(self.graph_data, "test_mask"):
            test_mask = self.graph_data.test_mask[node_ids]

        # Build Data object
        batch_data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes_batch,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            # Store original node IDs for mapping back
            original_node_ids=torch.tensor(node_ids, dtype=torch.long),
        )

        # If transforms are configured, skip basic structure addition
        # The transform will create proper simplicial complex structures
        if self.transform is None:
            # Add topological structures as basic incidence matrices
            if structures:
                batch_data = self._add_structures_to_batch(
                    batch_data, structures, node_to_idx
                )
        else:
            # Apply transforms to create proper simplicial complex structures
            # The transform operates on the graph structure (nodes + edges)
            # and creates x_all, laplacian_all, incidence_all
            batch_data = self.transform(batch_data)

        return batch_data

    def _extract_batch_edges(
        self, node_ids: list[int], node_to_idx: dict[int, int]
    ) -> torch.Tensor:
        """Extract edges connecting nodes in the batch.

        Parameters
        ----------
        node_ids : list[int]
            Node IDs in batch.
        node_to_idx : dict[int, int]
            Mapping from global node ID to local batch index.

        Returns
        -------
        torch.Tensor
            Edge index with shape (2, num_edges) in local indices.
        """
        node_set = set(node_ids)
        edge_index = self.graph_data.edge_index

        # Find edges where both endpoints are in batch
        mask = torch.tensor(
            [
                (edge_index[0, i].item() in node_set)
                and (edge_index[1, i].item() in node_set)
                for i in range(edge_index.shape[1])
            ],
            dtype=torch.bool,
        )

        batch_edges = edge_index[:, mask]

        # Convert to local indices
        local_edges = torch.zeros_like(batch_edges)
        for i in range(batch_edges.shape[1]):
            local_edges[0, i] = node_to_idx[batch_edges[0, i].item()]
            local_edges[1, i] = node_to_idx[batch_edges[1, i].item()]

        return local_edges

    def _add_structures_to_batch(
        self,
        batch_data: Data,
        structures: list[tuple[int, list[int]]],
        node_to_idx: dict[int, int],
    ) -> Data:
        """Add topological structures to batch Data object.

        Parameters
        ----------
        batch_data : Data
            Batch data to augment.
        structures : list[tuple[int, list[int]]]
            Structures to add (structure_id, node_list).
        node_to_idx : dict[int, int]
            Mapping from global to local indices.

        Returns
        -------
        Data
            Batch data with structures added.
        """
        # Group structures by size
        structures_by_size: dict[int, list[list[int]]] = {}

        for _struct_id, node_list in structures:
            size = len(node_list)
            # Convert to local indices
            local_indices = [
                node_to_idx[n] for n in node_list if n in node_to_idx
            ]

            # Only add if all nodes are in batch (shouldn't happen with fully_contained=True)
            if len(local_indices) == size:
                if size not in structures_by_size:
                    structures_by_size[size] = []
                structures_by_size[size].append(local_indices)

        # Add structures as incidence matrices (following TopoBench conventions)
        for size, struct_list in structures_by_size.items():
            if struct_list:
                # Create incidence matrix: [num_structures, num_nodes]
                num_structs = len(struct_list)
                incidence = torch.zeros(
                    (num_structs, batch_data.num_nodes), dtype=torch.float32
                )

                for i, nodes in enumerate(struct_list):
                    for node_idx in nodes:
                        incidence[i, node_idx] = 1.0

                # Store as x_{size-1} following TopoBench convention
                # (e.g., triangles (size 3) stored as x_2)
                setattr(batch_data, f"x_{size - 1}", incidence)

        return batch_data

    def _instantiate_transform(
        self, transforms_config: DictConfig
    ) -> torch_geometric.transforms.Compose:
        """Instantiate transform from configuration.

        Transforms are applied at batch-time to create proper simplicial
        complex structures from the queried topological structures. The transform
        creates individual attributes (x_0, x_1, x_2, hodge_laplacian_0,
        down_laplacian_1, incidence_1, etc.) on the batch Data object.

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.

        Returns
        -------
        torch_geometric.transforms.Compose
            Composed transform object.

        Notes
        -----
        Unlike inductive learning where transforms are applied offline during
        preprocessing, transductive learning applies transforms online during
        batch collation. This is necessary because:
        1. The full graph is too large to lift entirely into memory
        2. Transforms are applied to mini-batch subgraphs (O(batch_size) memory)
        3. Each batch gets properly lifted structures for model consumption
        """
        # Handle nested liftings config (for compatibility)
        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        # Check if single or multiple transforms
        if "transform_name" in transforms_config:
            # Single transform
            pre_transforms_dict = {
                transforms_config.transform_name: DataTransform(
                    **transforms_config
                )
            }
        else:
            # Multiple transforms
            pre_transforms_dict = {
                key: DataTransform(**value)
                for key, value in transforms_config.items()
            }

        # Return composed transform
        return torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
        )


class NodeBatchSampler:
    """Helper class for sampling node batches for transductive learning.

    This sampler yields batches of node indices for mini-batch training
    on transductive graphs.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    batch_size : int
        Number of nodes per batch.
    shuffle : bool, optional
        Whether to shuffle nodes before batching (default: False).
    mask : torch.Tensor, optional
        Boolean mask indicating which nodes to sample (e.g., train_mask).
        If provided, only samples from masked nodes (default: None).
    """

    def __init__(
        self,
        num_nodes: int,
        batch_size: int,
        shuffle: bool = False,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Initialize node batch sampler."""
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Determine which nodes to sample
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                self.node_indices = torch.where(mask)[0].tolist()
            else:
                self.node_indices = [i for i, m in enumerate(mask) if m]
        else:
            self.node_indices = list(range(num_nodes))

        self.num_samples = len(self.node_indices)

    def __iter__(self):
        """Iterate over node batches."""
        indices = self.node_indices.copy()

        if self.shuffle:
            import random

            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            yield batch

    def __len__(self) -> int:
        """Return number of batches.

        Returns
        -------
        int
            Number of batches that will be yielded.
        """
        return (self.num_samples + self.batch_size - 1) // self.batch_size
