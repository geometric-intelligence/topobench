"""Extended context collate for transductive learning with existing samplers.

This module implements collate functions that expand node samples to include
additional context nodes needed for complete structures while maintaining
backward compatibility with existing samplers.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch_geometric.data import Data

from topobench.data.preprocessor.ondisk_transductive import (
    OnDiskTransductivePreprocessor,
)


class ExtendedContextCollate:
    """Collate function with context expansion for structure completeness.

    This collate function works with existing node samplers (Louvain, METIS, etc.)
    and expands the sampled nodes to include additional "context" nodes needed
    to complete topological structures.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built structure index.
    max_expansion_ratio : float, optional
        Maximum ratio of total nodes to core nodes (default: 1.5).
        Example: 1.5 means batch can expand from 1000 to 1500 nodes.
    transform : Callable, optional
        Topological transform to apply (default: None).
    include_structure_metadata : bool, optional
        Include structure metadata in batch (default: True).
    filter_on_expansion : bool, optional
        Filter structures if expansion would exceed ratio (default: True).

    Attributes
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        The preprocessor instance.
    max_expansion_ratio : float
        Maximum expansion ratio.
    transform : Callable or None
        Transform to apply.
    """

    def __init__(
        self,
        preprocessor: OnDiskTransductivePreprocessor,
        max_expansion_ratio: float = 1.5,
        transform: Callable | None = None,
        include_structure_metadata: bool = True,
        filter_on_expansion: bool = True,
    ) -> None:
        """Initialize extended context collate function.

        Parameters
        ----------
        preprocessor : OnDiskTransductivePreprocessor
            Preprocessor with structure index.
        max_expansion_ratio : float, optional
            Max expansion ratio (default: 1.5).
        transform : Callable, optional
            Transform to apply (default: None).
        include_structure_metadata : bool, optional
            Include structure metadata (default: True).
        filter_on_expansion : bool, optional
            Filter structures on expansion (default: True).
        """
        self.preprocessor = preprocessor
        self.max_expansion_ratio = max_expansion_ratio
        self.transform = transform
        self.include_structure_metadata = include_structure_metadata
        self.filter_on_expansion = filter_on_expansion

        if not hasattr(preprocessor, "query_engine"):
            raise ValueError(
                "Preprocessor must have query_engine. "
                "Call preprocessor.build_index() first."
            )

        self.query_engine = preprocessor.query_engine
        self.graph_data = preprocessor.graph_data

    def __call__(self, batch_items: list[Any]) -> Data:
        """Build batch with extended context.

        Parameters
        ----------
        batch_items : list
            List containing batch data. Typically [node_tensor] from sampler.

        Returns
        -------
        Data
            PyG Data object with extended context and core_mask.
        """
        # Extract core node IDs
        core_node_ids = self._extract_node_ids(batch_items)

        if len(core_node_ids) == 0:
            return self._create_empty_batch()

        # Query structures involving core nodes (allow partial)
        structures = self.query_engine.query_batch(
            core_node_ids, fully_contained=False
        )

        # Gather all nodes needed for structures
        structure_node_set = set()
        for _, nodes in structures:
            structure_node_set.update(nodes)

        # Check expansion ratio and filter if needed
        max_nodes = int(len(core_node_ids) * self.max_expansion_ratio)

        if len(structure_node_set) > max_nodes and self.filter_on_expansion:
            # Filter structures to stay within budget
            structures = self._filter_structures_by_budget(
                structures, core_node_ids, max_nodes
            )
            # Recompute structure nodes
            structure_node_set = set()
            for _, nodes in structures:
                structure_node_set.update(nodes)

        # Combine core and structure nodes
        extended_node_set = set(core_node_ids) | structure_node_set
        batch_node_ids = sorted(list(extended_node_set))

        # Build subgraph
        batch_data = self._extract_subgraph(batch_node_ids)

        # Add core mask
        core_mask = torch.zeros(len(batch_node_ids), dtype=torch.bool)
        node_id_to_idx = {nid: idx for idx, nid in enumerate(batch_node_ids)}
        for core_nid in core_node_ids:
            core_mask[node_id_to_idx[core_nid]] = True

        batch_data.core_mask = core_mask
        batch_data.expansion_ratio = len(batch_node_ids) / len(core_node_ids)

        # Add structure metadata
        if self.include_structure_metadata:
            # Remap structure nodes to batch-local indices
            remapped_structures = []
            for struct_id, nodes in structures:
                # Check if all nodes are in batch (they should be)
                if all(n in node_id_to_idx for n in nodes):
                    remapped_nodes = [node_id_to_idx[n] for n in nodes]
                    remapped_structures.append((struct_id, remapped_nodes))

            batch_data.precomputed_structures = remapped_structures
            batch_data.num_structures = len(remapped_structures)

        # Store original node IDs
        batch_data.batch_node_ids = torch.tensor(
            batch_node_ids, dtype=torch.long
        )
        batch_data.core_node_ids = torch.tensor(
            core_node_ids, dtype=torch.long
        )

        # Apply transform if provided
        if self.transform is not None:
            batch_data = self.transform(batch_data)

        return batch_data

    def _extract_node_ids(self, batch_items: list[Any]) -> list[int]:
        """Extract node IDs from batch items.

        Parameters
        ----------
        batch_items : list
            Batch items from sampler (typically [node_tensor]).

        Returns
        -------
        list of int
            Node IDs.
        """
        if not batch_items:
            return []

        # Assume first item is node tensor or list
        item = batch_items[0]

        if torch.is_tensor(item):
            return item.cpu().tolist()
        elif isinstance(item, (list, tuple)):
            return list(item)
        else:
            raise ValueError(f"Unsupported batch item type: {type(item)}")

    def _filter_structures_by_budget(
        self,
        structures: list[tuple[int, list[int]]],
        core_nodes: list[int],
        max_nodes: int,
    ) -> list[tuple[int, list[int]]]:
        """Filter structures to fit within node budget.

        Parameters
        ----------
        structures : list of (int, list of int)
            Candidate structures.
        core_nodes : list of int
            Core node IDs.
        max_nodes : int
            Maximum total nodes allowed.

        Returns
        -------
        list of (int, list of int)
            Filtered structures.

        Notes
        -----
        This uses a greedy heuristic:
        1. Start with core nodes
        2. Add structures in order, prioritizing those with most core nodes
        3. Stop when budget would be exceeded
        """
        core_set = set(core_nodes)
        current_nodes = core_set.copy()
        selected_structures = []

        # Sort structures by number of core nodes (descending)
        def core_node_count(struct):
            """Count core nodes in structure for prioritization.

            Parameters
            ----------
            struct : tuple
                Structure tuple of (struct_id, nodes).

            Returns
            -------
            int
                Number of core nodes in the structure.
            """
            _, nodes = struct
            return sum(1 for n in nodes if n in core_set)

        sorted_structures = sorted(
            structures, key=core_node_count, reverse=True
        )

        # Greedily add structures
        for struct_id, nodes in sorted_structures:
            new_nodes = set(nodes) - current_nodes
            if len(current_nodes) + len(new_nodes) <= max_nodes:
                current_nodes.update(new_nodes)
                selected_structures.append((struct_id, nodes))

        return selected_structures

    def _extract_subgraph(self, node_ids: list[int]) -> Data:
        """Extract subgraph for given nodes.

        Parameters
        ----------
        node_ids : list of int
            Node IDs to include (sorted).

        Returns
        -------
        Data
            Subgraph Data object.
        """
        # Create node mask
        node_mask = torch.zeros(self.graph_data.num_nodes, dtype=torch.bool)
        node_mask[node_ids] = True

        # Create mapping
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Extract edges
        edge_index = self.graph_data.edge_index
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        batch_edge_index = edge_index[:, edge_mask]

        # Remap edge indices
        batch_edge_index = torch.tensor(
            [
                [node_id_to_idx[n.item()] for n in batch_edge_index[0]],
                [node_id_to_idx[n.item()] for n in batch_edge_index[1]],
            ],
            dtype=torch.long,
        )

        # Create batch Data
        batch_data = Data()
        batch_data.edge_index = batch_edge_index
        batch_data.num_nodes = len(node_ids)

        # Copy node features
        if hasattr(self.graph_data, "x") and self.graph_data.x is not None:
            batch_data.x = self.graph_data.x[node_ids]

        # Copy labels
        if hasattr(self.graph_data, "y") and self.graph_data.y is not None:
            batch_data.y = self.graph_data.y[node_ids]

        # Copy edge attributes
        if (
            hasattr(self.graph_data, "edge_attr")
            and self.graph_data.edge_attr is not None
        ):
            batch_data.edge_attr = self.graph_data.edge_attr[edge_mask]

        return batch_data

    def _create_empty_batch(self) -> Data:
        """Create empty batch for edge cases.

        Returns
        -------
        Data
            Empty Data object.
        """
        batch_data = Data()
        batch_data.num_nodes = 0
        batch_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch_data.core_mask = torch.tensor([], dtype=torch.bool)
        batch_data.expansion_ratio = 1.0
        batch_data.precomputed_structures = []
        batch_data.num_structures = 0
        batch_data.batch_node_ids = torch.tensor([], dtype=torch.long)
        batch_data.core_node_ids = torch.tensor([], dtype=torch.long)
        return batch_data

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExtendedContextCollate("
            f"max_expansion_ratio={self.max_expansion_ratio}, "
            f"transform={self.transform is not None})"
        )


def create_extended_context_dataloader(
    preprocessor: OnDiskTransductivePreprocessor,
    node_sampler: Any,
    max_expansion_ratio: float = 1.5,
    transform: Callable | None = None,
    filter_on_expansion: bool = True,
    **kwargs: Any,
):
    """Create a dataloader with extended context expansion.

    This convenience function combines a node sampler with extended context
    collate, creating a ready-to-use dataloader for training.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built structure index.
    node_sampler : Any
        Node sampler (e.g., ClusterAwareNodeSampler, or custom sampler).
        Must be iterable yielding node tensors.
    max_expansion_ratio : float, optional
        Maximum expansion ratio (default: 1.5).
    transform : Callable, optional
        Transform to apply (default: None).
    filter_on_expansion : bool, optional
        Filter structures if expansion exceeds ratio (default: True).
    **kwargs : dict
        Additional arguments for ExtendedContextCollate.

    Returns
    -------
    ExtendedContextLoader
        Configured loader combining sampler and extended context.
    """
    # Create collate function
    collate_fn = ExtendedContextCollate(
        preprocessor,
        max_expansion_ratio=max_expansion_ratio,
        transform=transform,
        filter_on_expansion=filter_on_expansion,
        **kwargs,
    )

    # Wrapper class combining sampler and collate
    class ExtendedContextLoader:
        """Loader combining node sampler with extended context.

        Parameters
        ----------
        sampler : Any
            Node sampler that yields batches of node IDs.
        collate_fn : ExtendedContextCollate
            Collate function for context expansion.
        """

        def __init__(self, sampler, collate_fn):
            self.sampler = sampler
            self.collate_fn = collate_fn

        def __iter__(self):
            """Iterate over batches with context expansion.

            Yields
            ------
            Data
                PyG Data object with extended context.
            """
            for core_nodes in self.sampler:
                yield self.collate_fn([core_nodes])

        def __len__(self):
            """Return number of batches.

            Returns
            -------
            int
                Number of batches in the sampler.
            """
            return len(self.sampler)

    return ExtendedContextLoader(node_sampler, collate_fn)
