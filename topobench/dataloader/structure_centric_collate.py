"""Structure-centric collate function for transductive learning.

This module implements collate functions that build PyG Data batches from
sampled structure IDs, ensuring 100% "structure completeness" by construction.
"""

from collections.abc import Callable

import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import Data

from topobench.data.preprocessor.ondisk_transductive import (
    OnDiskTransductivePreprocessor,
)
from topobench.transforms.data_transform import DataTransform


class StructureCentricCollate:
    """Collate function for structure-centric batching.

    This collate function takes structure IDs from StructureCentricSampler
    and builds PyG Data batches with complete topological structures.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built structure index.
    transform : Callable, optional
        Topological transform to apply to batches (e.g., SimplicialCliqueLifting).
        If None, structures are passed as metadata only (default: None).
    include_structure_metadata : bool, optional
        Include structure metadata in batch (default: True).

    Attributes
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        The preprocessor instance.
    transform : Callable or None
        Transform to apply.
    """

    def __init__(
        self,
        preprocessor: OnDiskTransductivePreprocessor,
        transform: Callable | None = None,
        include_structure_metadata: bool = True,
    ) -> None:
        """Initialize structure-centric collate function.

        Parameters
        ----------
        preprocessor : OnDiskTransductivePreprocessor
            Preprocessor with structure index.
        transform : Callable, optional
            Transform to apply (default: None).
        include_structure_metadata : bool, optional
            Include structure metadata (default: True).
        """
        self.preprocessor = preprocessor
        self.include_structure_metadata = include_structure_metadata

        # Handle DictConfig transforms by converting to callable
        if isinstance(transform, DictConfig):
            self.transform = self._instantiate_transform(transform)
        else:
            self.transform = transform

        if not hasattr(preprocessor, "query_engine"):
            raise ValueError(
                "Preprocessor must have query_engine. "
                "Call preprocessor.build_index() first."
            )

        self.query_engine = preprocessor.query_engine
        self.graph_data = preprocessor.graph_data

    def __call__(self, structure_ids: list[int]) -> Data:
        """Build batch from structure IDs.

        Parameters
        ----------
        structure_ids : list of int
            Structure IDs to include in batch (from sampler).

        Returns
        -------
        Data
            PyG Data object with complete structures.

        Notes
        -----
        The batch includes:
        - x: Node features for batch nodes
        - edge_index: Edges between batch nodes
        - y: Labels (if present)
        - precomputed_structures: List of (structure_id, nodes) if metadata enabled
        - num_nodes: Node count
        - batch_node_ids: Original node IDs (for transductive mask)
        """
        # Step 1: Query structures by ID
        structures = self.query_engine.query_cliques_by_id(structure_ids)

        if not structures:
            # Empty batch - shouldn't happen but be defensive
            return self._create_empty_batch()

        # Step 2: Gather all unique nodes
        node_set = set()
        for _, nodes in structures:
            node_set.update(nodes)
        batch_node_ids = sorted(list(node_set))

        # Step 3: Build subgraph Data object
        batch_data = self._extract_subgraph(batch_node_ids)

        # Step 4: Add structure metadata
        if self.include_structure_metadata:
            # Remap structure nodes to batch-local indices
            node_id_to_idx = {
                nid: idx for idx, nid in enumerate(batch_node_ids)
            }
            remapped_structures = []
            for struct_id, nodes in structures:
                remapped_nodes = [node_id_to_idx[n] for n in nodes]
                remapped_structures.append((struct_id, remapped_nodes))

            batch_data.precomputed_structures = remapped_structures
            batch_data.num_structures = len(structures)

        # Store original node IDs for transductive learning
        batch_data.batch_node_ids = torch.tensor(
            batch_node_ids, dtype=torch.long
        )

        # Step 5: Apply transform if provided
        if self.transform is not None:
            # Safety check: ensure transform is callable (handle DictConfig edge case)
            if isinstance(self.transform, DictConfig):
                self.transform = self._instantiate_transform(self.transform)
            batch_data = self.transform(batch_data)

        return batch_data

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

        # Create mapping from old node IDs to new indices
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Extract edges
        edge_index = self.graph_data.edge_index
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        batch_edge_index = edge_index[:, edge_mask]

        # Remap edge indices to local batch indices
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

        # Copy node features if present
        if hasattr(self.graph_data, "x") and self.graph_data.x is not None:
            batch_data.x = self.graph_data.x[node_ids]

        # Copy labels if present
        if hasattr(self.graph_data, "y") and self.graph_data.y is not None:
            batch_data.y = self.graph_data.y[node_ids]

        # Copy edge attributes if present
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
        batch_data.precomputed_structures = []
        batch_data.num_structures = 0
        batch_data.batch_node_ids = torch.tensor([], dtype=torch.long)
        return batch_data

    def _instantiate_transform(
        self, transforms_config: DictConfig
    ) -> torch_geometric.transforms.Compose:
        """Instantiate transform from configuration.

        Parameters
        ----------
        transforms_config : DictConfig
            Transform configuration parameters.

        Returns
        -------
        torch_geometric.transforms.Compose
            Composed transform object.
        """
        # Handle nested liftings config (for compatibility)
        if "liftings" in transforms_config:
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

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StructureCentricCollate("
            f"num_cliques={self.query_engine.num_cliques}, "
            f"transform={self.transform is not None})"
        )


def create_structure_centric_dataloader(
    preprocessor: OnDiskTransductivePreprocessor,
    cliques_per_batch: int = 500,
    node_budget: int = 2000,
    transform: Callable | None = None,
    shuffle: bool = True,
):
    """Create an iterable with structure-centric sampling.

    This is a convenience function that sets up the sampler, collate function,
    and creates an iterable for structure-centric batching.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built index.
    cliques_per_batch : int, optional
        Target structures per batch (default: 500).
    node_budget : int, optional
        Max nodes per batch (default: 2000).
    transform : Callable, optional
        Transform to apply (default: None).
    shuffle : bool, optional
        Shuffle structures (default: True).

    Returns
    -------
    StructureCentricLoader
        Configured loader for structure-centric batching.
    """
    from topobench.dataloader.structure_centric_sampler import (
        StructureCentricSampler,
    )

    # Create sampler
    sampler = StructureCentricSampler(
        preprocessor,
        cliques_per_batch=cliques_per_batch,
        node_budget=node_budget,
        shuffle=shuffle,
    )

    # Create collate function
    collate_fn = StructureCentricCollate(preprocessor, transform=transform)

    # Return a simple loader wrapper
    class StructureCentricLoader:
        """Simple wrapper that combines sampler and collate.

        Parameters
        ----------
        sampler : StructureCentricSampler
            The sampler to use.
        collate_fn : callable
            The collate function to use.
        """

        def __init__(self, sampler, collate_fn):
            self.sampler = sampler
            self.collate_fn = collate_fn

        def __iter__(self):
            """Iterate over batches of structures.

            Yields
            ------
            torch_geometric.data.Batch
                Batch of structure data.
            """
            for structure_ids in self.sampler:
                yield self.collate_fn(structure_ids)

        def __len__(self):
            """Return the number of batches in the sampler.

            Returns
            -------
            int
                Number of batches.
            """
            return len(self.sampler)

    return StructureCentricLoader(sampler, collate_fn)
