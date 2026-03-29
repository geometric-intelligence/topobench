"""Structure-centric sampler for transductive learning.

This module provides samplers that implement structure-centric batching where
cliques are sampled first, then their constituent nodes are gathered. This
prevents structure fragmentation and ensures complete structures in each batch,
which is critical for topological deep learning.
"""

import random
from collections.abc import Iterator
from typing import Any

from topobench.dataloader.budget_utils import select_structures_within_budget


class StructureCentricSampler:
    """Sample cliques directly for batch construction.

    Implements structure-centric batching where cliques are sampled first,
    then nodes are gathered. This guarantees all nodes of each sampled clique
    are included together, preventing fragmentation.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built structure index.
    cliques_per_batch : int, optional
        Target number of cliques per batch (default: 500).
    node_budget : int, optional
        Maximum unique nodes per batch (default: 2000).
    shuffle : bool, optional
        Shuffle structure order each epoch (default: True).
    drop_last : bool, optional
        Drop last incomplete batch (default: False).

    Attributes
    ----------
    num_cliques : int
        Total number of cliques available.
    num_batches : int
        Expected number of batches per epoch (approximate).

    Examples
    --------
    >>> from topobench.data.preprocessor.ondisk_transductive import OnDiskTransductivePreprocessor
    >>> from topobench.dataloader.structure_centric_sampler import StructureCentricSampler
    >>> import torch_geometric as pyg
    >>>
    >>> # Load data and build index
    >>> data = pyg.datasets.Planetoid(root="/tmp", name="Cora")[0]
    >>> preprocessor = OnDiskTransductivePreprocessor(data, "./index")
    >>> preprocessor.build_index(max_clique_size=3)
    >>>
    >>> # Create sampler
    >>> sampler = StructureCentricSampler(
    ...     preprocessor, cliques_per_batch=500, node_budget=2000
    ... )
    >>>
    >>> # Iterate over clique ID batches
    >>> for clique_ids in sampler:
    ...     print(f"Batch: {len(clique_ids)} cliques")
    ...     # Use with StructureCentricCollate to build Data objects
    """

    def __init__(
        self,
        preprocessor: Any,
        cliques_per_batch: int = 500,
        node_budget: int = 2000,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """Initialize structure-centric sampler.

        Parameters
        ----------
        preprocessor : OnDiskTransductivePreprocessor
            Preprocessor with built structure index.
        cliques_per_batch : int, optional
            Target cliques per batch (default: 500).
        node_budget : int, optional
            Max unique nodes per batch (default: 2000).
        shuffle : bool, optional
            Shuffle cliques each epoch (default: True).
        drop_last : bool, optional
            Drop last incomplete batch (default: False).
        """
        self.preprocessor = preprocessor
        self.cliques_per_batch = cliques_per_batch
        self.node_budget = node_budget
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Get clique count from query engine
        if not hasattr(preprocessor, "query_engine"):
            raise ValueError(
                "Preprocessor must have a query_engine. "
                "Call preprocessor.build_index() first."
            )

        self.query_engine = preprocessor.query_engine
        self.num_cliques = self.query_engine.num_cliques

        if self.num_cliques == 0:
            raise ValueError(
                "No cliques indexed. "
                "Ensure build_index() was called and cliques were found."
            )

        # Pre-allocate clique ID list
        self.all_clique_ids = list(range(self.num_cliques))

        # Estimate number of batches (rough approximation)
        self.num_batches = max(1, self.num_cliques // cliques_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches of clique IDs.

        Yields
        ------
        list of int
            Batch of clique IDs that fit within budget constraints.
        """
        # Shuffle if requested
        clique_ids = self.all_clique_ids.copy()
        if self.shuffle:
            random.shuffle(clique_ids)

        # Iterate through cliques, forming batches
        i = 0
        while i < len(clique_ids):
            # Select cliques within budget
            remaining_ids = clique_ids[i:]
            batch_ids, _ = select_structures_within_budget(
                remaining_ids,
                self.query_engine,
                node_budget=self.node_budget,
                structures_per_batch=self.cliques_per_batch,
            )

            # Handle empty batch (shouldn't happen but be defensive)
            if not batch_ids:
                # Skip one clique and continue
                i += 1
                continue

            # Yield batch
            if not self.drop_last or len(batch_ids) == self.cliques_per_batch:
                yield batch_ids

            # Advance index
            i += len(batch_ids)

    def __len__(self) -> int:
        """Get approximate number of batches per epoch.

        Returns
        -------
        int
            Approximate batch count.

        Notes
        -----
        This is an estimate because budget constraints may vary batch sizes.
        """
        return self.num_batches

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StructureCentricSampler("
            f"num_cliques={self.num_cliques}, "
            f"cliques_per_batch={self.cliques_per_batch}, "
            f"node_budget={self.node_budget}, "
            f"shuffle={self.shuffle})"
        )


class StructureCentricBatchSampler:
    """Alternative sampler that returns exact batch indices.

    This sampler pre-computes batch boundaries for more predictable
    batch sizes, at the cost of less optimal budget packing.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built structure index.
    batch_size : int, optional
        Exact number of cliques per batch (default: 500).
    shuffle : bool, optional
        Shuffle clique order (default: True).
    drop_last : bool, optional
        Drop last incomplete batch (default: False).
    """

    def __init__(
        self,
        preprocessor: Any,
        batch_size: int = 500,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """Initialize batch sampler."""
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Get clique count
        if not hasattr(preprocessor, "query_engine"):
            raise ValueError(
                "Preprocessor must have a query_engine. "
                "Call preprocessor.build_index() first."
            )

        self.query_engine = preprocessor.query_engine
        self.num_cliques = self.query_engine.num_cliques

        if self.num_cliques == 0:
            raise ValueError("No cliques indexed.")

        self.all_clique_ids = list(range(self.num_cliques))

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches of clique IDs."""
        clique_ids = self.all_clique_ids.copy()
        if self.shuffle:
            random.shuffle(clique_ids)

        # Split into fixed-size batches
        for i in range(0, len(clique_ids), self.batch_size):
            batch = clique_ids[i : i + self.batch_size]

            # Skip last batch if drop_last and incomplete
            if self.drop_last and len(batch) < self.batch_size:
                continue

            yield batch

    def __len__(self) -> int:
        """Get number of batches per epoch.

        Returns
        -------
        int
            Number of batches in an epoch.
        """
        if self.drop_last:
            return self.num_cliques // self.batch_size
        return (self.num_cliques + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StructureCentricBatchSampler("
            f"num_cliques={self.num_cliques}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle})"
        )
