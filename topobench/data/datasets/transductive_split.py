"""Transductive split dataset wrapper for train/val/test splits."""

from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class TransductiveSplitDataset(Dataset):
    """Dataset wrapper for transductive splits that works with TBDataloader.

    This class wraps a transductive preprocessor and provides a dataset
    interface compatible with TBDataloader.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        The transductive preprocessor with built index.
    split_config : DictConfig
        Configuration specifying strategy and parameters.
    mask : torch.Tensor, optional
        Boolean mask for this split (default: None for all nodes).
    split_name : str, optional
        Name of this split ('train', 'val', or 'test').

    Examples
    --------
    >>> # Typically created via preprocessor.load_dataset_splits()
    >>> split_config = OmegaConf.create({
    ...     "strategy": "structure_centric",
    ...     "cliques_per_batch": 500,
    ...     "node_budget": 2000,
    ... })
    >>> train, val, test = preprocessor.load_dataset_splits(split_config)
    >>> type(train)
    <class 'topobench.data.datasets.transductive_split.TransductiveSplitDataset'>
    >>> # Each dataset item is a pre-batched subgraph, so use batch_size=1
    >>> datamodule = TBDataloader(train, val, test, batch_size=1)
    """

    def __init__(
        self,
        preprocessor: Any,
        split_config: DictConfig,
        mask: torch.Tensor | None = None,
        split_name: str = "train",
    ):
        """Initialize transductive split dataset."""
        self.preprocessor = preprocessor
        self.split_config = split_config
        self.mask = mask
        self.split_name = split_name

        # Get strategy from config
        self.strategy = split_config.get("strategy", "structure_centric")

        # Cache for batches (lazy materialization)
        self._batches = None
        self._loader = None

        # Create the underlying loader based on strategy
        self._create_loader()

    def _create_loader(self):
        """Create the appropriate loader based on strategy."""
        if self.strategy == "structure_centric":
            self._create_structure_centric_loader()
        elif self.strategy == "extended_context":
            self._create_extended_context_loader()
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Must be 'structure_centric' or 'extended_context'."
            )

    def _materialize_batches(self):
        """Materialize all batches from loader (lazy, only done once).

        Returns
        -------
        list
            List of pre-batched subgraph tensors.
        """
        if self._batches is None:
            self._batches = list(self._loader)
        return self._batches

    def _create_structure_centric_loader(self):
        """Create structure-centric loader."""
        from topobench.dataloader.structure_centric_collate import (
            create_structure_centric_dataloader,
        )

        # Get parameters from config
        cliques_per_batch = self.split_config.get("cliques_per_batch", 500)
        node_budget = self.split_config.get("node_budget", 2000)
        shuffle = self.split_config.get("shuffle", self.split_name == "train")

        # Create loader
        self._loader = create_structure_centric_dataloader(
            self.preprocessor,
            cliques_per_batch=cliques_per_batch,
            node_budget=node_budget,
            shuffle=shuffle,
            transform=self.preprocessor.transforms_config,
        )

    def _create_extended_context_loader(self):
        """Create extended context loader."""
        from topobench.dataloader.cluster_aware_sampler import (
            ClusterAwareNodeSampler,
        )
        from topobench.dataloader.extended_context_collate import (
            create_extended_context_dataloader,
        )

        # Get parameters from config
        nodes_per_batch = self.split_config.get("nodes_per_batch", 1000)
        max_expansion_ratio = self.split_config.get("max_expansion_ratio", 1.5)
        sampler_method = self.split_config.get("sampler_method", "louvain")
        shuffle = self.split_config.get("shuffle", self.split_name == "train")

        # Create node sampler
        node_sampler = ClusterAwareNodeSampler(
            graph_data=self.preprocessor.graph_data,
            batch_size=nodes_per_batch,
            clustering_method=sampler_method,
            shuffle=shuffle,
            mask=self.mask,
        )

        # Create loader with extended context
        self._loader = create_extended_context_dataloader(
            self.preprocessor,
            node_sampler=node_sampler,
            max_expansion_ratio=max_expansion_ratio,
            transform=self.preprocessor.transforms_config,
        )

    def __iter__(self):
        """Iterate over batches.

        For efficiency, if batches haven't been materialized yet and we're
        just iterating once, use the loader directly. Otherwise use cached batches.

        Returns
        -------
        iterator
            Iterator over pre-batched subgraph tensors.
        """
        if self._batches is not None:
            # Already materialized, iterate over cache
            return iter(self._batches)
        else:
            # First iteration, use loader directly for efficiency
            return iter(self._loader)

    def __len__(self):
        """Return number of batches.

        This requires materializing all batches if not already done,
        since the loader length may not be available upfront.

        Returns
        -------
        int
            Number of batches in the dataset.
        """
        batches = self._materialize_batches()
        return len(batches)

    def __getitem__(self, idx):
        """Get batch by index.

        This provides random access to batches, required for PyTorch DataLoader.
        Batches are materialized lazily on first access and cached.

        Parameters
        ----------
        idx : int
            Index of the batch to retrieve.

        Returns
        -------
        Data
            The batch at the specified index.
        """
        batches = self._materialize_batches()
        return batches[idx]

    def __repr__(self):
        """String representation."""
        return (
            f"TransductiveSplitDataset("
            f"split={self.split_name}, "
            f"strategy={self.strategy}, "
            f"batches={len(self)})"
        )
