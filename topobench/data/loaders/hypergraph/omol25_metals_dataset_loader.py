"""Dataset loader for OMol25 metals in hypergraph domain."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader

from topobench.data.datasets.omol25_metals_dataset import OMol25MetalsDataset
from topobench.data.loaders.base import AbstractLoader


class OMol25MetalsDatasetLoader(AbstractLoader):
    """Loader for the OMol25 metal-complex subset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration dictionary containing:
            - data_domain: Domain string (e.g., "hypergraph")
            - data_type: Dataset family identifier (e.g., "omol25_metals")
            - data_name: Specific dataset name (e.g., "omol25_metals")
            - data_dir: Base directory containing the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        """Initialize the OMol25 metals dataset loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration parameters for the loader.
        """
        super().__init__(parameters)
        # Extract parameters for convenience
        self.data_domain = parameters.get("data_domain")
        self.data_type = parameters.get("data_type")
        self.data_name = parameters.get("data_name")
        self.data_dir = parameters.get("data_dir")

    def load_dataset(self) -> OMol25MetalsDataset:
        """Load the OMol25 metals dataset.

        Returns
        -------
        OMol25MetalsDataset
            The loaded dataset.
        """
        return self._load_full_dataset()

    def _get_dataset_root(self) -> Path:
        """Resolve the root path passed to :class:`OMol25MetalsDataset`.

        Returns
        -------
        pathlib.Path
            Root directory that contains the ``processed`` folder.
        """
        base = Path(self.data_dir).expanduser().resolve()
        return base / self.data_name

    def _load_full_dataset(self) -> OMol25MetalsDataset:
        """Instantiate the underlying :class:`OMol25MetalsDataset`.

        Returns
        -------
        OMol25MetalsDataset
            The loaded dataset instance.
        """
        root = self._get_dataset_root()
        return OMol25MetalsDataset(root=str(root))

    def get_splits(
        self,
        split_params: Mapping[str, Any],
    ) -> dict[str, torch.utils.data.Dataset]:
        """Create train, validation, and test splits.

        Parameters
        ----------
        split_params : Mapping[str, Any]
            Dictionary describing the split configuration. Expected keys
            include ``"learning_setting"``, ``"split_type"``, ``"data_seed"``,
            ``"train_prop"``, and optionally ``"val_prop"``.

        Returns
        -------
        dict of str to Dataset
            Mapping with keys ``"train"``, ``"val"``, and ``"test"``.
        """
        dataset = self._load_full_dataset()
        n_total = len(dataset)

        learning_setting = split_params.get("learning_setting", "inductive")
        split_type = split_params.get("split_type", "random_in_train")

        if learning_setting != "inductive":
            msg = (
                "OMol25MetalsDatasetLoader currently supports only "
                f'learning_setting="inductive", got "{learning_setting}".'
            )
            raise NotImplementedError(msg)

        if split_type not in {"random_in_train", "random"}:
            msg = (
                "OMol25MetalsDatasetLoader currently supports only random "
                f'splits, got split_type="{split_type}".'
            )
            raise NotImplementedError(msg)

        train_prop = float(split_params.get("train_prop", 0.8))
        val_prop = split_params.get("val_prop")
        val_prop = None if val_prop is None else float(val_prop)

        if not 0.0 < train_prop < 1.0:
            msg = f"train_prop must be in (0, 1), got {train_prop}."
            raise ValueError(msg)

        if val_prop is not None and not 0.0 <= val_prop < 1.0:
            msg = f"val_prop must be in [0, 1), got {val_prop}."
            raise ValueError(msg)

        if val_prop is None:
            val_prop = 0.1

        if train_prop + val_prop >= 1.0:
            msg = (
                "train_prop + val_prop must be < 1.0, "
                f"got {train_prop + val_prop}."
            )
            raise ValueError(msg)

        n_train = int(round(train_prop * n_total))
        n_val = int(round(val_prop * n_total))
        n_test = n_total - n_train - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            msg = (
                "Invalid split sizes with "
                f"n_total={n_total}, train={n_train}, "
                f"val={n_val}, test={n_test}."
            )
            raise ValueError(msg)

        generator = torch.Generator()
        generator.manual_seed(int(split_params.get("data_seed", 0)))

        ds_train, ds_val, ds_test = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

        return {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
        }

    def get_dataloaders(
        self,
        split_params: Mapping[str, Any],
        dataloader_params: Mapping[str, Any],
    ) -> dict[str, DataLoader]:
        """Wrap the splits into PyG :class:`DataLoader` objects.

        Parameters
        ----------
        split_params : Mapping[str, Any]
            Split configuration dictionary (same as for :meth:`get_splits`).
        dataloader_params : Mapping[str, Any]
            Dictionary with loader settings such as ``"batch_size"``,
            ``"num_workers"``, ``"pin_memory"``, and ``"persistent_workers"``.

        Returns
        -------
        dict of str to DataLoader
            Mapping with keys ``"train"``, ``"val"``, and ``"test"``.
        """
        splits = self.get_splits(split_params)

        batch_size = int(dataloader_params.get("batch_size", 64))
        num_workers = int(dataloader_params.get("num_workers", 0))
        pin_memory = bool(dataloader_params.get("pin_memory", False))
        persistent_workers = bool(
            dataloader_params.get("persistent_workers", False)
            and num_workers > 0
        )

        loaders: dict[str, DataLoader] = {}
        for split_name, split_dataset in splits.items():
            shuffle = split_name == "train"
            loaders[split_name] = DataLoader(
                split_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

        return loaders

    def __call__(
        self,
        split_params: Mapping[str, Any],
        dataloader_params: Mapping[str, Any],
    ) -> tuple[dict[str, torch.utils.data.Dataset], dict[str, DataLoader]]:
        """Return dataset splits and dataloaders.

        Parameters
        ----------
        split_params : Mapping[str, Any]
            Split configuration dictionary.
        dataloader_params : Mapping[str, Any]
            Loader configuration dictionary.

        Returns
        -------
        tuple
            Two-element tuple ``(splits, loaders)`` where both entries are
            dictionaries keyed by ``"train"``, ``"val"``, and ``"test"``.
        """
        splits = self.get_splits(split_params)
        loaders = self.get_dataloaders(split_params, dataloader_params)
        return splits, loaders

    def load(
        self,
        slice: int | None = None,
    ) -> tuple[OMol25MetalsDataset, str]:
        """Load the full dataset without splits.

        Parameters
        ----------
        slice : int, optional
            If provided, limit the dataset to the first ``slice`` samples
            for quick testing. If None, load the entire dataset.

        Returns
        -------
        tuple
            Two-element tuple ``(dataset, data_dir)`` containing the loaded
            dataset and the data directory path.
        """
        dataset = self._load_full_dataset()
        if slice is not None and slice > 0:
            dataset = Subset(dataset, range(min(slice, len(dataset))))
        return dataset, str(self._get_dataset_root())
