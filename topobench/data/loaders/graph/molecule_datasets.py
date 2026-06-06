"""Loaders for Molecule datasets (ZINC, AQSOL, and QM9)."""

import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Dataset
from torch_geometric.datasets import AQSOL, QM9, ZINC

from topobench.data.loaders.base import AbstractLoader


class MoleculeDatasetLoader(AbstractLoader):
    """Load molecule datasets (ZINC and AQSOL) with predefined splits, or QM9.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "molecule")
            - qm9_target_index: (QM9 only) Which of the 19 regression targets to use (default 0).
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)
        self.datasets: list[Dataset] = []

    def load_dataset(self) -> Dataset:
        """Load the molecule dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        if self.parameters.data_name == "QM9":
            dataset = QM9(root=str(self.root_data_dir))
            self._collapse_qm9_targets(dataset)
            return dataset

        self._load_splits()
        split_idx = self._prepare_split_idx()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset

    def _collapse_qm9_targets(self, dataset: QM9) -> None:
        """Keep a single regression target; QM9 stores 19 columns in ``y``.

        Parameters
        ----------
        dataset : QM9
            The QM9 dataset whose batched targets should be collapsed in-place
            to a single regression target.
        """
        target_idx = int(
            OmegaConf.select(self.parameters, "qm9_target_index", default=0)
        )
        if target_idx < 0 or target_idx > 18:
            raise ValueError(
                f"qm9_target_index must be in [0, 18], got {target_idx}."
            )
        data = dataset._data
        if data.y is None:
            return
        if data.y.dim() != 2 or data.y.size(1) <= target_idx:
            raise ValueError(
                "QM9 expected batched y with shape [num_graphs, 19]."
            )
        # One scalar per graph ([num_graphs]) so batched labels are [B], not [B, 1, 1]
        data.y = data.y[:, target_idx].contiguous()

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified dataset."""
        for split in ["train", "val", "test"]:
            if self.parameters.data_name == "ZINC":
                self.datasets.append(
                    ZINC(
                        root=str(self.root_data_dir),
                        subset=True,
                        split=split,
                    )
                )
            elif self.parameters.data_name == "AQSOL":
                self.datasets.append(
                    AQSOL(
                        root=str(self.root_data_dir),
                        split=split,
                    )
                )

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        split_idx = {"train": np.arange(len(self.datasets[0]))}
        split_idx["valid"] = np.arange(
            len(self.datasets[0]),
            len(self.datasets[0]) + len(self.datasets[1]),
        )
        split_idx["test"] = np.arange(
            len(self.datasets[0]) + len(self.datasets[1]),
            len(self.datasets[0])
            + len(self.datasets[1])
            + len(self.datasets[2]),
        )
        return split_idx

    def _combine_splits(self) -> Dataset:
        """Combine the dataset splits into a single dataset.

        Returns
        -------
        Dataset
            The combined dataset containing all splits.
        """
        return self.datasets[0] + self.datasets[1] + self.datasets[2]

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
