"""Loader for ATLAS Top Tagging Dataset."""

from pathlib import Path

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets import ATLASTopTaggingDataset
from topobench.data.loaders.base import AbstractLoader


class ATLASTopTaggingDatasetLoader(AbstractLoader):
    """Load ATLAS Top Tagging dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - split: Which split to use ('train' or 'test')
            - subset: Fraction of dataset to load (0.0 to 1.0)
            - max_constituents: Maximum number of constituents per jet
            - use_high_level: Whether to include high-level features
            - verbose: Verbosity level
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load the ATLAS Top Tagging dataset.

        This is the main method called by TopoBench.
        It initializes the dataset and returns it.

        Returns
        -------
        Dataset
            The loaded ATLAS Top Tagging dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> ATLASTopTaggingDataset:
        """Helper method to instantiate the dataset class.

        Returns
        -------
        ATLASTopTaggingDataset
            The instantiated dataset.
        """
        # Extract parameters with defaults
        split = self.parameters.get("split", "train")
        subset = self.parameters.get("subset", 0.01)
        max_constituents = self.parameters.get("max_constituents", 80)
        use_high_level = self.parameters.get("use_high_level", True)
        verbose = self.parameters.get("verbose", False)

        return ATLASTopTaggingDataset(
            root=str(self.root_data_dir),
            split=split,
            subset=subset,
            max_constituents=max_constituents,
            use_high_level=use_high_level,
            verbose=verbose,
        )

    def _redefine_data_dir(self, dataset: ATLASTopTaggingDataset) -> Path:
        """Helper method to get the final processed data path.

        Parameters
        ----------
        dataset : ATLASTopTaggingDataset
            The dataset instance.

        Returns
        -------
        Path
            Path to the processed data directory.
        """
        return Path(dataset.processed_dir)
