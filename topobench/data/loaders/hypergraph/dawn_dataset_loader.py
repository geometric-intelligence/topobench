"""Loaders for Dawn Temporal Hypergraph dataset."""

from omegaconf import DictConfig

from topobench.data.datasets.dawn_hypergraph_dataset import (
    DawnDataset as HypergraphDataset,
)
from topobench.data.loaders.base import AbstractLoader


class DawnDatasetLoader(AbstractLoader):
    """Load Citation Hypergraph dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - google_drive_url: URL for downloading the dataset
            - other relevant parameters
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> HypergraphDataset:
        """Load the Citation Hypergraph dataset.

        Returns
        -------
        HypergraphDataset
            The loaded Citation Hypergraph dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = self._initialize_dataset()
        # Update the loader's data_dir to match the dataset's root
        self.data_dir = dataset.root
        return dataset

    def _initialize_dataset(self) -> HypergraphDataset:
        """Initialize the Citation Hypergraph dataset.

        Returns
        -------
        HypergraphDataset
            The initialized dataset instance.
        """
        # Retrieve URL from parameters (defined in YAML)
        google_drive_url = self.parameters.get("google_drive_url", None)

        # Initialize the dataset with all required parameters
        # We explicitly convert data_dir to string to satisfy PyG's root expectation
        return HypergraphDataset(
            root=str(self.parameters.data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
            google_drive_url=google_drive_url,
        )
