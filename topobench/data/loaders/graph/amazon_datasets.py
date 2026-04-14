"""Loaders for Amazon datasets (Computers, Photo)."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import Amazon

from topobench.data.loaders.base import AbstractLoader


class AmazonDatasetLoader(AbstractLoader):
    """Load Amazon datasets (Computers, Photo).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset ('Computers' or 'Photo')
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Amazon dataset.

        Returns
        -------
        Dataset
            The loaded Amazon dataset.
        """
        dataset = Amazon(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )
        return dataset
