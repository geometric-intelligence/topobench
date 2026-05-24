"""Loader for WikiCS dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import WikiCS

from topobench.data.loaders.base import AbstractLoader


class WikiCSDatasetLoader(AbstractLoader):
    """Load WikiCS dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load WikiCS dataset.

        Returns
        -------
        Dataset
            The loaded WikiCS dataset.
        """
        dataset = WikiCS(
            root=str(self.root_data_dir),
        )
        return dataset
