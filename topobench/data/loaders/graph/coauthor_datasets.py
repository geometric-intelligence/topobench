"""Loaders for Coauthor Datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import Coauthor

from topobench.data.loaders.base import AbstractLoader


class CoauthorDatasetLoader(AbstractLoader):
    """Load Coauthor datasets.

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
        """Load Coauthor dataset.

        Returns
        -------
        Dataset
            The loaded Coauthor dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = Coauthor(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )
        return dataset
