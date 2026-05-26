"""Loaders for BA2Motif dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import BA2MotifDataset

from topobench.data.loaders.base import AbstractLoader


class BA2MotifDatasetLoader(AbstractLoader):
    """Load BA2Motif dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load BA2Motif dataset.

        Returns
        -------
        Dataset
            The loaded BA2Motif dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = BA2MotifDataset(
            root=str(self.root_data_dir),
        )
        return dataset
