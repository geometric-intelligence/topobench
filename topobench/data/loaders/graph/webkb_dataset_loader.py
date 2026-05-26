"""Loaders for WebKB datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import WebKB

from topobench.data.loaders.base import AbstractLoader


class WebKBDatasetLoader(AbstractLoader):
    """Load WebKB datasets (Cornell, Texas, Wisconsin).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: One of {"Cornell", "Texas", "Wisconsin"}
            - data_type: Type of the dataset (e.g., "node_classification")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load WebKB dataset.

        Returns
        -------
        Dataset
            The loaded WebKB dataset (single-graph, node-classification).

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = WebKB(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )
        return dataset
