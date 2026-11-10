"""Loaders for TU datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset

from topobench.data.loaders.base import AbstractLoader


class TUDatasetLoader(AbstractLoader):
    """Load TU datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "graph_classification")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def get_data_dir(self) -> str:
        """Get the processed-data directory for the selected attributes.

        Optional continuous node/edge attributes change ``x`` and
        ``edge_attr``, so they get their own preprocessing cache. The default
        (no attributes) keeps the upstream path.

        Returns
        -------
        str
            The path to the dataset directory.
        """
        data_dir = super().get_data_dir()
        suffix = "".join(
            name
            for key, name in (
                ("use_node_attr", "_node_attr"),
                ("use_edge_attr", "_edge_attr"),
            )
            if self.parameters.get(key, False)
        )
        return data_dir + suffix

    def load_dataset(self) -> Dataset:
        """Load TU dataset.

        Returns
        -------
        Dataset
            The loaded TU dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = TUDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            use_node_attr=self.parameters.get("use_node_attr", False),
            use_edge_attr=self.parameters.get("use_edge_attr", False),
        )
        return dataset
