from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.loaders.base import AbstractLoader
from topobench.data.loaders.graph.graphland.dataset import GraphlandDataset


class GraphlandDatasetLoader(AbstractLoader):
    """Load a Graphland dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "cocitation")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Graphland dataset.

        Returns
        -------
        Dataset
            The loaded Graphland dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = GraphlandDataset(
            root = str(self.root_data_dir),
            name = self.parameters.data_name,
            drop_missing_y = self.parameters.get("drop_missing_y", True),
            impute_missing_x = self.parameters.get("impute_missing_x", None),
        )
        return dataset
