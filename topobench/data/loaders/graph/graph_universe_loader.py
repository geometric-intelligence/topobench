"""Loaders for GraphUniverse [1] datasets.

[1] "GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization" by Louis Van Langendonck and Guillermo Bernardez and Nina Miolane and Pere Barlet-Ros
Accepted at The Fourteenth International Conference on Learning Representations, 2026},
https://openreview.net/forum?id=jRWxvQnqUt
"""

from graph_universe import GraphUniverseDataset
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset

from topobench.data.loaders.base import AbstractLoader


class GraphUniverseDatasetLoader(AbstractLoader):
    """Load Graph Universe datasets.

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

    def load_dataset(self) -> Dataset:
        """Load Graph Universe dataset.

        Returns
        -------
        Dataset
            The loaded Graph Universe dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = GraphUniverseDataset(
            root=str(self.root_data_dir),
            parameters=self.parameters["generation_parameters"],
        )

        return dataset

    def load(self, **kwargs) -> tuple[Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        dataset = self.load_dataset(**kwargs)
        data_dir = dataset.raw_dir

        return dataset, data_dir
