"""Loaders for Wikipedia Network datasets (Squirrel, Chameleon)."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import WikipediaNetwork

from topobench.data.loaders.base import AbstractLoader


class WikipediaNetworkDatasetLoader(AbstractLoader):
    """Load Wikipedia Network datasets (Squirrel, Chameleon).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset ('Squirrel' or 'Chameleon')
            - geom_gcn_preprocess: Whether to use geom-gcn preprocessed
              features (default: True). The preprocessed version filters
              node features and includes fixed 10-fold splits.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Wikipedia Network dataset.

        Returns
        -------
        Dataset
            The loaded Wikipedia Network dataset.
        """
        geom_gcn_preprocess = self.parameters.get("geom_gcn_preprocess", True)
        dataset = WikipediaNetwork(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            geom_gcn_preprocess=geom_gcn_preprocess,
        )
        return dataset
