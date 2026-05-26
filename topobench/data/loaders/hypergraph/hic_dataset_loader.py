"""Loaders for HIC hypergraph/graph classification datasets."""

from omegaconf import DictConfig

from topobench.data.datasets import HICDataset
from topobench.data.loaders.base import AbstractLoader


class HICDatasetLoader(AbstractLoader):
    """Loader for HIC datasets with configurable options.

    Parameters
    ----------
    parameters : DictConfig
        Configuration with at least ``data_dir`` and ``data_name`` fields and
        optional flags such as ``use_degree_as_tag``.
    """

    def __init__(self, parameters: DictConfig) -> None:
        """Initialize the HIC dataset loader."""
        super().__init__(parameters)
        self.data_dir = None

    def load_dataset(self) -> HICDataset:
        """Load the HIC dataset based on the configuration.

        Returns
        -------
        HICDataset
            Processed HIC dataset instance.
        """
        self.data_dir = self.get_data_dir()
        dataset = self._initialize_dataset()
        return dataset

    def _initialize_dataset(self) -> HICDataset:
        """Instantiate the underlying HICDataset.

        Returns
        -------
        HICDataset
            Initialized HICDataset object.
        """
        use_degree_as_tag = getattr(self.parameters, "use_degree_as_tag", False)

        return HICDataset(
            root=str(self.data_dir),
            name=self.parameters.data_name,
            use_degree_as_tag=use_degree_as_tag,
        )
