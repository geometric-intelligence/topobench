"""Loader for Wiki CS dataset."""


from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets import WikiCSDataset
from topobench.data.loaders.base import AbstractLoader


class WikiCsDatasetLoader(AbstractLoader):
    """Load Wiki CS dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> WikiCSDataset:
        """Load the Wiki CS dataset.

        Returns
        -------
        WikiCSDataset
            The loaded Wiki CS dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> WikiCSDataset:
        """Initialize the Wiki CS dataset.

        Returns
        -------
        WikiCSDataset
            The initialized dataset instance.
        """
        return WikiCSDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: WikiCSDataset) -> Path:
        """Redefine the data directory based on the chosen (year, task_variable) pair.

        Parameters
        ----------
        dataset : WikiCSDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
