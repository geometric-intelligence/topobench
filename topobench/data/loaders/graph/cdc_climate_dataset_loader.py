"""Loaders for US County Demos dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.cdc_climate_dataset import CDCClimateDataset
from topobench.data.loaders.base import AbstractLoader


class CDCClimateDatasetLoader(AbstractLoader):
    """Load CDC Climate dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - year: Year of the dataset (if applicable)
            - task_variable: Task variable for the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> CDCClimateDataset:
        """Load the CDC Climate dataset.

        Returns
        -------
        CDCClimateDataset
            The loaded CDC Climate dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> CDCClimateDataset:
        """Initialize the CDC Climate dataset.

        Returns
        -------
        CDCClimateDataset
            The initialized dataset instance.
        """
        return CDCClimateDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: CDCClimateDataset) -> Path:
        """Redefine the data directory based on the chosen (year, task_variable) pair.

        Parameters
        ----------
        dataset : CDCClimateDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
