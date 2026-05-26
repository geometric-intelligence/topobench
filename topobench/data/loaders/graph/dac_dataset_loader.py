"""Loaders for US County Demos dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets import DACDataset
from topobench.data.loaders.base import AbstractLoader


class DACDatasetLoader(AbstractLoader):
    """Load DAC dataset with configurable year and task variable.

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

    def load_dataset(self) -> DACDataset:
        """Load the DAC dataset.

        Returns
        -------
        DACDataset
            The loaded DAC dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> DACDataset:
        """Initialize the DAC dataset.

        Returns
        -------
        DADataset
            The initialized dataset instance.
        """
        return DACDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: DACDataset) -> Path:
        """Redefine the data directory based on the chosen (year, task_variable) pair.

        Parameters
        ----------
        dataset : DACDataset
            The dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return dataset.processed_root
