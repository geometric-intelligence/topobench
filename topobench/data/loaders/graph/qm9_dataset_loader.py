
"""Loader for QM9 molecular dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.qm9_dataset import QM9Dataset
from topobench.data.loaders.base import AbstractLoader


class QM9DatasetLoader(AbstractLoader):
    """Load QM9 molecular dataset for molecular property prediction.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset ('QM9')
            - target_property: Which molecular property to predict (default: 'gap')
            - subset_size: Size of subset for testing (None for full dataset)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> QM9Dataset:
        """Load the QM9 molecular dataset.

        Returns
        -------
        QM9Dataset
            The loaded QM9 dataset with the appropriate data directory.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> QM9Dataset:
        """Initialize the QM9 molecular dataset.

        Returns
        -------
        QM9Dataset
            The initialized QM9 dataset.

        Raises
        ------
        RuntimeError
            If dataset initialization fails.
        """
        try:
            dataset = QM9Dataset(
                root=str(self.get_data_dir()),
                name=self.parameters.data_name,
                parameters=self.parameters,
            )
            return dataset
        except Exception as e:
            msg = f"Error initializing QM9 dataset: {e}"
            raise RuntimeError(msg) from e

    def _redefine_data_dir(self, dataset: QM9Dataset) -> Path:
        """Redefine the data directory based on dataset configuration.

        Parameters
        ----------
        dataset : QM9Dataset
            The QM9 dataset instance.

        Returns
        -------
        Path
            The redefined data directory path.
        """
        return self.get_data_dir()
