"""Loader for HAL Arlequin dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import HALArlequinDataset
from topobench.data.loaders.base import AbstractLoader


class HALArlequinDatasetLoader(AbstractLoader):
    """Load HAL Arlequin dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - other relevant parameters
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> HALArlequinDataset:
        """Load the HAL Arlequin dataset.

        Returns
        -------
        HALArlequinDataset
            The loaded HAL Arlequin dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> HALArlequinDataset:
        """Initialize the HAL Arlequin dataset.

        Returns
        -------
        HALArlequinDataset
            The initialized dataset instance.
        """
        return HALArlequinDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
