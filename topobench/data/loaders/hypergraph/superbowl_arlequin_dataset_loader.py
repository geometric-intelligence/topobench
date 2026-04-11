"""Loader for Superbowl Arlequin dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import SuperbowlArlequinDataset
from topobench.data.loaders.base import AbstractLoader


class SuperbowlArlequinDatasetLoader(AbstractLoader):
    """Load Superbowl Arlequin dataset with configurable parameters.

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

    def load_dataset(self) -> SuperbowlArlequinDataset:
        """Load the Superbowl Arlequin dataset.

        Returns
        -------
        SuperbowlArlequinDataset
            The loaded dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> SuperbowlArlequinDataset:
        """Initialize the Superbowl Arlequin dataset.

        Returns
        -------
        SuperbowlArlequinDataset
            The initialized dataset instance.
        """
        return SuperbowlArlequinDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
