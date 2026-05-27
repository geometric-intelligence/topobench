"""Loader for Twitch dataset."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.twitch_dataset import TwitchDataset
from topobench.data.loaders.base import AbstractLoader


class TwitchDatasetLoader(AbstractLoader):
    """Load Twitch dataset.
    
    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset ('Twitch')
            - subset_size: Size of subset for testing (None for full dataset)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> TwitchDataset:
        """Load the Twitch dataset.
        Returns
        -------
        TwitchDataset
            The loaded Twitch dataset with the appropriate data directory.
        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)
        return dataset

    def _initialize_dataset(self) -> TwitchDataset:
        """Initialize the Twitch dataset.
        Returns
        -------
        TwitchDataset
            The initialized Twitch dataset.
        Raises
        ------
        RuntimeError
            If dataset initialization fails.
        """
        try:
            dataset = TwitchDataset(
                root=str(self.get_data_dir()),
                name=self.parameters.language,
                parameters=self.parameters,
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Error initializing Twitch dataset: {e}") from e

    def _redefine_data_dir(self, dataset: TwitchDataset) -> Path:
        """Redefine the data directory based on dataset configuration.
        Parameters
        ----------
        dataset : TwitchDataset
            The Twitch dataset instance.
        Returns
        -------
        Path
            The redefined data directory path.
        """
        return self.get_data_dir()
