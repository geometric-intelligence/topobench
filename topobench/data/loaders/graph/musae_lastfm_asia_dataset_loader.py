"""Loader for LastFM Asia Graph dataset."""


from omegaconf import DictConfig

from topobench.data.datasets.musae_lastfm_asia_dataset import (
    LastFmAsiaDataset,
)
from topobench.data.loaders.base import AbstractLoader


class LastFmAsiaDatasetLoader(AbstractLoader):
    """Load LastFM Asia Graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> LastFmAsiaDataset:
        """Load LastFM Asia Graph dataset.

        Returns
        -------
        Dataset
            The loaded LastFM Asia Graph dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = LastFmAsiaDataset(
            root=str(self.root_data_dir),
                name=self.parameters.data_name,
                parameters=self.parameters,
        )
        return dataset
