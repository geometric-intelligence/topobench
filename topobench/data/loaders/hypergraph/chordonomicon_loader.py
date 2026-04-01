"""Loader for Chordonomicon dataset."""

from topobench.data.datasets import ChordonomiconDataset
from topobench.data.loaders.base import AbstractLoader


class ChordonomiconDatasetLoader(AbstractLoader):
    """Loader class for Chordonomicon dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir (str): Root directory where the dataset folder is stored.
            - data_name (str): Name of the dataset.
            - version (str): Version of the dataset, options are 'single_scale', 'all_scales'.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.version = parameters.version

    def load_dataset(self) -> ChordonomiconDataset:
        """Load the Chordonomicon dataset.

        Returns
        -------
        ChordonomiconDataset
            The loaded Chordonomicon dataset.
        """
        return ChordonomiconDataset(
            root=self.root_data_dir,
            name=self.parameters.data_name,
            version=self.version,
        )
