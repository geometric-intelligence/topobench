"""CityNetwork dataset loader implementation."""

from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets.citynetwork_dataset import CityNetworkDataset
from topobench.data.loaders.base import AbstractLoader


class CityNetworkDatasetLoader(AbstractLoader):
    """
    Loader for CityNetwork datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration object containing dataset parameters.
    """

    def __init__(self, parameters: DictConfig):
        super().__init__(parameters)

    def load_dataset(self) -> CityNetworkDataset:
        """
        Load the CityNetwork dataset.

        Returns
        -------
        CityNetworkDataset
            Loaded dataset instance.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir()
        return dataset

    def _initialize_dataset(self) -> CityNetworkDataset:
        """
        Initialize the CityNetwork dataset.

        Returns
        -------
        CityNetworkDataset
            Initialized CityNetwork dataset instance.
        """
        return CityNetworkDataset(
            root=str(self.parameters.data_dir),
            name=self.parameters.get("name", "paris"),
            augmented=self.parameters.get("augmented", True),
        )

    def _redefine_data_dir(self) -> Path:
        r"""Redefine the data directory path.

        Returns
        -------
        Path
            Path to the redefined data directory.
        """
        self.parameters.data_dir = (
            Path(self.parameters.data_dir)
            / "CityNetwork"
            / self.parameters.get("name", "paris")
        )
        return self.parameters.data_dir
