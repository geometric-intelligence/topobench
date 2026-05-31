# location: topobench/data/loaders/graph/ws1000_gamma_dataset_loader.py
from pathlib import Path

from omegaconf import DictConfig

from topobench.data.datasets import WS1000GammaDataset
from topobench.data.loaders.base import AbstractLoader


class WS1000GammaDatasetLoader(AbstractLoader):
    """
    Loader for the WS1000-Gamma synthetic dataset.

    Parameters
    ----------
    parameters : omegaconf.DictConfig
        The configuration block located at
        ``dataset.loader.parameters`` in the Hydra config. It must
        contain at least the following fields:

        - ``data_domain`` : str
        - ``data_type`` : str
        - ``data_name`` : str
        - ``data_dir`` : str
        - ``num_nodes`` : int
        - ``feature_dim`` : int
        - ``mean_degree`` : int
        - ``beta`` : float
        - ``gamma`` : float
        - ``noise_scale`` : float
        - ``seed`` : int
    """
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> WS1000GammaDataset:
        """
        Load the WS1000-Gamma dataset.
        Returns
        -------
        WS1000GammaDataset
            The instantiated dataset containing one synthetic graph with
            BFS-derived node features.
        """


        dataset = self._initialize_dataset()
        self.data_dir = self._redefine_data_dir(dataset)

        return dataset

    def _initialize_dataset(self) -> WS1000GammaDataset:
        """
        Instantiate the underlying :class:`WS1000GammaDataset`.
        Returns
        -------
        WS1000GammaDataset
            A dataset instance that will trigger processing if the
            processed data file is missing.
        """
        return WS1000GammaDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )

    def _redefine_data_dir(self, dataset: WS1000GammaDataset) -> Path:
        """
        Resolve the dataset directory to the processed root.

        TopoBench components expect ``loader.data_dir`` to point to the
        directory containing processed files. This method extracts the
        correct processed directory from the dataset object.

        Parameters
        ----------
        dataset : WS1000GammaDataset
            The dataset whose processed directory is being queried.

        Returns
        -------
        pathlib.Path
            Path to the processed dataset directory.
        """
        return Path(dataset.processed_dir)
