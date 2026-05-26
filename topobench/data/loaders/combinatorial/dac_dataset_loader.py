"""Loaders for Mantra dataset as simplicial."""

from omegaconf import DictConfig

from topobench.data.datasets import DACDataset
from topobench.data.loaders.base import AbstractLoader


class DACCombinatorialDatasetLoader(AbstractLoader):
    """Load Mantra dataset with configurable parameters.

     Note: for the simplicial datasets it is necessary to include DatasetLoader into the name of the class!

     Parameters
     ----------
     parameters : DictConfig
         Configuration parameters containing:
             - data_dir: Root directory for data
             - data_name: Name of the dataset
             - other relevant parameters

    **kwargs : dict
         Additional keyword arguments.
    """

    def __init__(self, parameters: DictConfig, **kwargs) -> None:
        super().__init__(parameters, **kwargs)

    def load_dataset(self, **kwargs) -> DACDataset:
        """Load the DAC Combinatorial dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        DACCombinatorialDataset
            The loaded DAC Combinatorial dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = self._initialize_dataset(**kwargs)
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self, **kwargs) -> DACDataset:
        """Initialize the Citation Hypergraph dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        CitationHypergraphDataset
            The initialized dataset instance.
        """
        return DACDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
            **kwargs,
        )
