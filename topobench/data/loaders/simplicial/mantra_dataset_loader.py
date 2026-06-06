"""Loaders for Mantra dataset as simplicial."""

from omegaconf import DictConfig

from topobench.data.datasets import MantraDataset
from topobench.data.loaders.base import AbstractLoader


class MantraSimplicialDatasetLoader(AbstractLoader):
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

    def load(self, **kwargs) -> tuple[MantraDataset, str]:
        """Load the Mantra dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        MantraDataset
            The loaded Mantra dataset with the appropriate `data_dir`.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = self.load_dataset(**kwargs)
        data_dir = dataset.processed_root
        return dataset, data_dir

    def load_dataset(self, **kwargs) -> MantraDataset:
        """Initialize the Mantra dataset.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for dataset initialization.

        Returns
        -------
        MantraDataset
            The initialized dataset instance.
        """
        slice_val = kwargs.pop("slice", None)
        if slice_val is None:
            slice_val = self.parameters.get("slice", False)
        return MantraDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
            slice=slice_val,
            **kwargs,
        )
