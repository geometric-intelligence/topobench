"""
Data loader for the Bowen et al. mouse auditory cortex calcium imaging dataset.

This script downloads and processes the original dataset introduced in:

[Citation] Bowen et al. (2024), "Fractured columnar small-world functional network
organization in volumes of L2/3 of mouse auditory cortex," PNAS Nexus, 3(2): pgae074.
https://doi.org/10.1093/pnasnexus/pgae074

We apply the preprocessing and graph-construction steps defined in this module to obtain
a representation of neuronal activity suitable for our experiments.

Please cite the original paper when using this dataset or any derivatives.
"""

import torch
from omegaconf import DictConfig

from topobench.data.datasets.a123 import A123CortexMDataset
from topobench.data.loaders.base import AbstractLoader


class A123DatasetLoader(AbstractLoader):
    """Loader for A123 mouse auditory cortex dataset.

    Implements the AbstractLoader interface: accepts a DictConfig `parameters`
    and implements `load_dataset()` which returns a dataset object.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters for the dataset.
    **overrides
        Additional keyword arguments to override parameters.
    """

    def __init__(self, parameters: DictConfig, **overrides):
        """Initialize the A123 dataset loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration parameters for the dataset.
        **overrides
            Additional keyword arguments to override parameters.
        """
        # Initialize AbstractLoader (sets self.parameters and self.root_data_dir)
        super().__init__(parameters)

        # hyperparameters can come from the DictConfig or be passed as overrides
        params = parameters if parameters is not None else {}

        self.batch_size = int(params.get("batch_size", overrides.get("batch_size", 32)))
        # dataset will be created when load_dataset() is called
        self.dataset = None

    def load_dataset(
        self,
    ) -> torch.utils.data.Dataset:
        """Instantiate and return the underlying dataset.

        Returns a `A123CortexMDataset` instance constructed from the loader's
        parameters and root data directory.

        Returns
        -------
        torch.utils.data.Dataset
            A123CortexMDataset instance.
        """
        # determine dataset name from parameters, fallback to expected id
        name = self.parameters.data_name

        # root path for dataset: use the parent of root_data_dir since the dataset
        # constructs its own subdirectory based on name
        root = str(self.root_data_dir.parent)

        # Construct dataset; A123CortexMDataset expects (root, name, parameters)
        self.dataset = A123CortexMDataset(
            root=root, name=name, parameters=self.parameters
        )

        return self.dataset
