"""Loader for filtered Wikipedia Network datasets (Chameleon, Squirrel).

Uses the de-duplicated node versions from the heterophilous-graphs
repository (https://github.com/yandex-research/heterophilous-graphs),
which remove overlapping nodes present in the original geom-gcn data.
These are the versions used by TunedGNN (NeurIPS 2024) and other recent
heterophilic GNN benchmarks.
"""

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.loaders.base import AbstractLoader

# URLs for the filtered .npz files from the heterophilous-graphs repo.
_URLS = {
    "chameleon": (
        "https://github.com/yandex-research/heterophilous-graphs"
        "/raw/main/data/chameleon_filtered.npz"
    ),
    "squirrel": (
        "https://github.com/yandex-research/heterophilous-graphs"
        "/raw/main/data/squirrel_filtered.npz"
    ),
}


class _FilteredWikipediaDataset(InMemoryDataset):
    """InMemoryDataset wrapper for filtered Wikipedia Network .npz files.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    name : str
        Name of the dataset ('chameleon' or 'squirrel').
    """

    def __init__(self, root, name):
        self.dataset_name = name.lower()
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Return list of raw file names.

        Returns
        -------
        list[str]
            Raw file names.
        """
        return [f"{self.dataset_name}_filtered.npz"]

    @property
    def processed_file_names(self):
        """Return list of processed file names.

        Returns
        -------
        list[str]
            Processed file names.
        """
        return [f"{self.dataset_name}_filtered.pt"]

    def download(self):
        """Download the filtered .npz file."""
        from torch_geometric.data.download import download_url

        url = _URLS[self.dataset_name]
        download_url(url, self.raw_dir)

    def process(self):
        """Convert the raw .npz file into a PyG Data object."""
        raw_path = self.raw_paths[0]
        npz = np.load(raw_path)

        x = torch.from_numpy(npz["node_features"]).float()
        y = torch.from_numpy(npz["node_labels"]).long()
        edges = npz["edges"]  # (E, 2)
        edge_index = torch.from_numpy(edges.T).long()

        # Load built-in 10-fold splits if present.
        train_mask = val_mask = test_mask = None
        if "train_masks" in npz:
            # Shape: (10, N) boolean
            train_mask = torch.from_numpy(npz["train_masks"]).bool().T
            val_mask = torch.from_numpy(npz["val_masks"]).bool().T
            test_mask = torch.from_numpy(npz["test_masks"]).bool().T

        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        self.save([data], self.processed_paths[0])


class FilteredWikipediaDatasetLoader(AbstractLoader):
    """Load filtered Wikipedia Network datasets (Chameleon, Squirrel).

    These are the de-duplicated versions from the heterophilous-graphs
    repository, used by TunedGNN and other recent benchmarks.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset ('chameleon' or 'squirrel')
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self):
        """Load filtered Wikipedia Network dataset.

        Returns
        -------
        _FilteredWikipediaDataset
            The loaded dataset with de-duplicated nodes.
        """
        dataset = _FilteredWikipediaDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )
        return dataset
