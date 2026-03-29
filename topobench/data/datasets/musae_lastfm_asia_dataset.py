"""Dataset class for LastFM Asia dataset."""

import json
import os
import os.path as osp
import shutil
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobench.data.utils import (
    download_file_from_link,
)


class LastFmAsiaDataset(InMemoryDataset):
    r"""Dataset class for LastFM Asia dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "musae_lastfm_asia": "https://snap.stanford.edu/data/lastfm_asia.zip",
    }

    FILE_FORMAT: ClassVar = {
        "musae_lastfm_asia": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.raw_name = "lasftm_asia"
        self.parameters = parameters
        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """

        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["lastfm_asia_edges.csv", "lastfm_asia_features.json", "lastfm_asia_target.csv"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Download data from the source
        self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]
        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.raw_name,
            file_format=self.file_format,
        )

        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.raw_name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)

        # Move files from osp.join(folder, name_download) to folder
        for file in os.listdir(osp.join(folder, self.raw_name)):
            shutil.move(osp.join(folder, self.raw_name, file), folder)
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.raw_name))

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the LastFM Asia data, applies any pre-
        processing transformations if specified, and saves the processed data
        to the appropriate location.
        """
        # Step 1: Load raw data files
        folder = self.raw_dir
        # Edges:
        tmp = pd.read_csv(osp.join(folder,"lastfm_asia_edges.csv"))[["node_1","node_2"]].to_numpy()
        edge_index = torch.tensor(tmp, dtype=torch.long).t().contiguous()
        # Targets:
        tmp = pd.read_csv(osp.join(folder,"lastfm_asia_target.csv")).sort_values("id")["target"].to_numpy()
        y = torch.tensor(tmp, dtype=torch.long)
        # Node features:
        with open(osp.join(folder,"lastfm_asia_features.json")) as infile:
            featdict = json.load(infile)
        row = []
        col = []
        values = []
        for node_id_str, feature_list in featdict.items():
            node_id = int(node_id_str)
            for feature_id in feature_list:
                row.append(node_id)
                col.append(int(feature_id))
                values.append(1)
        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        values = np.array(values, dtype=int)
        node_count = row.max() + 1
        feature_count = col.max() + 1
        shape = (node_count, feature_count)
        x = torch.sparse_coo_tensor(np.stack([row, col], axis=0), values, shape)
        data = Data(x=x, y=y, edge_index=edge_index)
        data_list = [data]

        # Step 2: collate the graphs
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.

        # Step 3: save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
