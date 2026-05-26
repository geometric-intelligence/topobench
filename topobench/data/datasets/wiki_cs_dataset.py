"""Dataset class for Wiki CS dataset."""

import itertools
import json
import os
from typing import ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import (
    download_file_from_link,
)


class WikiCSDataset(InMemoryDataset):
    r"""Dataset class for Wiki CS dataset.

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
        "Wiki-CS": "https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset/data.json",
    }

    FILE_FORMAT: ClassVar = {
        "Wiki-CS": "json",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
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
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters})"

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
            dataset_name=self.name,
            file_format=self.file_format,
        )

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the Wiki CS data, applies any pre-processing
        transformation if specified, and saves the processed data to
        the appropriate location.
        Based on: https://github.com/pmernyei/wiki-cs-dataset/
        """
        # Step 1: read the data
        with open(
            os.path.join(self.raw_dir, self.name + "." + self.file_format)
        ) as f:
            raw_data = json.load(f)
        features = torch.FloatTensor(np.array(raw_data["features"]))
        labels = torch.LongTensor(np.array(raw_data["labels"]))

        edges = [
            [(i, j) for j in js] for i, js in enumerate(raw_data["links"])
        ]
        edges = list(itertools.chain(*edges))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(x=features, edge_index=edge_index, y=labels)

        data_list = [data]

        # Step 2: collate the graphs
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.

        # Step 3: save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

    @property
    def raw_file_names(self):
        r"""
        Return files that must be present in raw_dir for "download() not needed".

        Returns
        -------
        raw_file_names
            List of filenames.
        """
        return ["data.json"]

    @property
    def processed_paths(self):
        r"""
        The processed path to avoid processing.

        Returns
        -------
        processed_paths
            List of paths.
        """

        return [
            os.path.join(self.root, "processed", self.processed_file_names)
        ]

    @property
    def processed_root(self):
        r"""
        The processed root.

        Returns
        -------
        processed_root
            Path of processed root.
        """
        return self.root

    @property
    def processed_file_names(self):
        r"""
        The processed file produced by `process()`.

        Returns
        -------
        processed_file_names
            List of processed file names.
        """
        return "data.json"
