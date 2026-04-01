"""Dataset class for Chordonomicon dataset."""

import ast
import os
import os.path as osp

import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs


class ChordonomiconDataset(InMemoryDataset):
    """Dataset class for Chordonomicon dataset.

    Parameters
    ----------
    root : str
        Directory where the dataset will be stored, raw
        and processed will be subdirectories of it.
    name : str
        Name of the dataset (e.g., 'Chordonomicon').
    version : str
        Version of the dataset, options are 'single_scale' or 'all_scales'.
    """

    def __init__(self, root, name, version):
        self.name = name
        self.root = root
        self.version = version
        self.folder_chordonomicon = osp.join(self.root, self.name)
        if self.version == "single_scale":
            self.url = "https://huggingface.co/datasets/PierrickLeKing/topobench-music-synergy/resolve/main/dataframe_226.zip"  # pylint: disable=line-too-long
        elif self.version == "all_scales":
            self.url = "https://huggingface.co/datasets/PierrickLeKing/topobench-music-synergy/resolve/main/dataframe_4313.zip"  # pylint: disable=line-too-long
        super().__init__(
            root,
        )
        out = fs.torch_load(self.processed_paths[0])
        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)

    def download(self):
        """Download the Chordonomicon dataset.

        Raises:
            requests.exceptions.HTTPError: If the download fails.
        """
        r = requests.get(self.url, timeout=30)
        r.raise_for_status()
        with open(
            osp.join(self.folder_chordonomicon, "dataframe.zip"), "wb"
        ) as f:
            f.write(r.content)
        extract_zip(
            osp.join(self.folder_chordonomicon, "dataframe.zip"),
            osp.join(self.folder_chordonomicon, "raw"),
        )
        os.unlink(osp.join(self.folder_chordonomicon, "dataframe.zip"))

    def process(self):
        """Handle the Chordonomicon dataset.

        Convert the raw data into a PyTorch Geometric Data object and save it.
        """
        df = pd.read_csv(
            osp.join(self.folder_chordonomicon, "raw", self.raw_file_names[0])
        )
        df["chords"] = (
            df["chords"].apply(ast.literal_eval).apply(list).apply(np.array)
        )
        t1 = torch.from_numpy(np.concatenate(df["chords"].values))
        t2 = torch.tensor(df["chords"].apply(len).values)
        indices = torch.stack(
            (t1, torch.repeat_interleave(torch.arange(len(t2)), t2))
        )
        incidence_hyperedges = torch.sparse_coo_tensor(
            indices, torch.ones(indices.shape[1])
        ).coalesce()
        x_hyperedges = torch.tensor(
            df["frequency"].values, dtype=torch.float32
        ).unsqueeze(1)
        y_hyperedges = torch.tensor(
            df["local_o_info"].values, dtype=torch.float32
        )
        data = Data(
            incidence_hyperedges=incidence_hyperedges,
            num_hyperedges=incidence_hyperedges.size(1),
            x_hyperedges=x_hyperedges,
            y_hyperedges=y_hyperedges,
            y=y_hyperedges,
            x=torch.eye(incidence_hyperedges.size(0)),
        )
        data_list = [data]
        data, slices = self.collate(data_list)
        fs.torch_save(
            (
                data.to_dict(),
                slices,
                {},
                data.__class__,
            ),
            self.processed_paths[0],
        )

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        if self.version == "single_scale":
            return ["dataframe_226.csv"]
        elif self.version == "all_scales":
            return ["dataframe_4313.csv"]
        else:
            raise ValueError(f"Unknown version: {self.version}")

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        if self.version == "single_scale":
            return "data_226.pt"
        elif self.version == "all_scales":
            return "data_4313.pt"
        else:
            raise ValueError(f"Unknown version: {self.version}")

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
