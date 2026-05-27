"""Dataset class for Twitch dataset."""

import json
import os
import os.path as osp
from typing import ClassVar

import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobench.data.utils import (
    download_file_from_link,
)


class TwitchDataset(InMemoryDataset):
    r"""Dataset class for Twitch dataset (language-specific).

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Language (e.g. "EN", "DE") used to select files.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Notes
    -----
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    LANGUAGE_MAP (dict): Dictionary containing the mapping from language codes to folder name in the Dataset.
    """

    URLS: ClassVar = {
        "twitch": "https://snap.stanford.edu/data/twitch.zip",
    }
    FILE_FORMAT: ClassVar = {"twitch": "zip"}

    # Mapping torch_geometric language codes to folder names in ZIP
    LANGUAGE_MAP: ClassVar = {
        "EN": "ENGB",
        "PT": "PTBR",
    }

    def __init__(self, root: str, name: str, parameters: DictConfig) -> None:
        self.name = name.upper()  # Ensure language code is uppercase
        self.parameters = parameters
        assert self.name in ["DE", "EN", "ES", "FR", "PT", "RU"]
        # Map simplified language code
        self.mapped_name = self.LANGUAGE_MAP.get(self.name, self.name)

        super().__init__(root)

        # Load processed data
        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 3:
            data, self.slices, _ = out
            data_cls = Data
        else:
            data, self.slices, _, data_cls = out

        self.data = data if not isinstance(data, dict) else data_cls.from_dict(data)
        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"TwitchDataset(root={self.root}, language={self.name})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, "twitch", "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, f"twitch_{self.name}", "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset (language-specific).

        Returns
        -------
        list[str]
            List of raw file names.
        """
        # Files are stored in subfolder per language in the ZIP
        return [
            f"twitch/{self.mapped_name}/musae_{self.mapped_name}_edges.csv",
            f"twitch/{self.mapped_name}/musae_{self.mapped_name}_features.json",
            f"twitch/{self.mapped_name}/musae_{self.mapped_name}_target.csv",
        ]

    @property
    def processed_file_names(self) -> str:
        """Processed dataset file name."""
        return "data.pt"

    def download(self) -> None:
        r"""Download Twitch dataset (all languages in one zip)."""
        # Download zip file
        download_file_from_link(
            file_link=self.URLS["twitch"],
            path_to_save=self.raw_dir,
            dataset_name="twitch",
            file_format=self.FILE_FORMAT["twitch"],
        )

        # Extract downloaded zip
        zip_path = osp.join(self.raw_dir, "twitch.zip")
        extract_zip(zip_path, self.raw_dir)
        os.unlink(zip_path)  # Remove .zip after extraction

    def process(self) -> None:
        r"""Process Twitch dataset for a specific language.

        Step 1. Load edge list
        Step 2. Load target labels (binary whether channel streams mature content or not)
        Step 3. Load features
        Step 4. Create torch_geometric.data.Data object
        Step 5. Save processed dataset to disk
        """

        # 1. Load edge list
        edges = pd.read_csv(self.raw_paths[0])[["from", "to"]].to_numpy()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 2. Load labels
        target_raw = pd.read_csv(self.raw_paths[2])
        y = torch.tensor(pd.Categorical(target_raw["mature"]).codes, dtype=torch.long)
        
        # 3. Load node features
        with open(self.raw_paths[1]) as f:
            feat_dict = json.load(f)

        # Get feature vectors with padding
        x = []

        amount_features = 128
        for i in target_raw["id"].values:
            feature = [0] * amount_features

            if str(i) in feat_dict:
                n_len = len(feat_dict[str(i)])
                feature = feat_dict[str(i)] + [0] * (amount_features - n_len)

            x.append(feature)

        x = torch.tensor(x, dtype=torch.float)
        
        # 4. Create Data object
        data = Data(x=x, y=y, edge_index=edge_index)
        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache

        # 5. Save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
