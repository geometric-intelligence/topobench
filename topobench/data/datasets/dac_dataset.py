"""Dataset class for Dynamic Activity Complex (DAC) dataset."""

import os
import os.path as osp
import shutil
from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip

from topobench.data.utils import download_file_from_link


class DACDataset(InMemoryDataset):
    r"""Dataset class for the Dynamic Activity Complexes (DAC) dataset.

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
    """

    URLS: ClassVar = {
        "4-325-1": "https://zenodo.org/records/17700425/files/4_325_1.zip",
        "4-325-3": "https://zenodo.org/records/17700425/files/4_325_3.zip",
        "4-325-5": "https://zenodo.org/records/17700425/files/4_325_5.zip",
    }

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ):
        # Load processed data (created in process())
        self.name = name
        super().__init__(root)
        self.data, self.slices, self.splits = torch.load(
            self.processed_paths[0]
        )

        split_num = parameters.split_num
        self.split_idx = self.splits[split_num]

    @property
    def raw_file_names(self):
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return [
            "all_edges.pt",
            "all_x.pt",
            "y.pt",
            "split_0.pt",
            "split_1.pt",
            "split_2.pt",
            "split_3.pt",
            "split_4.pt",
        ]

    @property
    def processed_file_names(self):
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return ["data.pt"]

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(self.root)
        return osp.join(self.processed_root, "processed")

    def download(self):
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: Download data from the source
        self.url = self.URLS[self.name]
        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format="zip",
        )

        # Step 2: extract zip file
        folder = self.raw_dir
        filename = f"{self.name}.zip"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)

        # Step 3: organize files
        # Move files from osp.join(folder, name_download) to folder
        folder_name = "4_325_" + self.name.split("-")[2]
        for file in os.listdir(osp.join(folder, folder_name)):
            shutil.move(osp.join(folder, folder_name, file), folder)
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, folder_name))

    def process(self):
        r"""Handle the data for the dataset.

        This method loads the DAC raw data, creates one object for
        each graph, and saves the processed data
        to the appropriate location.
        """
        # Load raw tensors
        relations = torch.load(os.path.join(self.raw_dir, "all_edges.pt"))
        all_x = torch.load(os.path.join(self.raw_dir, "all_x.pt"))
        y = torch.load(os.path.join(self.raw_dir, "y.pt"))

        data_list = []
        for i in range(len(all_x)):
            # Create PyG Data object
            data = Data(
                x=all_x[i],
                edge_index=relations[i],
                y=y[i].unsqueeze(0) if y[i].ndim == 0 else y[i],
            )

            data_list.append(data)

        # Save to processed dir using slicing format
        data, slices = self.collate(data_list)

        splits = []
        for s in range(5):
            split = torch.load(os.path.join(self.raw_dir, f"split_{s}.pt"))
            splits.append(split)

        torch.save((data, slices, splits), self.processed_paths[0])
