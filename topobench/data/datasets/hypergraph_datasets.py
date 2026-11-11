"""Dataset class for US County Demographics dataset."""

import os
import os.path as osp
import shutil
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobench.data.utils import (
    download_file_from_drive,
    load_hypergraph_content_dataset,
)


class HypergraphDataset(InMemoryDataset):
    r"""Dataset class for Hypergaph dataset.

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
        "ModelNet40": "https://drive.google.com/file/d/1u3-SFCjOIh1G0U8pVclfGIlDCceJ0qxr/view?usp=drive_link",
        "NTU2012": "https://drive.google.com/file/d/1g9P-uEVSATg6B_JRnyey78YbliIfst3Z/view?usp=drive_link",
        "Mushroom": "https://drive.google.com/file/d/1iad2l9w58UJvMMXOz6PtrbZkvGyFjWK6/view?usp=drive_link",
        "20newsW100": "https://drive.google.com/file/d/1D1NtyS4g9LZJPlnxOOySGlRR2km1wGMm/view?usp=drive_link",
        "zoo": "https://drive.google.com/file/d/18TuuGv3qiBfU-wqB3USB3HiiI9G-8X71/view?usp=drive_link",
    }

    FILE_FORMAT: ClassVar = {
        "ModelNet40": "zip",
        "NTU2012": "zip",
        "Mushroom": "zip",
        "20newsW100": "zip",
        "zoo": "zip",
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
        # self.year = parameters.year
        # self.task_variable = parameters.task_variable
        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        data, self.slices, self.sizes, data_cls = out

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
        return []  # ["county_graph.csv", f"county_stats_{self.year}.csv"]

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
        # Step 1: Download data from the source
        self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]

        download_file_from_drive(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=self.file_format,
        )
        # Extract zip file
        folder = self.raw_dir
        filename = f"{self.name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_zip(path, folder)
        # Delete zip file
        os.unlink(path)

        # Move files from osp.join(folder, name_download) to folder.
        # Overwrite leftovers from earlier downloads: some hosted
        # archives contain junk directories (e.g. .ipynb_checkpoints),
        # and moving onto an existing directory nests it and raises.
        for file in os.listdir(osp.join(folder, self.name)):
            dst = osp.join(folder, file)
            if osp.isdir(dst):
                shutil.rmtree(dst)
            elif osp.exists(dst):
                os.unlink(dst)
            shutil.move(osp.join(folder, self.name, file), dst)
        # Delete osp.join(folder, self.name) dir
        shutil.rmtree(osp.join(folder, self.name))

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the US county demographics data, applies any pre-
        processing transformations if specified, and saves the processed data
        to the appropriate location.
        """

        data, _ = load_hypergraph_content_dataset(
            data_dir=self.raw_dir, data_name=self.name
        )

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
