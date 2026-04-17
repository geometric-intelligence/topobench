"""Dataset class MANTRA dataset."""

import os
import os.path as osp
from typing import ClassVar

from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_gz
from torch_geometric.io import fs

from topobench.data.utils import (
    download_file_from_link,
    read_ndim_manifolds,
)


class MantraDataset(InMemoryDataset):
    r"""Dataset class for MANTRA manifold dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {
        "2_manifolds": "https://github.com/aidos-lab/mantra/releases/download/{version}/2_manifolds.json.gz",
        "3_manifolds": "https://github.com/aidos-lab/mantra/releases/download/{version}/3_manifolds.json.gz",
    }

    FILE_FORMAT: ClassVar = {
        "2_manifolds": "json.gz",
        "3_manifolds": "json.gz",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
        **kwargs,
    ) -> None:
        self.parameters = parameters
        self.manifold_dim = parameters.manifold_dim
        self.version = parameters.version
        self.task_variable = parameters.task_variable
        self.load_as_graph = parameters.load_as_graph
        self.name = "_".join(
            [name, str(self.version), f"manifold_dim_{self.manifold_dim}"]
        )
        if kwargs.get("slice"):
            self.slice = 100
        else:
            self.slice = None
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
        return osp.join(
            self.root,
            self.name,
            "raw",
        )

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
            self.task_variable,
        )
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return [f"{self.manifold_dim}_manifolds.json"]

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
        self.url = self.URLS[f"{self.manifold_dim}_manifolds"].format(
            version=self.version
        )
        self.file_format = self.FILE_FORMAT[f"{self.manifold_dim}_manifolds"]
        dataset_name = f"{self.manifold_dim}_manifolds"

        download_file_from_link(
            file_link=self.url,
            path_to_save=self.raw_dir,
            dataset_name=dataset_name,
            file_format=self.file_format,
        )

        # Extract zip file
        folder = self.raw_dir
        filename = f"{dataset_name}.{self.file_format}"
        path = osp.join(folder, filename)
        extract_gz(path, folder)

        # Delete zip file
        os.unlink(path)

        # # Move files from osp.join(folder, name_download) to folder
        # for file in os.listdir(osp.join(folder, self.name)):
        #     shutil.move(osp.join(folder, self.name, file), folder)
        # # Delete osp.join(folder, self.name) dir
        # shutil.rmtree(osp.join(folder, self.name))

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the JSON file for MANTRA for the specified manifold
        dimension, applies the respective preprocessing if specified and saves
        the preprocessed data to the appropriate location.
        """

        data = read_ndim_manifolds(
            # TODO Fix this
            osp.join(self.raw_dir, self.raw_file_names[0]),
            self.manifold_dim,
            self.task_variable,
            self.slice,
            self.load_as_graph,
        )

        data_list = data
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
