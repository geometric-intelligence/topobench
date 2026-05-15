"""
Dataset class for the Bowen et al. mouse auditory cortex calcium imaging dataset.

This script downloads and processes the original dataset introduced in:

[Citation] Bowen et al. (2024), "Fractured columnar small-world functional network
organization in volumes of L2/3 of mouse auditory cortex," PNAS Nexus, 3(2): pgae074.
https://doi.org/10.1093/pnasnexus/pgae074

We apply the preprocessing and graph-construction steps defined in this module to obtain
a representation of neuronal activity suitable for our experiments.

Please cite the original paper when using this dataset or any derivatives.
"""

import os
import os.path as osp
import shutil
from typing import ClassVar

import numpy as np
import pandas as pd
import scipy.io
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs
from torch_geometric.utils import to_undirected

from topobench.data.utils import download_file_from_link
from topobench.data.utils.io_utils import collect_mat_files, process_mat


class A123CortexMDataset(InMemoryDataset):
    """A1 and A2/3 mouse auditory cortex dataset.

    Loads neural correlation data from mouse auditory cortex regions. Supports:

    1. Graph Classification: Predict frequency bin (0-8) from graph structure

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset including corr_threshold,
        n_bins, and min_neurons.

    Attributes
    ----------
    URLS : dict
        Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT : dict
        Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES : list[str]
        List containing some of the raw file names for the dataset, to check after extraction.
    """

    URLS: ClassVar = {
        "Auditory cortex data": "https://gcell.umd.edu/data/Auditory_cortex_data.zip",
    }

    FILE_FORMAT: ClassVar = {
        "Auditory cortex data": "zip",
    }

    RAW_FILE_NAMES: ClassVar = [
        "031020_367n_100um20st_FRA",
        "ant.m",
        "README.txt",
    ]

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # defensive parameter access with sensible defaults
        self.corr_threshold = float(parameters.get("corr_threshold", 0.2))

        self.n_bins = int(parameters.get("n_bins", 9))

        self.min_neurons = int(parameters.get("min_neurons", 8))

        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4
        data, self.slices, self.sizes, data_cls = out

        self.data = data_cls.from_dict(data)

        # For this dataset we don't assume the internal _data is a torch_geometric Data
        # (this dataset exposes helper methods to construct subgraphs on demand).

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Path to the processed directory of the dataset.

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
        return self.RAW_FILE_NAMES

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
        """Download the dataset from a URL and extract to the raw directory."""
        # Download data from the source
        dataset_key = "Auditory cortex data"
        url = self.URLS[dataset_key]
        file_format = self.FILE_FORMAT[dataset_key]

        # Use self.name as the downloadable dataset name
        download_file_from_link(
            file_link=url,
            path_to_save=self.raw_dir,
            dataset_name=self.name,
            file_format=file_format,
            verify=False,
            timeout=60,  # 60 seconds per chunk read timeout
            retries=3,  # Retry up to 3 times
        )

        # Extract zip file
        filename = f"{self.name}.{file_format}"
        path = osp.join(self.raw_dir, filename)
        extract_zip(path, self.raw_dir)
        # Delete zip file
        os.unlink(path)

        extracted_path = osp.join(self.raw_dir, "Auditory cortex data")
        if osp.exists(extracted_path):
            for file in os.listdir(extracted_path):
                shutil.move(
                    osp.join(extracted_path, file),
                    osp.join(self.raw_dir, file),
                )
            shutil.rmtree(extracted_path)

    @staticmethod
    def extract_samples(data_dir: str, n_bins: int, min_neurons: int = 8):
        """Extract subgraph samples from raw .mat files.

        Parameters
        ----------
        data_dir : str
            Directory containing the raw .mat files.
        n_bins : int
            Number of frequency bins to use for binning.
        min_neurons : int, optional
            Minimum number of neurons required per sample. Defaults to 8.

        Returns
        -------
        pd.DataFrame
            DataFrame containing extracted samples with columns for
            session_file, session_id, layer, bf_bin, neuron_indices,
            corr, and noise_corr.
        """
        mat_files = collect_mat_files(data_dir)

        samples = []
        session_id = 0
        for f in mat_files:
            print(f"Processing session {session_id}: {os.path.basename(f)}")
            mt = process_mat(scipy.io.loadmat(f))
            for layer in range(1, 6):
                scorrs = np.array(mt["selectZCorrInfo"]["SigCorrs"])
                ncorrs = np.array(mt["selectZCorrInfo"]["NoiseCorrsTrial"])
                bfvals = np.array(mt["BFInfo"][layer]["BFval"]).ravel()

                bin_ids = bfvals.astype(int)

                for bin_idx in range(n_bins):
                    sel = np.where(bin_ids == bin_idx)[0]
                    if len(sel) < min_neurons:
                        continue
                    subcorr = scorrs[np.ix_(sel, sel)]
                    samples.append(
                        {
                            "session_file": f,
                            "session_id": session_id,
                            "layer": layer,
                            "bf_bin": int(bin_idx),
                            "neuron_indices": sel.tolist(),
                            "corr": subcorr.astype(float),
                            "noise_corr": ncorrs[np.ix_(sel, sel)].astype(
                                float
                            ),
                        }
                    )
            session_id += 1

        samples = pd.DataFrame(samples)
        return samples

    def _sample_to_pyg_data(
        self, sample: dict, threshold: float = 0.2
    ) -> Data:
        """Convert a sample dictionary to a PyTorch Geometric Data object.

        Converts correlation matrices to graph representation with node features
        and edges for graph-level classification tasks.

        Parameters
        ----------
        sample : dict
            Sample dictionary containing 'corr', 'noise_corr', 'session_id',
            'layer', and 'bf_bin' keys.
        threshold : float, optional
            Correlation threshold for creating edges. Defaults to 0.2.

        Returns
        -------
        torch_geometric.data.Data
            Data object with node features [mean_corr, std_corr, noise_diag],
            edges from thresholded correlation, and label y as integer bf_bin.
        """
        corr = np.asarray(sample.get("corr"))
        n = corr.shape[0]
        # sanitize
        corr = np.nan_to_num(corr)

        mean_corr = corr.mean(axis=1)
        std_corr = corr.std(axis=1)
        noise_diag = np.zeros(n)

        x_np = np.vstack([mean_corr, std_corr, noise_diag]).T
        x = torch.tensor(x_np, dtype=torch.float)

        # build edges from thresholded correlation (upper triangle)
        adj = (corr >= threshold).astype(int)
        iu = np.triu_indices(n, k=1)
        sel = np.where(adj[iu] == 1)[0]
        rows = iu[0][sel]
        cols = iu[1][sel]
        edge_index_np = np.vstack([rows, cols])
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # make undirected
        edge_index = to_undirected(edge_index)
        # edge_attr: corresponding corr weights (for both directions, if made undirected)
        weights = corr[rows, cols]
        weights = (
            np.repeat(weights, 2)
            if edge_index.size(1) == weights.size * 2
            else weights
        )
        edge_attr = torch.tensor(weights.reshape(-1, 1), dtype=torch.float)

        y = torch.tensor([int(sample.get("bf_bin", -1))], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        # attach metadata
        data.session_id = int(sample.get("session_id", -1))
        data.layer = int(sample.get("layer", -1))
        return data

    def process(self) -> None:
        """Generate raw files into collated PyG dataset and save to disk.

        This implementation mirrors other datasets in the repo: it calls the
        static helper `extract_samples()` to enumerate subgraphs, converts each
        to a `torch_geometric.data.Data` object via `_sample_to_pyg_data()`,
        optionally computes/attaches topology vectors, collates and saves.
        """
        data_dir = self.raw_dir

        print(f"[A123] Processing dataset from: {data_dir}")
        print(f"[A123] Files in raw_dir: {os.listdir(data_dir)}")

        # extract sample descriptions
        print("[A123] Starting extract_samples()...")
        samples = A123CortexMDataset.extract_samples(
            data_dir, self.n_bins, self.min_neurons
        )

        print(f"[A123] Extracted {len(samples)} samples")

        data_list = []
        skipped_count = 0
        for idx, (_, s) in enumerate(samples.iterrows()):
            if idx % 100 == 0:
                print(
                    f"[A123] Converting sample {idx}/{len(samples)} to PyG Data..."
                )
            d = self._sample_to_pyg_data(s, threshold=self.corr_threshold)
            data_list.append(d)

        # collate and save processed dataset
        print(
            f"[A123] Collating {len(data_list)} samples (removed {skipped_count} empty graphs)..."
        )
        self.data, self.slices = self.collate(data_list)
        self._data_list = None
        print(f"[A123] Saving processed data to {self.processed_paths[0]}...")
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
        print("[A123] Processing complete!")
