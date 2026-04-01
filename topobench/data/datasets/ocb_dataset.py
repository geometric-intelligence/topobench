"""OCB Circuit Benchmark Datasets for TopoBench."""

import os
import pickle
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.download import download_url
from tqdm import tqdm

# Node type count for one-hot encoding, from OCB/src/utils_src.py
NUM_NODE_TYPES = 10


class OCBDataset(InMemoryDataset):
    """Base class for OCB datasets.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    name : str
        Name of the dataset.
    parameters : dict | None, optional
        Additional parameters for the dataset, by default None.
    transform : Callable | None, optional
        A function/transform that takes in an :obj:`torch_geometric.data.Data`
        object and returns a transformed version. The data object will be
        transformed before every access.
    pre_transform : Callable | None, optional
        A function/transform that takes in an :obj:`torch_geometric.data.Data`
        object and returns a transformed version. The data object will be
        transformed once before being saved to disk.
    """

    def __init__(
        self,
        root: str,
        name: str,
        parameters: dict | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.name = name
        self.parameters = parameters or {}
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def url_prefix(self) -> str:
        """Return the URL prefix for downloading raw data files.

        Raises
        ------
        NotImplementedError
            If not implemented in subclass.

        Returns
        -------
        str
            The URL prefix.
        """
        raise NotImplementedError("url_prefix must be defined in subclass")

    @property
    def num_node_features(self) -> int:
        """Return the number of node features in the dataset.

        Returns
        -------
        int
            Number of node features.
        """
        # One-hot encoding of 10 node types + 1 feature value
        return NUM_NODE_TYPES + 1

    @property
    def num_edge_features(self) -> int:
        """Return the number of edge features in the dataset.

        Returns
        -------
        int
            Number of edge features.
        """
        return 0  # No edge features in the original data

    @property
    def num_classes(self) -> int:
        """Return the number of classes (regression targets) in the dataset.

        Returns
        -------
        int
            Number of classes.
        """
        # Single-target regression: 'fom'
        return 1

    @property
    def processed_file_names(self) -> list[str]:
        """Return the names of the files in the `processed` directory.

        Returns
        -------
        list[str]
            List of processed file names.
        """
        return ["data.pt"]

    def download(self):
        """Download the raw data files from the specified URLs."""
        print(f"Downloading data for {self.name}...")
        for name in self.raw_file_names:
            download_url(f"{self.url_prefix}/{name}", self.raw_dir)

    def process(self):
        r"""Process the raw OCB data.

        This method loads the raw OCB data, converts it into a list of
        torch_geometric.data.Data objects, and saves the processed data to disk.
        """
        print(f"Processing raw data for {self.name}...")

        # Load raw data
        pkl_path = os.path.join(self.raw_dir, "ckt_bench_101.pkl")
        csv_path = os.path.join(self.raw_dir, "perform101.csv")

        with open(pkl_path, "rb") as f:
            all_igraph_data = pickle.load(f)

        perform_df = pd.read_csv(csv_path)

        # Combine train and test sets
        combined_igraph_list = all_igraph_data[0] + all_igraph_data[1]

        data_list = []
        pbar = tqdm(
            total=len(combined_igraph_list),
            desc=f"Converting {self.name} graphs",
        )

        for i, (_g_sort, g_all_sort) in enumerate(combined_igraph_list):
            g = g_all_sort  # Use the g_all_sort graph

            # Extract edge_index
            edges = np.array(g.get_edgelist()).T
            edge_index = torch.tensor(edges, dtype=torch.long)

            # Extract node features (x)
            node_types = torch.tensor(g.vs["type"], dtype=torch.long)
            node_type_one_hot = F.one_hot(
                node_types, num_classes=NUM_NODE_TYPES
            ).to(torch.float)
            node_feats = torch.tensor(
                g.vs["feat"], dtype=torch.float
            ).unsqueeze(1)
            x = torch.cat([node_type_one_hot, node_feats], dim=1)

            # Extract graph-level targets (y) and other info
            perf_row = perform_df.iloc[i]
            # Using 'fom' as the single regression target to comply with the original evaluator.
            y = torch.tensor([perf_row["fom"]], dtype=torch.float)

            vid = torch.tensor(g.vs["vid"], dtype=torch.long)
            valid = torch.tensor([perf_row["valid"]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y, vid=vid, valid=valid)

            if g.vcount() == 0:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)

            data_list.append(data)
            pbar.update(1)

        pbar.close()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(
            f"Processing complete! {len(data_list)} graphs saved to '{self.processed_paths[0]}'"
        )

    def get_target_statistics(self) -> dict[str, float]:
        """Get statistics of regression targets.

        Returns
        -------
        dict[str, float]
            A dictionary containing 'mean', 'std', 'min', 'max' of the targets.
        """
        all_targets = torch.cat([self[i].y for i in range(len(self))])
        return {
            "mean": float(all_targets.mean()),
            "std": float(all_targets.std()),
            "min": float(all_targets.min()),
            "max": float(all_targets.max()),
        }


class OCB101Dataset(OCBDataset):
    """OCB101: Open Circuit Benchmark 101 (10K circuits).

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="OCB101", **kwargs)

    @property
    def url_prefix(self) -> str:
        """Return the URL prefix for downloading raw data files for OCB101.

        Returns
        -------
        str
            The URL prefix.
        """
        return "https://raw.githubusercontent.com/zehao-dong/CktGNN/main/OCB/CktBench101"

    @property
    def raw_file_names(self) -> list[str]:
        """Return the names of the raw data files for OCB101.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["ckt_bench_101.pkl", "perform101.csv"]


class OCB301Dataset(OCBDataset):
    """OCB301: Open Circuit Benchmark 301 (30.1K circuits).

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="OCB301", **kwargs)

    @property
    def url_prefix(self) -> str:
        """Return the URL prefix for downloading raw data files for OCB301.

        Returns
        -------
        str
            The URL prefix.
        """
        return "https://raw.githubusercontent.com/zehao-dong/CktGNN/main/OCB/CktBench301"

    @property
    def raw_file_names(self) -> list[str]:
        """Return the names of the raw data files for OCB301.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return ["ckt_bench_301.pkl.zip", "perform301.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the names of the files in the `processed` directory for OCB301.

        Returns
        -------
        list[str]
            List of processed file names.
        """
        return ["data_301.pt"]

    def process(self):
        r"""Process the raw OCB301 data.

        This method unzips and loads the raw OCB301 data, converts it into a
        list of torch_geometric.data.Data objects, and saves the processed
        data to disk.
        """
        print(f"Processing raw data for {self.name}...")

        # Unzip the data
        zip_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        # Paths to the extracted files
        pkl_path = os.path.join(self.raw_dir, "ckt_bench_301.pkl")
        csv_path = os.path.join(self.raw_dir, self.raw_file_names[1])

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Extracted file 'ckt_bench_301.pkl' not found. "
                f"Please check the contents of {self.raw_file_names[0]}."
            )

        with open(pkl_path, "rb") as f:
            all_igraph_data = pickle.load(f)

        perform_df = pd.read_csv(csv_path)

        # OCB301's pickle file is a single list of graphs
        combined_igraph_list = all_igraph_data

        data_list = []
        pbar = tqdm(
            total=len(combined_igraph_list),
            desc=f"Converting {self.name} graphs",
        )

        for i, (_g_sort, g_all_sort) in enumerate(combined_igraph_list):
            g = g_all_sort

            edges = np.array(g.get_edgelist()).T
            edge_index = torch.tensor(edges, dtype=torch.long)

            node_types = torch.tensor(g.vs["type"], dtype=torch.long)
            node_type_one_hot = F.one_hot(
                node_types, num_classes=NUM_NODE_TYPES
            ).to(torch.float)
            node_feats = torch.tensor(
                g.vs["feat"], dtype=torch.float
            ).unsqueeze(1)
            x = torch.cat([node_type_one_hot, node_feats], dim=1)

            # Using 'fom' as the single regression target
            y = torch.tensor([perform_df.iloc[i]["fom"]], dtype=torch.float)

            vid = torch.tensor(g.vs["vid"], dtype=torch.long)
            valid = torch.tensor(
                [perform_df.iloc[i]["valid"]], dtype=torch.long
            )

            data = Data(x=x, edge_index=edge_index, y=y, vid=vid, valid=valid)

            if g.vcount() == 0:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)

            data_list.append(data)
            pbar.update(1)

        pbar.close()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(
            f"Processing complete! {len(data_list)} graphs saved to '{self.processed_paths[0]}'"
        )
