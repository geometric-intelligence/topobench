import os

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_integer_dtype
from torch_geometric.data import Data, InMemoryDataset

from .repository.zenodo import ZenodoZip

ZENODO_RECORD_ID = "16895532"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"

class GraphlandDataset(InMemoryDataset):
    """
    Example InMemoryDataset that:
      - (Optionally) downloads and extracts a ZIP (e.g., from Zenodo) into raw_dir
      - Reads nodes.csv (features + labels) and edges.csv (src, dst)
      - Builds a single large graph (Data) and stores it as processed <split>.pt
      - Supports pre_filter and pre_transform hooks

    Directory layout (after download/extract):
      root/
        raw/
          nodes.csv
          edges.csv
          READY  (sentinel file created after successful download/extract)
        processed/
          full.pt  (or <split>.pt if you change `split`)
    """

    def __init__(
        self,
        root: str | os.PathLike,
        name: str,
        drop_missing_y = True,
        impute_missing_x = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.zip_url = f"{ZENODO_BASE}{name}.zip?"
        self.name = name
        self.drop_missing_y = drop_missing_y
        self.impute_missing_x = impute_missing_x
        super().__init__(os.path.join(root, name), transform, pre_transform, pre_filter)

        # After super().__init__, processed file must exist. Load it:
        self.data, self.slices = torch.load(os.path.join(self.processed_paths[0], self.processed_file_names))


    def download(self) -> None:
        """
        If `raw_file_names` are missing, PyG calls this.
        """
        downloader = ZenodoZip(
            url = self.zip_url
        )
        data = downloader.fetch()

        os.makedirs(self.raw_dir, exist_ok=True)

        for file_path, binary_content in data.items():
            # taking only the filename
            filename = file_path.split("/")[-1]
            complete_file_path = os.path.join(self.raw_dir, filename)
            with open(complete_file_path, "wb") as f:   # use wb since values are binary
                f.write(binary_content)

    # ---------- Building the graph(s) ----------

    def process(self) -> None:
        """
        Create and save processed tensors. For InMemoryDataset
        """
        # Reading 'csv's
        edges_df = pd.read_csv(os.path.join(self.raw_dir, "edgelist.csv"))
        feats_df = pd.read_csv(os.path.join(self.raw_dir, "features.csv"), index_col="node_id")
        targs_df = pd.read_csv(os.path.join(self.raw_dir, "targets.csv"))

        # Imputing missing values in X
        if self.impute_missing_x is not None:
            x_numpy = self.impute_missing_x.fit_transform(feats_df)
        else:
            x_numpy = feats_df.values

        # creating X tensor
        x = torch.tensor(x_numpy, dtype=torch.float)

        # create y tensor (assuming node_id is the dataframe)
        targs_df = targs_df.set_index("node_id")
        # transforming to series
        targ_values = targs_df.squeeze()

        # inferring data type (NaN cannot be integer)
        if is_integer_dtype(targ_values.fillna(0)) \
            or \
                targ_values.fillna(0).apply(float.is_integer).all():
            y = torch.tensor(targs_df.values, dtype=torch.long).squeeze() # classification
        else:
            y = torch.tensor(targs_df.values, dtype=torch.double).squeeze() # regression

        if self.drop_missing_y:
            mask = ~torch.tensor(targ_values.isna().values)
            x = x[mask]
            y = y[mask]
            feats_df = feats_df[mask.numpy()]
            # filter edges to keep only nodes still present
            old_to_new = {old: new for new, old in enumerate(mask.numpy().nonzero()[0])}

            edges_df = edges_df[
                edges_df["source"].isin(old_to_new.keys()) &
                edges_df["target"].isin(old_to_new.keys())
            ].copy()

            # remap old indices to new consecutive indices
            edges_df["source"] = edges_df["source"].map(old_to_new)
            edges_df["target"] = edges_df["target"].map(old_to_new)

        # creating the edge indexes
        src = edges_df["source"].to_numpy()
        dst = edges_df["target"].to_numpy()
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_filter is not None and not self.pre_filter(data):
            data_list = []
        else:
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list = [data]

        # collate
        data_big, slices = self.collate(data_list)

        # creating the folder
        os.makedirs(self.processed_paths[0], exist_ok=True)

        #saving
        torch.save(
            (data_big, slices),
            os.path.join(self.processed_paths[0], self.processed_file_names)
            )


    # ---------- Required properties ----------

    @property
    def raw_file_names(self):
        """
        Files that must be present in raw_dir for "download() not needed".
        If you don't know them upfront (e.g., a ZIP with many files), use a
        sentinel. We'll create 'READY' after extracting to signal completeness.
        """
        return ["edgelist.csv", "features.csv", "targets.csv"]

    @property
    def processed_paths(self):
        """The processed path to avoid processing."""
        return [ os.path.join(self.root, "processed" ) ]

    @property
    def processed_file_names(self):
        """The processed file produced by `process()`."""
        return "data.pt"
