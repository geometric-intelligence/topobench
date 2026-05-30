import os

import numpy as np
import pandas as pd
import torch
import yaml
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, InMemoryDataset

from .repository.zenodo import ZenodoZip

ZENODO_RECORD_ID = "16895532"
ZENODO_BASE = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"
)


class GraphlandDataset(InMemoryDataset):
    """GraphLand benchmark dataset for node property prediction.

    Downloads and processes datasets from the GraphLand benchmark
    (arXiv:2409.14500). Handles categorical features via one-hot
    encoding and numerical features via optional imputation,
    following the preprocessing pipeline recommended by the
    GraphLand authors.

    Parameters
    ----------
    root : str or os.PathLike
        Root directory for storing raw and processed data.
    name : str
        Name of the GraphLand dataset (e.g., 'tolokers-2').
    drop_missing_y : bool, optional
        Whether to drop nodes with missing target values.
        Default is True.
    impute_missing_x : object or None, optional
        Sklearn-compatible imputer for missing numerical features.
        Applied only to numerical (non-categorical) features.
        Default is None.
    transform : callable, optional
        PyG transform applied at access time.
    pre_transform : callable, optional
        PyG transform applied during processing.
    pre_filter : callable, optional
        PyG filter applied during processing.

    Notes
    -----
    Feature preprocessing follows the GraphLand paper:

    - Categorical features are one-hot encoded using
      ``OneHotEncoder(drop='if_binary')``, which avoids
      redundant columns for binary features.
    - Numerical features are optionally imputed (default:
      most-frequent imputation).
    - Fraction features (a subset of numerical, bounded [0,1])
      are included with numerical features.
    - Final feature matrix: [numerical | one_hot_categorical].
    - The ``info.yaml`` bundled with each dataset specifies
      which features are categorical vs numerical.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        name: str,
        drop_missing_y=True,
        impute_missing_x=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.zip_url = f"{ZENODO_BASE}{name}.zip?"
        self.name = name
        self.drop_missing_y = drop_missing_y
        self.impute_missing_x = impute_missing_x
        super().__init__(
            os.path.join(root, name),
            transform,
            pre_transform,
            pre_filter,
        )

        self.data, self.slices = torch.load(
            os.path.join(
                self.processed_paths[0],
                self.processed_file_names,
            )
        )

    def download(self) -> None:
        """Download dataset ZIP from Zenodo and extract to raw_dir."""
        downloader = ZenodoZip(url=self.zip_url)
        data = downloader.fetch()

        os.makedirs(self.raw_dir, exist_ok=True)

        for file_path, binary_content in data.items():
            filename = file_path.split("/")[-1]
            complete_file_path = os.path.join(
                self.raw_dir, filename
            )
            with open(complete_file_path, "wb") as f:
                f.write(binary_content)

    def _load_info(self) -> dict:
        """Load dataset metadata from info.yaml.

        Returns
        -------
        dict
            Dataset metadata with keys including
            'categorical_features_names',
            'numerical_features_names', and
            'fraction_features_names'.
        """
        info_path = os.path.join(self.raw_dir, "info.yaml")
        with open(info_path) as f:
            return yaml.safe_load(f)

    def _preprocess_features(
        self, feats_df: pd.DataFrame, info: dict
    ) -> np.ndarray:
        """Preprocess features following GraphLand conventions.

        Separates features into numerical and categorical types
        based on info.yaml metadata. Numerical features are
        optionally imputed. Categorical features are one-hot
        encoded with ``drop='if_binary'``.

        Parameters
        ----------
        feats_df : pd.DataFrame
            Raw feature DataFrame indexed by node_id.
        info : dict
            Dataset metadata from info.yaml.

        Returns
        -------
        np.ndarray
            Preprocessed feature matrix of shape
            [num_nodes, num_processed_features].
        """
        cat_names = info.get("categorical_features_names", [])
        num_names = info.get("numerical_features_names", [])

        # Numerical features (includes fraction features)
        num_cols = [c for c in num_names if c in feats_df.columns]
        numerical = feats_df[num_cols].values.astype(np.float32)

        if self.impute_missing_x is not None and len(num_cols) > 0:
            numerical = self.impute_missing_x.fit_transform(
                numerical
            )

        # Categorical features: one-hot encode
        cat_cols = [c for c in cat_names if c in feats_df.columns]
        if len(cat_cols) > 0:
            categorical_raw = (
                feats_df[cat_cols].values.astype(np.float32)
            )
            encoder = OneHotEncoder(
                drop="if_binary",
                sparse_output=False,
                dtype=np.float32,
            )
            categorical_encoded = encoder.fit_transform(
                categorical_raw
            )
        else:
            categorical_encoded = np.empty(
                (len(feats_df), 0), dtype=np.float32
            )

        # Concatenate: [numerical | one_hot_categorical]
        return np.concatenate(
            [numerical, categorical_encoded], axis=1
        )

    def process(self) -> None:
        """Build processed graph data from raw CSV files.

        Reads features, targets, and edges from raw CSVs.
        Applies feature preprocessing (imputation + one-hot
        encoding) based on info.yaml metadata. Optionally
        drops nodes with missing targets.
        """
        edges_df = pd.read_csv(
            os.path.join(self.raw_dir, "edgelist.csv")
        )
        feats_df = pd.read_csv(
            os.path.join(self.raw_dir, "features.csv"),
            index_col="node_id",
        )
        targs_df = pd.read_csv(
            os.path.join(self.raw_dir, "targets.csv")
        )

        # Load info.yaml for feature type metadata
        info = self._load_info()

        # Preprocess features with proper categorical handling
        x_numpy = self._preprocess_features(feats_df, info)
        x = torch.tensor(x_numpy, dtype=torch.float)

        # Build target tensor
        targs_df = targs_df.set_index("node_id")
        targ_values = targs_df.squeeze()

        if is_integer_dtype(
            targ_values.fillna(0)
        ) or targ_values.fillna(0).apply(float.is_integer).all():
            y = torch.tensor(
                targs_df.values, dtype=torch.long
            ).squeeze()
        else:
            y = torch.tensor(
                targs_df.values, dtype=torch.double
            ).squeeze()

        # Drop nodes with missing targets
        if self.drop_missing_y:
            mask = ~torch.tensor(targ_values.isna().values)
            x = x[mask]
            y = y[mask]
            feats_df = feats_df[mask.numpy()]

            old_to_new = {
                old: new
                for new, old in enumerate(
                    mask.numpy().nonzero()[0]
                )
            }

            edges_df = edges_df[
                edges_df["source"].isin(old_to_new.keys())
                & edges_df["target"].isin(old_to_new.keys())
            ].copy()

            edges_df["source"] = edges_df["source"].map(
                old_to_new
            )
            edges_df["target"] = edges_df["target"].map(
                old_to_new
            )

        # Build edge index
        src = edges_df["source"].to_numpy()
        dst = edges_df["target"].to_numpy()
        edge_index = torch.tensor(
            np.array([src, dst]), dtype=torch.long
        )

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_filter is not None and not self.pre_filter(
            data
        ):
            data_list = []
        else:
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list = [data]

        data_big, slices = self.collate(data_list)

        os.makedirs(self.processed_paths[0], exist_ok=True)

        torch.save(
            (data_big, slices),
            os.path.join(
                self.processed_paths[0],
                self.processed_file_names,
            ),
        )

    @property
    def raw_file_names(self):
        """Files required in raw_dir before processing."""
        return [
            "edgelist.csv",
            "features.csv",
            "targets.csv",
            "info.yaml",
        ]

    @property
    def processed_paths(self):
        """The processed data directory path."""
        return [os.path.join(self.root, "processed")]

    @property
    def processed_file_names(self):
        """The processed data filename."""
        return "data.pt"
