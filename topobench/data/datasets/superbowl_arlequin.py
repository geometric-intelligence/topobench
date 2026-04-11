"""Dataset class for Superbowl Arlequin dataset (social media messages)."""

import hashlib
import json
import os.path as osp
from typing import ClassVar

import numpy as np
import pandas as pd
import toponetx as tnx
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import get_colored_hypergraph_connectivity


class SuperbowlArlequinDataset(InMemoryDataset):
    r"""Dataset class for Superbowl Arlequin dataset.

    Models social media messages around the Super Bowl as a colored
    hypergraph with author, thread, and semantic cluster hyperedges.
    Supports configurable semantic clustering depth and node feature modes.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.
    """

    URLS: ClassVar = {}

    FILE_FORMAT: ClassVar = {}

    RAW_FILE_NAMES: ClassVar = {
        "raw_data": "/data/gbg141/Arlequin/data/superbowl_raw_data/superbowl_raw_data.xlsx",
        "clusters": "/data/gbg141/Arlequin/data/superbowl_raw_data/superbowl_content_x_clusters.csv",
        "embeddings": "/data/gbg141/Arlequin/data/superbowl_raw_data/superbowl_content_with_embeddings (1).csv",
    }

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.max_rank = parameters.get("max_rank", 3)
        self.cluster_seed = parameters.get("cluster_seed", 42)
        self.neighborhoods = parameters.get("neighborhoods", None)
        self.semantic_depth = parameters.get("semantic_depth", 3)
        self.node_feature_mode = parameters.get("node_feature_mode", "embedding")
        self.min_posts_per_author = parameters.get("min_posts_per_author", 2)

        if self.neighborhoods is not None:
            neighborhoods_str = ",".join(sorted(self.neighborhoods))
            neighborhoods_hash = hashlib.md5(
                neighborhoods_str.encode()
            ).hexdigest()[:12]
        else:
            neighborhoods_hash = "none"
        self.hypergraph_id = (
            f"depth{self.semantic_depth}_{self.node_feature_mode}_"
            f"minauth{self.min_posts_per_author}_{self.max_rank}_"
            f"{self.cluster_seed}_{neighborhoods_hash}"
        )
        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return (
            f"{self.name}(root={self.root}, name={self.name}, "
            f"parameters={self.parameters})"
        )

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset."""
        return osp.join(self.root, self.name, self.hypergraph_id, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset."""
        return osp.join(self.root, self.name, self.hypergraph_id, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset."""
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset."""
        return "data.pt"

    def get_data_dir(self) -> str:
        """Return the path to the data directory."""
        return osp.join(self.root, self.name, self.hypergraph_id)

    def _load_and_merge_data(self):
        """Load all source files and merge into a single DataFrame.

        Returns
        -------
        df : pd.DataFrame
            Merged dataframe with columns: content_url, content_entity_id,
            user_source_id, parent, embedding, cluster_id.
        """
        print("Loading clusters data...")
        df_clusters = pd.read_csv(self.RAW_FILE_NAMES["clusters"])
        df_depth = df_clusters[
            df_clusters["layer_depth"] == self.semantic_depth
        ][["content_entity_id", "content_url", "cluster_id"]].copy()
        df_depth = df_depth.drop_duplicates(subset="content_entity_id")
        print(
            f"  Depth {self.semantic_depth}: "
            f"{len(df_depth)} messages, "
            f"{df_depth['cluster_id'].nunique()} clusters"
        )

        print("Loading embeddings data...")
        df_emb = pd.read_csv(self.RAW_FILE_NAMES["embeddings"])
        df_emb = df_emb.drop_duplicates(subset="content_entity_id")
        df_emb["embedding"] = df_emb["embedding"].apply(
            lambda x: np.array(json.loads(x), dtype=np.float32)
        )
        df_emb = df_emb[["content_entity_id", "content_url", "embedding"]]

        print("Loading raw metadata...")
        df_raw = pd.read_excel(
            self.RAW_FILE_NAMES["raw_data"],
            usecols=["id", "url", "user source id", "parent"],
        )
        df_raw = df_raw.rename(
            columns={
                "url": "content_url",
                "user source id": "user_source_id",
            }
        )

        # Merge on content_url
        df = df_depth.merge(df_emb, on=["content_entity_id", "content_url"])
        df = df.merge(df_raw, on="content_url")
        print(f"  Merged: {len(df)} messages")

        return df

    def _filter_by_author(self, df):
        """Filter to authors with at least min_posts_per_author posts.

        Returns
        -------
        df : pd.DataFrame
            Filtered dataframe.
        """
        posts_per_author = df.groupby("user_source_id").size()
        valid_authors = posts_per_author[
            posts_per_author >= self.min_posts_per_author
        ].index
        df = df[df["user_source_id"].isin(valid_authors)].reset_index(drop=True)
        print(
            f"  After author filter (>= {self.min_posts_per_author} posts): "
            f"{len(df)} messages, "
            f"{df['user_source_id'].nunique()} authors"
        )
        return df

    def _encode_author_labels(self, df):
        """Encode author IDs as contiguous integer labels.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with added 'author_label' column.
        author_map : dict
            Mapping from user_source_id to integer label.
        """
        unique_authors = sorted(df["user_source_id"].unique())
        author_map = {uid: i for i, uid in enumerate(unique_authors)}
        df["author_label"] = df["user_source_id"].map(author_map)
        return df, author_map

    def _remap_cluster_ids(self, df):
        """Remap cluster IDs to contiguous integers.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with 'cluster_label' column (contiguous 0..n_clusters-1).
        """
        unique_clusters = sorted(df["cluster_id"].unique())
        cluster_map = {cid: i for i, cid in enumerate(unique_clusters)}
        df["cluster_label"] = df["cluster_id"].map(cluster_map)
        return df

    def create_message_nodes(self, df):
        """Create message nodes and add to the complex.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered and merged dataframe.
        """
        self.messages = []
        self.id_messages = dict()
        for idx in range(len(df)):
            msg = Message(idx, df.at[idx, "author_label"], df.at[idx, "cluster_label"])
            self.messages.append(msg)
            self.complex.add_node(msg)
            self.id_messages[idx] = msg

    def build_author_hyperedges(self, df, rank=1):
        """Build author hyperedges grouping each author's messages.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered dataframe with 'author_label' column.
        rank : int, optional
            Rank of the hyperedges, by default 1.
        """
        self.author_to_msgs = dict()
        for idx in range(len(df)):
            author = df.at[idx, "author_label"]
            if author not in self.author_to_msgs:
                self.author_to_msgs[author] = []
            self.author_to_msgs[author].append(self.messages[idx])

        self.author_labels_ordered = sorted(self.author_to_msgs.keys())
        self.author_hyperedges = []
        for author in self.author_labels_ordered:
            msgs = self.author_to_msgs[author]
            he = tnx.HyperEdge(msgs, rank=rank)
            self.author_hyperedges.append(msgs)
            self.complex.add_cell(he, rank=rank)

        print(
            f"Author hyperedges (rank {rank}): {len(self.author_hyperedges)}"
        )

    def build_thread_hyperedges(self, df, rank=2):
        """Build thread hyperedges from conversation structure.

        Groups messages that share a root ancestor via the 'parent' column.
        Only threads with >1 message are included.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered dataframe with 'id' and 'parent' columns.
        rank : int, optional
            Rank of the hyperedges, by default 2.
        """
        all_msg_ids = set(df["id"].values)
        parent_dict = df.set_index("id")["parent"].to_dict()

        def find_root(msg_id):
            visited = set()
            current = msg_id
            while (
                current in parent_dict
                and str(parent_dict[current]) != "0"
                and parent_dict[current] in all_msg_ids
            ):
                if current in visited:
                    break
                visited.add(current)
                current = parent_dict[current]
            return current

        df_id_to_idx = {row["id"]: idx for idx, row in df.iterrows()}
        root_groups = dict()
        for idx in range(len(df)):
            msg_id = df.at[idx, "id"]
            root = find_root(msg_id)
            if root not in root_groups:
                root_groups[root] = []
            root_groups[root].append(idx)

        self.thread_hyperedges = []
        for root_id, indices in root_groups.items():
            if len(indices) > 1:
                thread_msgs = [self.messages[i] for i in indices]
                he = tnx.HyperEdge(thread_msgs, rank=rank)
                self.thread_hyperedges.append(thread_msgs)
                self.complex.add_cell(he, rank=rank)

        print(
            f"Thread hyperedges (rank {rank}): {len(self.thread_hyperedges)}"
        )

    def build_semantic_hyperedges(self, df, rank=3):
        """Build semantic cluster hyperedges.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered dataframe with 'cluster_label' column.
        rank : int, optional
            Rank of the hyperedges, by default 3.
        """
        self.n_clusters = df["cluster_label"].nunique()
        self.semantic_hyperedges = []
        for label in range(self.n_clusters):
            mask = df["cluster_label"].values == label
            cluster_msgs = np.array(self.messages)[mask]
            if len(cluster_msgs) > 0:
                he = tnx.HyperEdge(cluster_msgs, rank=rank)
                self.semantic_hyperedges.append(cluster_msgs)
                self.complex.add_cell(he, rank=rank)

        print(
            f"Semantic hyperedges (rank {rank}): "
            f"{len(self.semantic_hyperedges)} clusters "
            f"(depth {self.semantic_depth})"
        )

    def get_connectivity(self):
        """Build the colored hypergraph and return its connectivity.

        Returns
        -------
        connectivity : dict
            Connectivity matrices and metadata for the colored hypergraph.
        embeddings : np.ndarray
            Message embeddings array.
        df : pd.DataFrame
            The filtered and merged dataframe.
        """
        df = self._load_and_merge_data()
        df = self._filter_by_author(df)
        df, self.author_map = self._encode_author_labels(df)
        df = self._remap_cluster_ids(df)
        df = df.reset_index(drop=True)

        embeddings = np.stack(df["embedding"].values)
        self.n_authors = len(self.author_map)

        self.complex = tnx.ColoredHyperGraph()

        self.create_message_nodes(df)
        self.build_author_hyperedges(df, rank=1)
        self.build_thread_hyperedges(df, rank=2)
        self.build_semantic_hyperedges(df, rank=3)

        connectivity = get_colored_hypergraph_connectivity(
            self.complex,
            max_rank=self.max_rank,
            neighborhoods=self.neighborhoods,
        )
        return connectivity, embeddings, df

    def get_node_features_and_labels(self, embeddings, connectivity, df):
        """Get node features and labels for the dataset.

        Parameters
        ----------
        embeddings : np.ndarray
            Message embeddings of shape (n_messages, 1024).
        connectivity : dict
            Connectivity of the colored hypergraph.
        df : pd.DataFrame
            Filtered dataframe with cluster_label and author_label.

        Returns
        -------
        features_and_labels : dict
            Features and labels for each rank.
        """
        features_and_labels = dict()

        # Rank 0: message features
        if self.node_feature_mode == "embedding":
            features_and_labels["x_0"] = torch.tensor(
                embeddings, dtype=torch.float32
            )
        elif self.node_feature_mode == "cluster_onehot":
            cluster_labels = df["cluster_label"].values
            features_and_labels["x_0"] = torch.zeros(
                len(df), self.n_clusters, dtype=torch.float32
            )
            features_and_labels["x_0"][
                torch.arange(len(df)),
                torch.tensor(cluster_labels, dtype=torch.long),
            ] = 1.0
        else:
            raise ValueError(
                f"Unknown node_feature_mode: {self.node_feature_mode}"
            )

        # Rank 1 (authors): average embedding of each author's messages
        author_features = []
        for msgs in self.author_hyperedges:
            msg_indices = [m.id for m in msgs]
            avg_emb = np.mean(embeddings[msg_indices], axis=0)
            author_features.append(avg_emb)
        features_and_labels["x_1"] = torch.tensor(
            np.array(author_features), dtype=torch.float32
        )

        # Rank 2 (threads): average embedding of each thread's messages
        thread_features = []
        for msgs in self.thread_hyperedges:
            msg_indices = [m.id for m in msgs]
            avg_emb = np.mean(embeddings[msg_indices], axis=0)
            thread_features.append(avg_emb)
        features_and_labels["x_2"] = torch.tensor(
            np.array(thread_features), dtype=torch.float32
        )

        # Rank 3 (semantic clusters): one-hot encoding
        num_semantic = connectivity["shape"][3]
        features_and_labels["x_3"] = torch.eye(
            num_semantic, dtype=torch.float32
        )

        # Labels: author prediction
        features_and_labels["y"] = torch.tensor(
            df["author_label"].values, dtype=torch.long
        )

        return features_and_labels

    def process(self) -> None:
        r"""Process the raw data and save it."""
        connectivity, embeddings, df = self.get_connectivity()
        features_and_labels = self.get_node_features_and_labels(
            embeddings, connectivity, df
        )
        data = Data(**connectivity, **features_and_labels)

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )


class Message:
    """Class representing a social media message node.

    Parameters
    ----------
    id : int
        Index in the dataset (contiguous 0..N-1).
    author_label : int
        Integer label of the author.
    cluster_label : int
        Integer label of the semantic cluster.
    """

    def __init__(self, id, author_label, cluster_label):
        self.id = id
        self.author_label = author_label
        self.cluster_label = cluster_label

    def __repr__(self):
        return f"Message(id={self.id})"

    def __lt__(self, other):
        return self.id < other.id
