"""Dataset class for US County Demographics dataset."""

import os.path as osp
from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl
import toponetx as tnx
import torch
from hnne.finch_clustering import FINCH
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import get_colored_hypergraph_connectivity


class TwitterArlequinDataset(InMemoryDataset):
    r"""Dataset class for Twitter Arlequin dataset.

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

    URLS: ClassVar = {}

    FILE_FORMAT: ClassVar = {}

    RAW_FILE_NAMES: ClassVar = {
        "raw_data": "/data/gbg141/Arlequin/data/twitter_dataset.csv",
        "embedded_data": "/data/gbg141/Arlequin/data/twitter_embedded.parquet",
        "processed_data": "/data/gbg141/Arlequin/data/twitter_s10k.parquet"
    }

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        # hypergraph construction parameters
        self.cluster_level_posts = parameters.get("cluster_level_posts", 1)
        self.cluster_level_users = parameters.get("cluster_level_users", 0)
        self.max_rank = parameters.get("max_rank", 4)
        self.cluster_seed = parameters.get("cluster_seed", 42)
        self.neighborhoods = parameters.get("neighborhoods", None)
        self.semantic_cluster_algorithm = parameters.get(
            "semantic_cluster_algorithm",
            "spherical_kmeans",
        )
        self.semantic_kmeans_k = parameters.get("semantic_kmeans_k", 10)
        self.semantic_kmeans_n_init = parameters.get("semantic_kmeans_n_init", 1)
        self.hypergraph_id = (
            f"{self.cluster_level_posts}_{self.cluster_level_users}_"
            f"{self.max_rank}_{self.cluster_seed}_{self.neighborhoods}_"
            f"{self.semantic_cluster_algorithm}_"
            f"{self.semantic_kmeans_k}_{self.semantic_kmeans_n_init}"
        )
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
        return osp.join(self.root, self.name, self.hypergraph_id, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """

        return osp.join(self.root, self.name, self.hypergraph_id, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"
    
    def create_posts_nodes(self, df):
        """Create post nodes from dataframe and add to complex.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the twitter data.
        """
        self.posts = []
        for idx in range(len(df)):
            # get values
            id = df.at[idx,"id"]
            msg = df.at[idx,"message"]
            user_id = df.at[idx, "user_id"]
            replies = df.at[idx, "reply_to_user_id"].split("'")[1::2]
            replies = [int(r) for r in replies]
            mentions = df.at[idx, "mentions_user_id"].split("'")[1::2]
            mentions = [int(m) for m in mentions]

            # create post object and add to chg
            post = Post(id, msg, user_id, replies, mentions)
            self.posts.append(post)
            self.complex.add_node(post)

    def build_user_hyperedges(self, rank=1):
        """Build user hyperedges from posts and add to complex.
        
        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 1.
        """
        # find posts by each user
        self.users_to_posts = dict()
        for user in self.user_ids:
            self.users_to_posts[user] = []
        for post in self.posts:
            self.users_to_posts[post.user_id].append(post)

        # create users as hyperedges
        ids_to_users = dict()
        for user in self.user_ids:
            user = tnx.HyperEdge(self.users_to_posts[user], rank=rank)
            ids_to_users[user] = user
            self.complex.add_cell(user, rank=rank)

    def build_connection_hyperedges(self, rank=2):
        """Build user connection hyperedges from posts and add to complex.
        
        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 2.
        """
        self.replies = []
        self.mentions = []
        for post in self.posts:
            for r in post.reply_to_user_id:
                if r in self.user_ids and post.user_id != r:
                    combined_posts = self.users_to_posts[post.user_id] + self.users_to_posts[r]
                    reply = tnx.HyperEdge(combined_posts, rank=rank)
                    self.replies.append(combined_posts)
                    self.complex.add_cell(reply, rank=rank)
            for m in post.mentions_user_id:
                if m in self.user_ids and post.user_id != m:
                    combined_posts = self.users_to_posts[post.user_id] + self.users_to_posts[m]
                    mention = tnx.HyperEdge(combined_posts, rank=rank)
                    self.mentions.append(combined_posts)
                    self.complex.add_cell(mention, rank=rank)

    def cluster_posts(self, embeddings):
        """Cluster posts based on embeddings and add semantic hyperedges to complex.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Array containing the post embeddings.
        """
        if self.semantic_cluster_algorithm == "finch":
            clusters, n_clusters, _, _ = FINCH(
                data=embeddings,
                distance="cosine",
                verbose=0,
                random_state=self.cluster_seed,
            )
            print("Embeddings: ", n_clusters)
            self.semantics = clusters[:, self.cluster_level_posts]
            self.n_clusters_posts = n_clusters[self.cluster_level_posts]
            return

        if self.semantic_cluster_algorithm == "spherical_kmeans":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normalized_embeddings = embeddings / norms
            kmeans = KMeans(
                n_clusters=self.semantic_kmeans_k,
                n_init=self.semantic_kmeans_n_init,
                random_state=self.cluster_seed,
            )
            labels = kmeans.fit_predict(normalized_embeddings)
            self.semantics = labels
            self.n_clusters_posts = self.semantic_kmeans_k
            return

        raise ValueError(
            "Unsupported semantic_cluster_algorithm: "
            f"{self.semantic_cluster_algorithm}"
        )

    def build_semantic_hyperedges(self, rank=4):
        """Build semantic hyperedges from clustered posts and add to complex.
        
        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 4.
        """
        self.semantic_hyperedges = []
        for label in range(self.n_clusters_posts):
            # get users in cluster
            mask = self.semantics == label
            cluster_posts = np.array(self.posts)[mask]
            # create hyperedge
            cluster = tnx.HyperEdge(cluster_posts, rank=rank)
            self.semantic_hyperedges.append(cluster_posts)
            self.complex.add_cell(cluster, rank=rank)

    def compute_and_cluster_user_feature_vectors(self):
        """Compute feature vectors for each user based on clustered posts."""
        feature_vectors = []
        for user in self.user_ids:
            feature = [0.0 for _ in range(self.n_clusters_posts)]
            for post in self.users_to_posts[user]:
                index = self.posts_ids.index(post.id)
                cluster = self.semantics[index]
                feature[cluster] += 1
            feature_vectors.append([f / sum(feature) for f in feature])
        self.user_feature_vectors = np.array(feature_vectors)
        
        # cluster feature vectors
        clusters, n_clusters, _, _ = FINCH(data=self.user_feature_vectors, distance="euclidean", verbose=0, random_state=self.cluster_seed)
        print("Users: ", n_clusters)
        self.user_clusters = clusters[:, self.cluster_level_users]
        self.n_clusters_users = n_clusters[self.cluster_level_users]

    def build_community_hyperedges(self, rank=3):
        """Build community hyperedges from clustered users and add to complex.
        
        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 3.
        """
        self.community_hyperedges = []
        for label in range(self.n_clusters_users):
            # get users in cluster
            mask = self.user_clusters == label
            cluster_users = np.array(self.user_ids)[mask]
            # get posts of users
            cluster_posts = []
            for user in cluster_users:
                cluster_posts.extend(self.users_to_posts[user])
            # create hyperedge
            cluster = tnx.HyperEdge(cluster_posts, rank=rank)
            self.community_hyperedges.append(cluster_posts)
            self.complex.add_cell(cluster, rank=rank)

    def get_connectivity(self):
        """Get connectivity of the complex."""

        # open dataset
        df = pd.read_parquet(self.RAW_FILE_NAMES["processed_data"])
        df_embeddings = pl.read_parquet(self.RAW_FILE_NAMES["embedded_data"]).to_pandas()
        
        # store user and post ids
        self.user_ids = list(set(df["user_id"]))
        self.posts_ids = df["id"].to_list()

        # set dfs to same orientation
        df_embeddings.set_index("id", inplace=True)
        df_embeddings = df_embeddings.loc[self.posts_ids]
        embeddings = np.array(df_embeddings["embedding"].to_list())

        # create empty colored hypergraph complex
        self.complex = tnx.ColoredHyperGraph()

        # create posts as nodes
        self.create_posts_nodes(df)
        
        # build user hyperedges
        self.build_user_hyperedges(rank=1)

        # create user connections as hyperedges
        self.build_connection_hyperedges(rank=2)

        # cluster posts based on embeddings and create hyperedge for semantic clusters
        self.cluster_posts(embeddings)
        self.build_semantic_hyperedges(rank=4)

        # compute feature vectors for each user
        self.compute_and_cluster_user_feature_vectors()
        self.build_community_hyperedges(rank=3)

        connectivity = get_colored_hypergraph_connectivity(self.complex, max_rank=self.max_rank, neighborhoods=self.neighborhoods)
        return connectivity, embeddings
    
    def get_node_features_and_labels(self, embeddings, connectivity):
        """Get node features and labels for the dataset.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Array containing the post embeddings.
        connectivity : dict
            Dictionary containing the connectivity of the complex.

        Returns
        -------
        features_and_labels : dict
            Dictionary containing the features and labels for each node.
        """
        features_and_labels = dict()
        # Rank 0 nodes (posts)
        features_and_labels["x_0"] = torch.tensor(embeddings, dtype=torch.float32)
        # Rank 1 (users)
        features_and_labels["x_1"] = torch.tensor(self.user_feature_vectors, dtype=torch.float32)
        # Rank 2 (connections)
        features_and_labels["x_2"] = torch.zeros(connectivity["shape"][2], features_and_labels["x_0"].shape[1], dtype=torch.float32)
        # Rank 3 (communities)
        features_and_labels["x_3"] = torch.zeros(connectivity["shape"][3], features_and_labels["x_0"].shape[1], dtype=torch.float32)
        # Rank 4 (semantic clusters) - convert to one-hot encoding
        num_classes = connectivity["shape"][4]
        features_and_labels["x_4"] = torch.eye(num_classes, dtype=torch.float32)
        # Labels (semantic clusters of posts)
        features_and_labels["y"] = torch.tensor(self.semantics, dtype=torch.long)
        return features_and_labels

    def process(self) -> None:
        r"""Handle the data for the dataset.
        """
        connectivity, embeddings = self.get_connectivity()
        features_and_labels = self.get_node_features_and_labels(embeddings, connectivity)
        data = Data(**connectivity, **features_and_labels)

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )


# node class
class Post:
    """Class representing a Twitter post node.
    
    Parameters
    ----------
    id : int
        Unique identifier for the post.
    msg : str
        Content of the post.
    user_id : int
        Identifier of the user who created the post.
    reply_to_user_id : list
        List of user IDs that the post replies to.
    mentions_user_id : list
        List of user IDs that the post mentions.
    """
    def __init__(self, id, msg, user_id, reply_to_user_id, mentions_user_id):
        self.id = id
        self.msg = msg
        self.user_id = user_id
        self.reply_to_user_id = reply_to_user_id
        self.mentions_user_id = mentions_user_id
        
    def __repr__(self):
        return f"Post(id={self.id}, user_id={self.user_id})"

    def __lt__(self, other):
        return self.msg < other.msg

