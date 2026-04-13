"""Dataset class for HAL Arlequin dataset (scientific publications)."""

import hashlib
import os.path as osp
from typing import ClassVar

import numpy as np
import pandas as pd
import toponetx as tnx
import torch
from hnne.finch_clustering import FINCH
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from topobench.data.utils import get_colored_hypergraph_connectivity


class HALArlequinDataset(InMemoryDataset):
    r"""Dataset class for HAL Arlequin dataset.

    Models scientific publications from the HAL/ALMANACH network as a colored
    hypergraph with author, institution, and semantic cluster hyperedges.

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
    URLS : dict
        Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT : dict
        Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES : dict
        Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {}

    FILE_FORMAT: ClassVar = {}

    RAW_FILE_NAMES: ClassVar = {
        "enriched_data": "/data/gbg141/Arlequin/data/almanach_documents_enriched.parquet",
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
        self.semantic_cluster_algorithm = parameters.get(
            "semantic_cluster_algorithm",
            "spherical_kmeans",
        )
        self.semantic_kmeans_k = parameters.get("semantic_kmeans_k", 10)
        self.semantic_kmeans_n_init = parameters.get("semantic_kmeans_n_init", 1)
        self.ho_init_method = parameters.get("ho_init_method", "avg_doc")
        self.min_papers_per_author = parameters.get("min_papers_per_author", 2)
        self.institution_level = parameters.get("institution_level", "laboratory")
        self.prediction_task = parameters.get("prediction_task", "semantic")
        self.min_papers_per_first_author = parameters.get(
            "min_papers_per_first_author", None
        )
        self.use_keyword_hyperedges = parameters.get(
            "use_keyword_hyperedges", False
        )
        self.min_papers_per_keyword = parameters.get("min_papers_per_keyword", 2)

        if self.neighborhoods is not None:
            neighborhoods_str = ",".join(sorted(self.neighborhoods))
            neighborhoods_hash = hashlib.md5(
                neighborhoods_str.encode()
            ).hexdigest()[:12]
        else:
            neighborhoods_hash = "none"
        self.hypergraph_id = (
            f"{self.max_rank}_{self.cluster_seed}_{neighborhoods_hash}_"
            f"{self.ho_init_method}_{self.semantic_cluster_algorithm}_"
            f"{self.semantic_kmeans_k}_{self.semantic_kmeans_n_init}_"
            f"{self.min_papers_per_author}_{self.institution_level}_"
            f"{self.prediction_task}_{self.min_papers_per_first_author}_"
            f"kw{self.use_keyword_hyperedges}_{self.min_papers_per_keyword}"
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
        return (
            f"{self.name}(root={self.root}, name={self.name}, "
            f"parameters={self.parameters})"
        )

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

    def get_data_dir(self) -> str:
        """Return the path to the data directory.

        Returns
        -------
        str
            Path to the data directory.
        """
        return osp.join(self.root, self.name, self.hypergraph_id)

    def create_document_nodes(self, df):
        """Create document nodes from dataframe and add to complex.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the enriched HAL publications data.
        """
        self.documents = []
        self.id_documents = dict()
        for idx in range(len(df)):
            doc = Document(idx, df.at[idx, "title"])
            self.documents.append(doc)
            self.complex.add_node(doc)
            self.id_documents[idx] = doc

    def build_author_hyperedges(self, df, rank=1):
        """Build author hyperedges connecting each author's papers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the enriched data with 'authors' column.
        rank : int, optional
            Rank of the hyperedges, by default 1.
        """
        author_to_docs = dict()
        for idx in range(len(df)):
            for author in df.at[idx, "authors"]:
                if author not in author_to_docs:
                    author_to_docs[author] = []
                author_to_docs[author].append(self.documents[idx])

        self.author_names = []
        self.author_hyperedges = []
        for author, docs in author_to_docs.items():
            if len(docs) < self.min_papers_per_author:
                continue
            he = tnx.HyperEdge(docs, rank=rank)
            self.author_names.append(author)
            self.author_hyperedges.append(docs)
            self.complex.add_cell(he, rank=rank)

        print(
            f"Author hyperedges (rank {rank}): {len(self.author_hyperedges)} "
            f"(from {len(author_to_docs)} unique authors, "
            f"filtered with min_papers={self.min_papers_per_author})"
        )

    def build_institution_hyperedges(self, df, rank=2):
        """Build institution hyperedges connecting papers from the same institution.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with 'struct_names' and 'struct_types' columns.
        rank : int, optional
            Rank of the hyperedges, by default 2.
        """
        inst_to_docs = dict()
        for idx in range(len(df)):
            names = df.at[idx, "struct_names"]
            types = df.at[idx, "struct_types"]
            seen = set()
            for sname, stype in zip(names, types):
                if stype == self.institution_level and sname not in seen:
                    seen.add(sname)
                    if sname not in inst_to_docs:
                        inst_to_docs[sname] = []
                    inst_to_docs[sname].append(self.documents[idx])

        self.institution_names = []
        self.institution_hyperedges = []
        for inst, docs in inst_to_docs.items():
            if len(docs) < 2:
                continue
            he = tnx.HyperEdge(docs, rank=rank)
            self.institution_names.append(inst)
            self.institution_hyperedges.append(docs)
            self.complex.add_cell(he, rank=rank)

        print(
            f"Institution hyperedges (rank {rank}): "
            f"{len(self.institution_hyperedges)} "
            f"(level={self.institution_level}, "
            f"from {len(inst_to_docs)} unique institutions)"
        )

    def cluster_documents(self, embeddings):
        """Cluster document embeddings for semantic hyperedges.

        Parameters
        ----------
        embeddings : np.ndarray
            Array of shape (n_docs, embedding_dim) with document embeddings.
        """
        if self.semantic_cluster_algorithm == "finch":
            clusters, n_clusters, _, _ = FINCH(
                data=embeddings,
                distance="cosine",
                verbose=0,
                random_state=self.cluster_seed,
            )
            level = min(0, clusters.shape[1] - 1)
            self.semantic_labels = clusters[:, level]
            self.n_semantic_clusters = n_clusters[level]
            print(
                f"FINCH: {clusters.shape[1]} levels, "
                f"using level {level} with {self.n_semantic_clusters} clusters"
            )
            return

        if self.semantic_cluster_algorithm == "spherical_kmeans":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normalized = embeddings / norms
            kmeans = KMeans(
                n_clusters=self.semantic_kmeans_k,
                n_init=self.semantic_kmeans_n_init,
                random_state=self.cluster_seed,
            )
            self.semantic_labels = kmeans.fit_predict(normalized)
            self.n_semantic_clusters = self.semantic_kmeans_k
            return

        raise ValueError(
            "Unsupported semantic_cluster_algorithm: "
            f"{self.semantic_cluster_algorithm}"
        )

    def build_semantic_hyperedges(self, rank=3):
        """Build semantic hyperedges from clustered documents.

        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 3.
        """
        self.semantic_hyperedges = []
        for label in range(self.n_semantic_clusters):
            mask = self.semantic_labels == label
            cluster_docs = np.array(self.documents)[mask]
            he = tnx.HyperEdge(cluster_docs, rank=rank)
            self.semantic_hyperedges.append(cluster_docs)
            self.complex.add_cell(he, rank=rank)

        print(
            f"Semantic hyperedges (rank {rank}): "
            f"{len(self.semantic_hyperedges)} clusters"
        )

    def build_keyword_hyperedges(self, df, rank=4):
        """Build keyword hyperedges connecting papers that share a keyword.

        Uses the ``keywords`` column of the dataframe (a list of author-supplied
        keyword strings per paper, as fetched from the HAL API). Keywords with
        fewer than ``min_papers_per_keyword`` papers are discarded to keep the
        hypergraph dense enough to carry useful signal.

        Keywords are stripped of surrounding whitespace before grouping; papers
        with an empty keyword list simply do not contribute to any hyperedge.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with a 'keywords' column (each entry is a list of str).
        rank : int, optional
            Rank of the keyword hyperedges, by default 4.
        """
        keyword_to_docs: dict[str, set] = {}
        for idx in range(len(df)):
            for kw in df.at[idx, "keywords"]:
                kw = kw.strip()
                if not kw:
                    continue
                if kw not in keyword_to_docs:
                    keyword_to_docs[kw] = set()
                keyword_to_docs[kw].add(self.documents[idx])

        self.keyword_names = []
        self.keyword_hyperedges = []
        for kw, docs_set in keyword_to_docs.items():
            docs = sorted(docs_set)  # sorted for determinism
            if len(docs) < self.min_papers_per_keyword:
                continue
            he = tnx.HyperEdge(docs, rank=rank)
            self.keyword_names.append(kw)
            self.keyword_hyperedges.append(docs)
            self.complex.add_cell(he, rank=rank)

        print(
            f"Keyword hyperedges (rank {rank}): {len(self.keyword_hyperedges)} "
            f"(from {len(keyword_to_docs)} unique keywords, "
            f"filtered with min_papers={self.min_papers_per_keyword})"
        )

    def _extract_all_authors_labels(self, df):
        """Build multilabel binary targets for all-author prediction.

        Each document gets a binary vector of length equal to the number of
        vocabulary authors (authors with >= min_papers_per_author papers).
        Entry i is 1.0 if the i-th vocabulary author co-authored the document.

        Uses the same ``min_papers_per_author`` filter as
        :meth:`build_author_hyperedges` so the label vocabulary is consistent
        with the hypergraph structure.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with an 'authors' column (each entry is a list/array).

        Returns
        -------
        df : pd.DataFrame
            Unchanged dataframe (returned for interface consistency).
        """
        author_counts: dict[str, int] = {}
        for authors in df["authors"]:
            for a in authors:
                author_counts[a] = author_counts.get(a, 0) + 1

        vocab = sorted(
            a for a, cnt in author_counts.items()
            if cnt >= self.min_papers_per_author
        )
        author_to_idx = {a: i for i, a in enumerate(vocab)}
        self.n_all_authors = len(vocab)

        n_docs = len(df)
        labels = np.zeros((n_docs, self.n_all_authors), dtype=np.float32)
        for doc_idx, authors in enumerate(df["authors"]):
            for a in authors:
                if a in author_to_idx:
                    labels[doc_idx, author_to_idx[a]] = 1.0

        self.all_author_labels = labels
        print(
            f"All-author multilabel prediction: {self.n_all_authors} vocabulary "
            f"authors (>= {self.min_papers_per_author} papers), "
            f"{n_docs} documents"
        )
        return df

    def _extract_first_author_labels(self, df):
        """Extract first-author labels and optionally filter the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with an 'authors' column (each entry is a list/array).

        Returns
        -------
        df : pd.DataFrame
            Possibly filtered and reindexed dataframe.
        """
        df = df.copy()
        df["first_author"] = df["authors"].apply(lambda a: a[0])

        if self.min_papers_per_first_author is not None:
            fa_counts = df["first_author"].value_counts()
            valid_fa = fa_counts[
                fa_counts >= self.min_papers_per_first_author
            ].index
            before = len(df)
            df = df[df["first_author"].isin(valid_fa)].reset_index(drop=True)
            print(
                f"First-author filter (>= {self.min_papers_per_first_author}): "
                f"{before} -> {len(df)} papers, "
                f"{len(valid_fa)} first authors"
            )

        unique_fa = sorted(df["first_author"].unique())
        fa_map = {name: i for i, name in enumerate(unique_fa)}
        self.first_author_labels = df["first_author"].map(fa_map).values
        self.n_first_authors = len(unique_fa)
        print(
            f"First-author prediction: {self.n_first_authors} classes, "
            f"{len(df)} papers"
        )
        return df

    def get_connectivity(self):
        """Build the colored hypergraph and return its connectivity.

        Returns
        -------
        connectivity : dict
            Connectivity matrices and metadata for the colored hypergraph.
        embeddings : np.ndarray
            Document embeddings array.
        """
        df = pd.read_parquet(self.RAW_FILE_NAMES["enriched_data"])

        if self.prediction_task == "first_author":
            df = self._extract_first_author_labels(df)
        elif self.prediction_task == "all_authors":
            df = self._extract_all_authors_labels(df)

        self.doc_ids = list(range(len(df)))
        embeddings = np.array(df["embedding"].to_list())

        self.complex = tnx.ColoredHyperGraph()

        self.create_document_nodes(df)
        self.build_author_hyperedges(df, rank=1)
        self.build_institution_hyperedges(df, rank=2)

        self.cluster_documents(embeddings)
        self.build_semantic_hyperedges(rank=3)

        if self.use_keyword_hyperedges:
            self.build_keyword_hyperedges(df, rank=4)

        connectivity = get_colored_hypergraph_connectivity(
            self.complex,
            max_rank=self.max_rank,
            neighborhoods=self.neighborhoods,
        )
        return connectivity, embeddings

    def get_node_features_and_labels(self, embeddings, connectivity):
        """Get node features and labels for the dataset.

        Parameters
        ----------
        embeddings : np.ndarray
            Array containing the document embeddings.
        connectivity : dict
            Dictionary containing the connectivity of the complex.

        Returns
        -------
        features_and_labels : dict
            Dictionary containing the features and labels for each rank.
        """
        features_and_labels = dict()

        features_and_labels["x_0"] = torch.tensor(
            embeddings, dtype=torch.float32
        )

        if self.ho_init_method == "avg_doc":
            # Rank 1 (authors): average embedding of each author's papers
            author_features = []
            for docs in self.author_hyperedges:
                doc_indices = [d.id for d in docs]
                avg_emb = np.mean(embeddings[doc_indices], axis=0)
                author_features.append(avg_emb)
            features_and_labels["x_1"] = torch.tensor(
                np.array(author_features), dtype=torch.float32
            )

            # Rank 2 (institutions): average embedding of each institution's papers
            inst_features = []
            for docs in self.institution_hyperedges:
                doc_indices = [d.id for d in docs]
                avg_emb = np.mean(embeddings[doc_indices], axis=0)
                inst_features.append(avg_emb)
            features_and_labels["x_2"] = torch.tensor(
                np.array(inst_features), dtype=torch.float32
            )

            # Rank 3 (semantic clusters): average embedding per cluster
            cluster_features = []
            for label in range(self.n_semantic_clusters):
                mask = self.semantic_labels == label
                avg_emb = np.mean(embeddings[mask], axis=0)
                cluster_features.append(avg_emb)
            features_and_labels["x_3"] = torch.tensor(
                np.array(cluster_features), dtype=torch.float32
            )

            if self.use_keyword_hyperedges:
                # Rank 4 (keywords): average embedding of each keyword's papers
                kw_features = []
                for docs in self.keyword_hyperedges:
                    doc_indices = [d.id for d in docs]
                    avg_emb = np.mean(embeddings[doc_indices], axis=0)
                    kw_features.append(avg_emb)
                features_and_labels["x_4"] = torch.tensor(
                    np.array(kw_features), dtype=torch.float32
                )
        else:
            for rank in range(1, self.max_rank + 1):
                num_cells = connectivity["shape"][rank]
                features_and_labels[f"x_{rank}"] = torch.eye(
                    num_cells, dtype=torch.float32
                )

        if self.prediction_task == "first_author":
            features_and_labels["y"] = torch.tensor(
                self.first_author_labels, dtype=torch.long
            )
        elif self.prediction_task == "all_authors":
            features_and_labels["y"] = torch.tensor(
                self.all_author_labels, dtype=torch.float32
            )
        else:
            features_and_labels["y"] = torch.tensor(
                self.semantic_labels, dtype=torch.long
            )
        return features_and_labels

    def process(self) -> None:
        r"""Process the raw data and save it."""
        connectivity, embeddings = self.get_connectivity()
        features_and_labels = self.get_node_features_and_labels(
            embeddings, connectivity
        )
        data = Data(**connectivity, **features_and_labels)

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )


class Document:
    """Class representing a scientific document node.

    Parameters
    ----------
    id : int
        Unique identifier for the document.
    title : str
        Title of the document.
    """

    def __init__(self, id, title):
        self.id = id
        self.title = title

    def __repr__(self):
        return f"Document(id={self.id})"

    def __lt__(self, other):
        return self.id < other.id
