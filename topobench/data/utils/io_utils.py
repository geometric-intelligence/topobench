"""Data IO utilities."""

import json
import os.path as osp
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex
from torch_geometric.data import Data
from torch_sparse import coalesce

from topobench.data.utils import get_complex_connectivity


def get_file_id_from_url(url):
    """Extract the file ID from a Google Drive file URL.

    Parameters
    ----------
    url : str
        The Google Drive file URL.

    Returns
    -------
    str
        The file ID extracted from the URL.

    Raises
    ------
    ValueError
        If the provided URL is not a valid Google Drive file URL.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "id" in query_params:  # Case 1: URL format contains '?id='
        file_id = query_params["id"][0]
    elif (
        "file/d/" in parsed_url.path
    ):  # Case 2: URL format contains '/file/d/'
        file_id = parsed_url.path.split("/")[3]
    else:
        raise ValueError(
            "The provided URL is not a valid Google Drive file URL."
        )
    return file_id


def download_file_from_drive(
    file_link, path_to_save, dataset_name, file_format="tar.gz"
):
    """Download a file from a Google Drive link and saves it to the specified path.

    Parameters
    ----------
    file_link : str
        The Google Drive link of the file to download.
    path_to_save : str
        The path where the downloaded file will be saved.
    dataset_name : str
        The name of the dataset.
    file_format : str, optional
        The format of the downloaded file. Defaults to "tar.gz".

    Raises
    ------
    None
    """
    file_id = get_file_id_from_url(file_link)

    download_link = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_link)

    output_path = f"{path_to_save}/{dataset_name}.{file_format}"
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")


def download_file_from_link(
    file_link, path_to_save, dataset_name, file_format="tar.gz"
):
    """Download a file from a link and saves it to the specified path.

    Parameters
    ----------
    file_link : str
        The link of the file to download.
    path_to_save : str
        The path where the downloaded file will be saved.
    dataset_name : str
        The name of the dataset.
    file_format : str, optional
        The format of the downloaded file. Defaults to "tar.gz".

    Raises
    ------
    None
    """
    response = requests.get(file_link)

    output_path = f"{path_to_save}/{dataset_name}.{file_format}"
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")


def read_ndim_manifolds(
    path, dim, y_val="betti_numbers", slice=None, load_as_graph=False
):
    """Load MANTRA dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.
    dim : int
        Dimension of the manifolds to load, required to make sanity checks.
    y_val : str, optional
        The triangulation information to use as label. Can be one of ['betti_numbers', 'torsion_coefficients',
        'name', 'genus', 'orientable'] (default: "orientable").
    slice : int, optional
        Slice of the dataset to load. If None, load the entire dataset (default: None). Used for testing.
    load_as_graph : bool
        Load mantra dataset as graph. Useful when arbitrary graph lifting need to be used.

    Returns
    -------
    torch_geometric.data.Data
        Data object of the manifold for the MANTRA dataset.
    """
    # Assert that y_val is one of the valid options
    # for each surface
    if dim == 2:
        assert y_val in [
            "betti_numbers",
            "torsion_coefficients",
            "name",
            "genus",
            "orientable",
        ]
    elif dim == 3:
        assert y_val in ["betti_numbers", "torsion_coefficients", "name"]
    else:
        raise ValueError("Invalid dimension. Only 2 and 3 are supported.")

    TORSION_COEF_NAMES = ["", "Z_2"]
    HOMEO_NAMES = [
        "",
        "Klein bottle",
        "RP^2",
        "S^2",
        "T^2",
        "S^2 twist S^1",
        "S^2 x S^1",
        "S^3",
    ]

    TORSION_COEF_NAME_TO_IDX = {
        name: i for i, name in enumerate(TORSION_COEF_NAMES)
    }
    HOMEO_NAME_TO_IDX = {name: i for i, name in enumerate(HOMEO_NAMES)}

    # Load file
    with open(path) as f:
        manifold_list = json.load(f)

    data_list = []
    # For each manifold
    for manifold in manifold_list[:slice]:
        n_vertices = manifold["n_vertices"]
        x = torch.ones(n_vertices, 1)
        y_value = manifold[y_val]

        if y_val == "betti_numbers":
            y = torch.tensor(y_value, dtype=torch.long).unsqueeze(dim=0)
        elif y_val == "genus":
            y = torch.tensor([y_value], dtype=torch.long).squeeze()
        elif y_val == "torsion_coefficients":
            y = torch.tensor(
                [TORSION_COEF_NAME_TO_IDX[coef] for coef in y_value],
                dtype=torch.long,
            ).unsqueeze(dim=0)
        elif y_val == "name":
            y = torch.tensor(
                [HOMEO_NAME_TO_IDX[y_value]], dtype=torch.long
            ).unsqueeze(0)
        elif y_val == "orientable":
            y = torch.tensor([y_value], dtype=torch.long).squeeze()
        else:
            raise ValueError(f"Invalid y_val: {y_val}")

        sc = SimplicialComplex()

        # Insert all simplices
        sc.add_simplices_from(manifold["triangulation"])

        # Build the simplex tensors for features, having only a one
        x_i = {
            f"x_{i}": torch.ones(len(sc.skeleton(i)), 1)
            for i in range(dim + 1)
        }

        if not load_as_graph:
            # Construct the connectivity matrices
            if dim == 2:
                inc_dict = get_complex_connectivity(sc, dim + 1, signed=False)
                assert inc_dict["incidence_3"].size(1) == 0, (
                    "For 2-dim manifolds there shouldn't be any tetrahedrons."
                )
            else:
                inc_dict = get_complex_connectivity(sc, dim, signed=False)

            data = Data(x=x, y=y, **x_i, **inc_dict)

        else:
            raise ValueError("Define if load_as_graph or not")

        data_list.append(data)
    return data_list


@dataclass
class GraphDatasetConfig:
    """Configuration for loading graph-based datasets with node features.

    Attributes
    ----------
    edge_file : str
        Filename for the edge list file.
    feature_file : str
        Filename template for the feature file (may contain format placeholders).
    edge_sep : str
        Separator for edge file (default: ",").
    feature_encoding : Optional[str]
        Encoding for feature file (default: None).
    keep_cols : list[str]
        List of column names to keep from feature file.
    node_id_col : str
        Column name for node identifiers (e.g., "FIPS").
    edges_use_node_ids : bool
        Indicates how node identifiers are represented in the edge file:
        - True: Edge file contains actual values from node_id_col (e.g., FIPS codes like 48201, 48453)
        - False: Edge file contains 0-indexed row numbers that reference positions in the feature file
    preprocessing_fn : Optional[Callable]
        Optional function to apply dataset-specific preprocessing to the stat DataFrame.
    """

    edge_file: str
    feature_file: str
    edge_sep: str = ","
    feature_encoding: str | None = None
    keep_cols: list[str] = None
    node_id_col: str = "FIPS"
    edges_use_node_ids: bool = True
    preprocessing_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def load_graph_with_features(
    path: str,
    config: GraphDatasetConfig,
    y_col: str,
    **format_kwargs,
) -> Data:
    """Unified function to load graph datasets with node features.

    This function handles the common pipeline for loading graph data:
    1. Load edge list and feature files
    2. Filter and clean data
    3. Handle node ID mapping
    4. Remove self-loops and isolated nodes
    5. Create undirected graph
    6. Return PyTorch Geometric Data object

    Parameters
    ----------
    path : str
        Path to the dataset directory.
    config : GraphDatasetConfig
        Configuration object specifying file formats and processing options.
    y_col : str
        Column name to use as the target variable.
    **format_kwargs : dict
        Additional keyword arguments for formatting file paths (e.g., year=2012).

    Returns
    -------
    torch_geometric.data.Data
        Data object containing the graph structure and features.
    """
    # Load edge list
    edge_file_path = f"{path}/{config.edge_file}"
    edges_df = pd.read_csv(
        edge_file_path,
        sep=config.edge_sep,
        header=None,
        names=["SRC", "DST"],
    )

    # Filter out rows that can't be converted to numeric (e.g., header lines, comments)
    edges_df = edges_df.apply(pd.to_numeric, errors="coerce")
    edges_df = edges_df.dropna()
    edges_df = edges_df.astype(int)

    # Load feature file
    feature_file_path = f"{path}/{config.feature_file.format(**format_kwargs)}"
    read_kwargs = {}
    if config.feature_encoding:
        read_kwargs["encoding"] = config.feature_encoding

    stat = pd.read_csv(feature_file_path, **read_kwargs)

    # Keep desired columns
    stat = stat.loc[:, config.keep_cols]

    # Apply dataset-specific preprocessing if provided
    if config.preprocessing_fn:
        stat = config.preprocessing_fn(stat)

    # Convert to numeric
    stat = stat.apply(pd.to_numeric, errors="coerce")

    # Fill NaN values with column means
    for column in stat.columns:
        if column != config.node_id_col:
            mean_value = stat[column].mean()
            stat[column] = stat[column].fillna(mean_value)

    # Drop any remaining NaN rows
    stat = stat.dropna()

    # Align node IDs with what edges reference
    if not config.edges_use_node_ids:
        # Edges reference row positions: overwrite node_id_col with row indices
        stat[config.node_id_col] = stat.index

    # Filter edges: keep only edges connecting nodes present in features
    node_ids_in_features = set(stat[config.node_id_col].unique())
    edges_df = edges_df[
        edges_df["SRC"].isin(node_ids_in_features)
        & edges_df["DST"].isin(node_ids_in_features)
    ]
    # and remove self-loops
    edges_df = edges_df[edges_df["SRC"] != edges_df["DST"]]

    # Filter features: keep only nodes in edges
    nodes_in_edges = set(edges_df["SRC"].unique()) | set(
        edges_df["DST"].unique()
    )
    stat = stat[stat[config.node_id_col].isin(nodes_in_edges)]
    stat = stat.reset_index(drop=True)

    # Map node IDs to be 0-indexed
    final_node_ids = stat[config.node_id_col].unique()
    id_to_idx = {node_id: idx for idx, node_id in enumerate(final_node_ids)}
    edges_df["SRC"] = edges_df["SRC"].map(id_to_idx)
    edges_df["DST"] = edges_df["DST"].map(id_to_idx)

    # Create edge_index and make undirected
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # Remove isolated nodes
    edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(
        edge_index
    )

    # Filter features to match final node set
    index = np.arange(mask.size(0))[mask]
    stat = stat.iloc[index].reset_index(drop=True)

    # Drop node ID column as it's no longer needed
    stat = stat.drop(columns=[config.node_id_col])

    # Separate features (x) from target (y)
    x_cols = [col for col in stat.columns if col != y_col]
    x = torch.tensor(stat[x_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(stat[y_col].to_numpy(), dtype=torch.float32)

    return Data(x=x, y=y, edge_index=edge_index)


def load_hypergraph_pickle_dataset(data_dir, data_name):
    """Load hypergraph datasets from pickle files.

    Parameters
    ----------
    data_dir : str
        Path to data.
    data_name : str
        Name of the dataset.

    Returns
    -------
    torch_geometric.data.Data
        Hypergraph dataset.
    """
    data_dir = osp.join(data_dir, data_name)

    # Load node features:

    with open(osp.join(data_dir, "features.pickle"), "rb") as f:
        features = pickle.load(f)
        features = features.todense()

    # Load node labels:
    with open(osp.join(data_dir, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # Load hypergraph.
    with open(osp.join(data_dir, "hypergraph.pickle"), "rb") as f:
        # Hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = 0  # num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph:
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    # check that every node is in some hyperedge
    if len(np.unique(node_list)) != num_nodes:
        # add self hyperedges to isolated nodes
        isolated_nodes = np.setdiff1d(
            np.arange(num_nodes), np.unique(node_list)
        )

        for node in isolated_nodes:
            node_list += [node]
            edge_list += [edge_idx]
            edge_idx += 1
            hypergraph[f"Unique_additonal_he_{edge_idx}"] = [node]

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        x_0=features,
        edge_index=edge_index,
        incidence_hyperedges=edge_index,
        y=labels,
    )

    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    num_class = len(np.unique(labels.numpy()))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = len(hypergraph)
    data.num_class = num_class

    data.incidence_hyperedges = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_hyperedges),
    )

    # Print some info
    print("Final num_hyperedges", data.num_hyperedges)
    print("Final num_nodes", data.num_nodes)
    print("Final num_class", data.num_class)

    return data, data_dir


def load_hypergraph_content_dataset(data_dir, data_name):
    """Load hypergraph datasets from pickle files.

    Parameters
    ----------
    data_dir : str
        Path to data.
    data_name : str
        Name of the dataset.

    Returns
    -------
    torch_geometric.data.Data
        Hypergraph dataset.
    """
    # data_dir = osp.join(data_dir, data_name)

    p2idx_features_labels = osp.join(data_dir, f"{data_name}.content")
    idx_features_labels = np.genfromtxt(
        p2idx_features_labels, dtype=np.dtype(str)
    )

    # features = np.array(idx_features_labels[:, 1:-1])
    features = torch.Tensor(idx_features_labels[:, 1:-1].astype(float)).float()
    labels = torch.Tensor(idx_features_labels[:, -1].astype(float)).long()

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    p2edges_unordered = p2idx_features_labels = osp.join(
        data_dir, f"{data_name}.edges"
    )
    edges_unordered = np.genfromtxt(p2edges_unordered, dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    # From adjacency matrix to edge_list
    edge_index = edges.T
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1

    features = features[:num_nodes]
    labels = labels[:num_nodes]

    # In case labels start from 1, we shift it to start from 0
    labels = labels - labels.min()

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index[1] = edge_index[1] - edge_index[1].min()

    node_list, edge_list = (
        list(edge_index[0].numpy()),
        list(edge_index[1].numpy()),
    )

    # # check that every node is in some hyperedge
    # if len(np.unique(node_list)) != num_nodes:
    #     # add self hyperedges to isolated nodes
    #     isolated_nodes = np.setdiff1d(
    #         np.arange(num_nodes), np.unique(node_list)
    #     )

    #     for node in isolated_nodes:
    #         node_list += [node]
    #         edge_list += [num_he]
    #         num_he += 1

    assert num_he == max(edge_list) + 1, (
        "Num hyperedges do not coincide after adding isolated nodes"
    )

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        x_0=features,
        edge_index=edge_index,
        incidence_hyperedges=edge_index,
        y=labels,
    )

    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_num_node_id_he_id, total_num_node_id_he_id
    )

    n_x = num_nodes
    num_class = len(np.unique(labels))

    # Add parameters to attribute
    data.n_x = n_x
    data.num_hyperedges = num_he
    data.num_class = num_class

    data.incidence_hyperedges = torch.sparse_coo_tensor(
        data.edge_index,
        values=torch.ones(data.edge_index.shape[1]),
        size=(data.num_nodes, data.num_hyperedges),
    )

    # Print some info
    print("Final num_hyperedges", data.num_hyperedges)
    print("Final num_nodes", data.num_nodes)
    print("Final num_class", data.num_class)

    return data, data_dir


# ----------------------
# BACKWARD COMPATIBILITY
# ----------------------
# The following code is written for backward compatibility and may be deleted.
# New datasets should use load_graph_with_features() directly with custom GraphDatasetConfig.
def _preprocess_us_county_demos(stat: pd.DataFrame) -> pd.DataFrame:
    """Apply US County Demographics specific preprocessing.

    Parameters
    ----------
    stat : pd.DataFrame
        Statistics DataFrame with demographic data.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with Election feature and cleaned columns.
    """
    # Replace comma with dot in MedianIncome (will be converted to numeric by pipeline)
    stat["MedianIncome"] = stat["MedianIncome"].replace(",", ".", regex=True)

    # Create Election variable from DEM and GOP
    stat["Election"] = (stat["DEM"] - stat["GOP"]) / (
        stat["DEM"] + stat["GOP"]
    )

    # Drop intermediate columns
    stat = stat.drop(columns=["DEM", "GOP"])

    return stat


US_COUNTY_DEMOS_CONFIG = GraphDatasetConfig(
    edge_file="county_graph.csv",
    feature_file="county_stats_{year}.csv",
    edge_sep=",",
    feature_encoding="ISO-8859-1",
    keep_cols=[
        "FIPS",
        "DEM",
        "GOP",
        "MedianIncome",
        "MigraRate",
        "BirthRate",
        "DeathRate",
        "BachelorRate",
        "UnemploymentRate",
    ],
    node_id_col="FIPS",
    edges_use_node_ids=True,
    preprocessing_fn=_preprocess_us_county_demos,
)


def read_us_county_demos(path, year=2012, y_col="Election"):
    """Load US County Demos dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.
    year : int, optional
        Year to load the features (default: 2012).
    y_col : str, optional
        Column to use as label. Can be one of ['Election', 'MedianIncome',
        'MigraRate', 'BirthRate', 'DeathRate', 'BachelorRate', 'UnemploymentRate'] (default: "Election").

    Returns
    -------
    torch_geometric.data.Data
        Data object of the graph for the US County Demos dataset.
    """
    return load_graph_with_features(
        path=path,
        config=US_COUNTY_DEMOS_CONFIG,
        y_col=y_col,
        year=year,
    )
