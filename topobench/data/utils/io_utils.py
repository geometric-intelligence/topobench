"""Data IO utilities."""

import glob
import json
import os
import os.path as osp
import pickle
import time
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
    file_link,
    path_to_save,
    dataset_name,
    file_format="tar.gz",
    verify=True,
    timeout=None,
    retries=3,
):
    """Download a file from a link and saves it to the specified path.

    Uses streaming with chunked download and includes retry logic for
    resilience against network interruptions.

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
    verify : bool, optional
        Whether to verify SSL certificates. Defaults to True.
    timeout : float, optional
        Timeout in seconds per chunk read (not for entire download). For very slow
        servers, increase this value. Default: 60 seconds per chunk.
    retries : int, optional
        Number of retry attempts if download fails. Defaults to 3.

    Notes
    -----
    This function downloads files in 5MB chunks for memory efficiency. Progress is
    reported every 10MB. Timeouts apply per chunk, not to the entire download,
    making it suitable for very large files and slow connections.

    If a download fails, it retries with exponential backoff (5s, 10s, 15s).

    Examples
    --------
    Basic download:

    >>> from topobench.data.utils import download_file_from_link
    >>> download_file_from_link(
    ...     file_link="https://example.com/dataset.tar.gz",
    ...     path_to_save="./data/",
    ...     dataset_name="my_dataset"
    ... )

    Download with custom timeout for slow servers:

    >>> download_file_from_link(
    ...     file_link="https://slow-server.com/dataset.zip",
    ...     path_to_save="./data/",
    ...     dataset_name="my_dataset",
    ...     file_format="zip",
    ...     timeout=300  # 5 minutes per chunk
    ... )

    Download with increased retries for unreliable connections:

    >>> download_file_from_link(
    ...     file_link="https://example.com/dataset.tar.gz",
    ...     path_to_save="./data/",
    ...     dataset_name="my_dataset",
    ...     retries=5  # Try up to 5 times
    ... )

    Raises
    ------
    Exception
        If download fails after all retry attempts.
    """
    # Ensure output directory exists
    os.makedirs(path_to_save, exist_ok=True)
    output_path = f"{path_to_save}/{dataset_name}.{file_format}"

    # Default timeout: 60 seconds per chunk read (for very slow servers)
    if timeout is None:
        timeout = 60

    for attempt in range(retries):
        try:
            print(
                f"[Download] Starting download from: {file_link} (attempt {attempt + 1}/{retries})"
            )

            # Use tuple (connect_timeout, read_timeout) for proper streaming
            response = requests.get(
                file_link,
                verify=verify,
                stream=True,  # Force streaming for chunked download
                timeout=(
                    30,
                    timeout,
                ),  # (connect timeout, read timeout per chunk)
            )

            if response.status_code != 200:
                print(
                    f"[Download] Failed to download the file. HTTP {response.status_code}"
                )
                return

            # Streaming download with progress reporting
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            start_time = time.time()

            if total_size > 0:
                print(
                    f"[Download] Total file size: {total_size / (1024**3):.2f} GB"
                )
            else:
                print("[Download] Total file size: unknown")

            # Stream download in chunks
            chunk_size = 5 * 1024 * 1024  # 5MB chunks for faster throughput
            progress_interval = (
                10 * 1024 * 1024
            )  # Report progress every 10MB (for slow connections)
            last_reported = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=chunk_size, decode_unicode=False
                ):
                    if chunk:
                        f.write(chunk)
                        f.flush()  # Ensure data is written to disk
                        downloaded += len(chunk)

                        # Print progress every 10MB
                        if (
                            total_size > 0
                            and (downloaded - last_reported)
                            >= progress_interval
                        ):
                            percent = (downloaded / total_size) * 100
                            remaining = total_size - downloaded
                            elapsed_time = time.time() - start_time
                            speed_mbps = (downloaded / (1024**2)) / (
                                elapsed_time + 0.001
                            )

                            # Calculate ETA
                            if speed_mbps > 0:
                                eta_seconds = (
                                    remaining / (1024**2) / speed_mbps
                                )
                                eta_hours = eta_seconds / 3600
                                eta_minutes = (eta_seconds % 3600) / 60
                                eta_str = (
                                    f"{eta_hours:.0f}h {eta_minutes:.0f}m"
                                )
                            else:
                                eta_str = "calculating..."

                            print(
                                f"[Download] {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB ({percent:.1f}%) | Speed: {speed_mbps:.2f} MB/s | ETA: {eta_str}"
                            )
                            last_reported = downloaded

            print(f"[Download] Download complete! Saved to: {output_path}")
            break

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            Exception,
        ) as e:
            print(
                f"[Download] Download failed with error: {type(e).__name__}: {str(e)}"
            )
            if attempt < retries - 1:
                wait_time = 5 * (
                    attempt + 1
                )  # Exponential backoff: 5s, 10s, 15s
                print(
                    f"[Download] Retrying in {wait_time} seconds... (attempt {attempt + 2}/{retries})"
                )
                time.sleep(wait_time)
            else:
                print(
                    f"[Download] Failed after {retries} attempts. Please check your connection and try again."
                )
                raise e


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
    edges_df = pd.read_csv(f"{path}/county_graph.csv")
    stat = pd.read_csv(
        f"{path}/county_stats_{year}.csv", encoding="ISO-8859-1"
    )

    keep_cols = [
        "FIPS",
        "DEM",
        "GOP",
        "MedianIncome",
        "MigraRate",
        "BirthRate",
        "DeathRate",
        "BachelorRate",
        "UnemploymentRate",
    ]

    # Select columns, replace ',' with '.' and convert to numeric
    stat = stat.loc[:, keep_cols]
    stat["MedianIncome"] = stat["MedianIncome"].replace(",", ".", regex=True)
    stat = stat.apply(pd.to_numeric, errors="coerce")

    # Step 2: Substitute NaN values with column mean
    for column in stat.columns:
        if column != "FIPS":
            mean_value = stat[column].mean()
            stat[column] = stat[column].fillna(mean_value)
    stat = stat[keep_cols].dropna()

    # Delete edges that are not present in stat df
    unique_fips = stat["FIPS"].unique()

    src_ = edges_df["SRC"].apply(lambda x: x in unique_fips)
    dst_ = edges_df["DST"].apply(lambda x: x in unique_fips)

    edges_df = edges_df[src_ & dst_]

    # Remove rows from stat df where edges_df['SRC'] or edges_df['DST'] are not present
    stat = stat[
        stat["FIPS"].isin(edges_df["SRC"]) & stat["FIPS"].isin(edges_df["DST"])
    ]
    stat = stat.reset_index(drop=True)

    # Remove rows where SRC == DST
    edges_df = edges_df[edges_df["SRC"] != edges_df["DST"]]

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Make edge_index undirected
    edge_index = torch_geometric.utils.to_undirected(edge_index)

    # Convert edge_index back to pandas DataFrame
    edges_df = pd.DataFrame(edge_index.numpy().T, columns=["SRC", "DST"])

    del edge_index

    # Map stat['FIPS'].unique() to [0, ..., num_nodes]
    fips_map = {fips: i for i, fips in enumerate(stat["FIPS"].unique())}
    stat["FIPS"] = stat["FIPS"].map(fips_map)

    # Map edges_df['SRC'] and edges_df['DST'] to [0, ..., num_nodes]
    edges_df["SRC"] = edges_df["SRC"].map(fips_map)
    edges_df["DST"] = edges_df["DST"].map(fips_map)

    # Get torch_geometric edge_index format
    edge_index = torch.tensor(
        np.stack([edges_df["SRC"].to_numpy(), edges_df["DST"].to_numpy()])
    )

    # Remove isolated nodes (Note: this function maps the nodes to [0, ..., num_nodes] automatically)
    edge_index, _, mask = torch_geometric.utils.remove_isolated_nodes(
        edge_index
    )

    # Convert mask to index
    index = np.arange(mask.size(0))[mask]
    stat = stat.iloc[index]
    stat = stat.reset_index(drop=True)

    # Get new values for FIPS from current index
    # To understand why please print stat.iloc[[516, 517, 518, 519, 520]] for 2012 year
    # Basically the FIPS values have been shifted
    stat["FIPS"] = stat.reset_index()["index"]

    # Create Election variable
    stat["Election"] = (stat["DEM"] - stat["GOP"]) / (
        stat["DEM"] + stat["GOP"]
    )

    # Drop DEM and GOP columns and FIPS
    stat = stat.drop(columns=["DEM", "GOP", "FIPS"])

    # Prediction col
    x_col = list(stat.columns)
    x_col.remove(y_col)

    x = torch.tensor(stat[x_col].to_numpy(), dtype=torch.float32)
    y = torch.tensor(stat[y_col].to_numpy(), dtype=torch.float32)

    data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)

    return data


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


def collect_mat_files(data_dir: str) -> list:
    """Collect all .mat files from a directory recursively.

    Excludes files containing "diffxy" in their names.

    Parameters
    ----------
    data_dir : str
        Root directory to search for .mat files.

    Returns
    -------
    list
        Sorted list of .mat file paths.
    """
    patterns = [os.path.join(data_dir, "**", "*.mat")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if "diffxy" not in f]
    files.sort()
    return files


def mat_cell_to_dict(mt) -> dict:
    """Convert MATLAB cell array to dictionary.

    Parameters
    ----------
    mt : np.ndarray
        MATLAB cell array (structured array).

    Returns
    -------
    dict
        Dictionary with keys from cell array field names and squeezed values.
    """
    clean_data = {}
    keys = mt.dtype.names
    for key_idx, key in enumerate(keys):
        clean_data[key] = (
            np.squeeze(mt[key_idx])
            if isinstance(mt[key_idx], np.ndarray)
            else mt[key_idx]
        )
    return clean_data


def planewise_mat_cell_to_dict(mt) -> dict:
    """Convert plane-wise MATLAB cell array to nested dictionary.

    Parameters
    ----------
    mt : np.ndarray
        MATLAB cell array with plane dimension.

    Returns
    -------
    dict
        Nested dictionary with plane IDs as keys.
    """
    clean_data = {}
    for plane_id in range(len(mt[0])):
        keys = mt[0, plane_id].dtype.names
        clean_data[plane_id] = {}
        for key_idx, key in enumerate(keys):
            clean_data[plane_id][key] = (
                np.squeeze(mt[0, plane_id][key_idx])
                if isinstance(mt[0, plane_id][key_idx], np.ndarray)
                else mt[0, plane_id][key_idx]
            )
    return clean_data


def process_mat(mat_data) -> dict:
    """Generate MATLAB data structure into organized dictionary.

    Converts MATLAB cell arrays for BFInfo, CellInfo, CorrInfo, and other
    experimental metadata into nested Python dictionaries.

    Parameters
    ----------
    mat_data : dict
        Dictionary loaded from MATLAB .mat file via scipy.io.loadmat.

    Returns
    -------
    dict
        Processed data structure with organized BFInfo, CellInfo, CorrInfo,
        coordinate arrays, and experimental variables.
    """
    mt = {}
    mt["BFInfo"] = planewise_mat_cell_to_dict(mat_data["BFinfo"])
    mt["CellInfo"] = planewise_mat_cell_to_dict(mat_data["CellInfo"])
    mt["CorrInfo"] = planewise_mat_cell_to_dict(mat_data["CorrInfo"])
    mt["allZCorrInfo"] = mat_cell_to_dict(mat_data["allZCorrInfo"][0, 0])

    for cord_key in ["allxc", "allyc", "allzc", "zDFF"]:
        mt[cord_key] = {}
        for p in range(mat_data[cord_key].shape[0]):
            mt[cord_key][p] = mat_data[cord_key][p, 0]

    mt["exptVars"] = mat_cell_to_dict(mat_data["exptVars"][0, 0])
    mt["selectZCorrInfo"] = mat_cell_to_dict(mat_data["selectZCorrInfo"][0, 0])
    mt["stimInfo"] = planewise_mat_cell_to_dict(mat_data["stimInfo"])
    mt["zStuff"] = planewise_mat_cell_to_dict(mat_data["zStuff"])

    return mt
