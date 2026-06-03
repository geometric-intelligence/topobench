"""Cornell hypergraph datasets with labeled nodes.

This module provides the CornellLabeledNodesDataset class for loading real-world
hypergraph datasets from Cornell's collection. These datasets are designed for
single-label node classification tasks in higher-order networks.

Key features:
    - Automatic dataset downloading from Google Drive
    - Parsing of Cornell's text format (hyperedges, node labels, label names)
    - Automatic 0-indexing and node deduplication
    - Construction of sparse incidence matrices for hypergraph neural networks

File format (for each dataset):
    - hyperedges-{name}.txt: One hyperedge per line (comma-separated node IDs)
    - node-labels-{name}.txt: One label per line (line i = label for node i)
    - label-names-{name}.txt: Class names (one per line)

References
----------
Cornell Hypergraph Data Repository: https://www.cs.cornell.edu/~arb/data/

Examples
--------
>>> from topobench.data.datasets import CornellLabeledNodesDataset
>>> dataset = CornellLabeledNodesDataset(data_dir='./data', data_name='walmart-trips')
>>> data = dataset[0]
>>> print(f"Dataset: {dataset.name}, Nodes: {data.num_nodes}, Classes: {data.num_class}")
"""

import os

import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_google_url,
    extract_zip,
)

from topobench.data.utils.io_utils import parse_cornell_hypergraph_files


class CornellLabeledNodesDataset(InMemoryDataset):
    """Cornell hypergraph datasets with single-label node classification.

    Unified PyTorch Geometric dataset class for loading Cornell's collection of
    real-world hypergraph datasets. Each dataset represents higher-order relationships
    (hyperedges connecting multiple nodes) with nodes labeled into discrete classes
    for supervised node classification tasks.

    The class automatically handles:
        - Downloading datasets from Google Drive
        - Parsing Cornell text format (hyperedges, node labels, label names)
        - Automatic 0-indexing of labels and node IDs
        - Node deduplication within hyperedges (ensures binary incidence matrices)
        - Construction of sparse incidence matrices for hypergraph neural networks

    Available datasets include e-commerce (walmart-trips, amazon-reviews), face-to-face contacts,
    and US Congress co-sponsorship/committee data.
    See DATASETS dict for complete list of supported datasets.

    Parameters
    ----------
    data_dir : str
        Root directory where the dataset should be stored. Dataset files will be
        saved in a subdirectory named after data_name.
    data_name : str
        Name of the dataset to load. Must be one of the keys in DATASETS dict.
        Examples: 'walmart-trips', 'house-committees', 'amazon-reviews'.
    transform : callable, optional
        A function/transform that takes in a Data object and returns a
        transformed version. Applied on-the-fly when accessing data.
    pre_transform : callable, optional
        A function/transform that takes in a Data object and returns a
        transformed version. Applied once during processing before saving to disk.

    Raises
    ------
    ValueError
        If data_name is not in the DATASETS dict.

    Examples
    --------
    Load and access walmart-trips dataset:
        >>> dataset = CornellLabeledNodesDataset(
        ...     data_dir='./data',
        ...     data_name='walmart-trips'
        ... )
        >>> data = dataset[0]
        >>> print(f"Nodes: {data.num_nodes}, Hyperedges: {data.num_hyperedges}")
        Nodes: 88860, Hyperedges: 69906

    Notes
    -----
    This class supports only single-label classification (one label per node).
    Multi-label datasets like stackoverflow-answers and mathoverflow-answers
    require a different implementation.
    """

    # Available datasets with their Google Drive IDs
    DATASETS = {
        "walmart-trips": "1kl6wuvopJ5_wvEIy6YnwoXh1SjlIpRhu",
        "trivago-clicks": "1Mcl28gC0YiQF0NOtWhobhvJU634c04xJ",
        "house-committees": "1402DFpsii-mqeBGk2mf-tDj8P-6Sk6tr",
        "senate-committees": "17ZRVwki_x_C_DlOAea5dPBO7Q4SRTRRw",
        "house-bills": "1-qiSw7YPfiTzJlA73MEH6jlxibHwlhFr",
        "senate-bills": "1DDrJO5fwDGuvMfnnGvrV_m6fA_YHojKF",
        "contact-primary-school": "1H7PGDPvjCyxbogUqw17YgzMc_GHLjbZA",
        "contact-high-school": "163ehVtR-HGVA-wmH88AJ2U2ev4bv2TG6",
        "amazon-reviews": "1dOeke9Rdh0vySIrsSqIZbGXIggVFqZwP",
    }

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        transform=None,
        pre_transform=None,
    ):
        """Initialize the Cornell labeled nodes dataset."""
        if data_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset '{data_name}'. "
                f"Available datasets: {list(self.DATASETS.keys())}"
            )

        self.name = data_name
        self.data_dir = data_dir
        root = os.path.join(data_dir, data_name)

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Return list of raw file names to check if download is needed.

        Returns
        -------
        list
            List containing the zip filename.
        """
        return [f"{self.name}.zip"]

    @property
    def processed_file_names(self):
        """Return list of processed file names.

        Returns
        -------
        list
            List containing the processed data filename.
        """
        return ["data.pt"]

    def download(self):
        """Download the dataset from Google Drive."""
        file_id = self.DATASETS[self.name]
        download_google_url(file_id, self.raw_dir, f"{self.name}.zip")
        print("Download complete.")

    def process(self):
        """Convert raw data into PyTorch Geometric Data format."""
        import shutil

        print(f"Extracting {os.path.join(self.raw_dir, self.name + '.zip')}")
        extract_zip(
            os.path.join(self.raw_dir, self.name + ".zip"), self.raw_dir
        )

        # Move files from subdirectory to raw_dir
        subdir = os.path.join(self.raw_dir, self.name)
        if os.path.exists(subdir):
            for file in os.listdir(subdir):
                src = os.path.join(subdir, file)
                dst = os.path.join(self.raw_dir, file)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            # Remove empty subdirectory
            os.rmdir(subdir)

        print("Processing...")
        data = self._load_cornell_format()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
        print("Done!")

    def _load_cornell_format(self):
        """Load hypergraph data from Cornell text format.

        Uses the shared parsing utility from io_utils to ensure consistency
        across the codebase. The parser automatically handles 0-indexing and
        deduplicates nodes within hyperedges.

        Returns
        -------
        Data
            PyTorch Geometric Data object containing:
            - x: Node features (constant features if not provided)
            - edge_index: COO format [node_id, hyperedge_id]
            - incidence_hyperedges: Sparse incidence matrix
            - y: Node labels (0-indexed)
            - num_nodes: Number of nodes
            - num_hyperedges: Number of hyperedges
            - num_class: Number of classes
        """
        # Use shared parsing utility
        parsed = parse_cornell_hypergraph_files(self.raw_dir, self.name)

        # Extract parsed data
        labels = parsed["labels"]
        edge_index = parsed["edge_index"]
        num_nodes = parsed["num_nodes"]
        num_hyperedges = parsed["num_hyperedges"]
        num_class = parsed["num_classes"]

        # Build incidence matrix (num_nodes x num_hyperedges)
        incidence_values = torch.ones(edge_index.shape[1], dtype=torch.float32)
        incidence_hyperedges = torch.sparse_coo_tensor(
            edge_index,
            incidence_values,
            (num_nodes, num_hyperedges),
            dtype=torch.float32,
        )

        # Create constant node features (can be replaced by transforms)
        x = torch.ones((num_nodes, 1), dtype=torch.float32)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            incidence_hyperedges=incidence_hyperedges,
            y=labels,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            num_class=num_class,
        )

        return data
