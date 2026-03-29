"""DAWN temporal hypergraph dataset (TopoBench-compatible).

Parses the raw DAWN .gz dataset and produces a processed PyTorch Geometric
InMemoryDataset ready for hypergraph learning tasks.
"""

import gzip
import os
import re
import struct
from collections.abc import Callable

import torch
from torch_geometric.data import Data, InMemoryDataset, download_google_url
from torch_sparse import coalesce


class DawnDataset(InMemoryDataset):
    """TopoBench-compatible loader for the DAWN temporal hypergraph dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    google_drive_url : str
        URL to download the raw dataset.
    name : str | None, optional
        Name of the dataset (used by TopoBench loader), by default None.
    parameters : dict | None, optional
        Configuration parameters (used by TopoBench loader), by default None.
    transform : callable | None, optional
        Function/transform applied to Data objects after processing, by default None.
    pre_transform : callable | None, optional
        Function/transform applied before saving the processed data, by default None.
    """

    def __init__(
        self,
        root: str,
        google_drive_url: str,
        name: str | None = None,
        parameters: dict | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self._google_drive_url = google_drive_url

        # Extract ID from the specific URL format provided in YAML
        # URL example: .../uc?export=download&id=1wGwoG7oBWnNN7J9TEpjqNpODbsYfMxp4
        match = re.search(r"id=([a-zA-Z0-9_-]+)", google_drive_url)
        if match:
            self._file_id = match.group(1)
        else:
            # Fallback: assume the user might have passed just the ID
            self._file_id = google_drive_url

        super().__init__(root, transform, pre_transform)
        # Load processed dataset
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return the expected raw file names.

        Returns
        -------
        list[str]
            List containing the raw file name.
        """
        return ["dawn_dataset.gz"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the expected processed file names.

        Returns
        -------
        list[str]
            List containing the processed file name.
        """
        return ["data.pt"]

    def download(self) -> None:
        """Download dataset from Google Drive.

        Uses the extracted file ID to download the file via PyG's utility.
        """
        print(f"Downloading from Google Drive ID: {self._file_id}")
        # This handles the Google Drive 'confirm' token logic automatically
        download_google_url(self._file_id, self.raw_dir, "dawn_dataset.gz")

    def process(self) -> None:
        """Process the raw DAWN dataset into a PyG Data object.

        Reads the custom binary format, constructs the hypergraph structure,
        coalesces indices, and saves the processed Data object.

        Raises
        ------
        RuntimeError
            If the GZ file cannot be read or is corrupted.
        ValueError
            If parsing fails or the resulting data is empty.
        """
        gz_path = os.path.join(self.raw_dir, "dawn_dataset.gz")

        print(f"Processing {gz_path}...")

        # Verify file is actually GZIP
        try:
            with gzip.open(gz_path, "rb") as f:
                # Read raw bytes
                raw = f.read()
        except Exception as e:
            raise RuntimeError(
                f"Could not read GZ file. The download might have failed "
                f"(check if file is HTML): {e}"
            ) from e

        # Skip header (first 5 bytes as per original specification)
        data_bytes = raw[5:]

        nverts = []
        all_node_ids = []
        timestamps = []

        idx = 0
        total_bytes = len(data_bytes)

        # Parse binary data
        try:
            while idx < total_bytes:
                if idx + 2 > total_bytes:
                    break
                nv = struct.unpack_from("<H", data_bytes, idx)[0]
                idx += 2
                nverts.append(nv)

                if idx + 4 * nv > total_bytes:
                    break
                for _ in range(nv):
                    node_id = struct.unpack_from("<I", data_bytes, idx)[0]
                    all_node_ids.append(node_id - 1)  # 1-indexed -> 0-indexed
                    idx += 4

                if idx + 4 > total_bytes:
                    break
                ts = struct.unpack_from("<I", data_bytes, idx)[0]
                timestamps.append(ts)
                idx += 4
        except struct.error as e:
            raise ValueError(
                "Binary parsing failed. Ensure the dataset is the correct binary format, "
                "not a text file."
            ) from e

        if not nverts:
            raise ValueError("Parsed data is empty. Check raw_file format.")

        # Build hypergraph edge_index [node, hyperedge]
        node_list = []
        edge_list = []
        node_idx = 0

        for e_idx, nv in enumerate(nverts):
            # Get nodes for this hyperedge and remove duplicates (simplicial set)
            simplex_nodes = list(set(all_node_ids[node_idx : node_idx + nv]))
            for node_id in simplex_nodes:
                node_list.append(node_id)
                edge_list.append(e_idx)
            node_idx += nv

        # Determine dimensions
        num_nodes = max(node_list) + 1 if node_list else 0
        num_simplices = len(nverts)

        edge_index = torch.tensor([node_list, edge_list], dtype=torch.long)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_simplices)

        # Node features: ones
        x = torch.ones((num_nodes, 1), dtype=torch.float)
        # Edge timestamps
        edge_timestamps = torch.tensor(timestamps, dtype=torch.float)

        # Incidence Matrix
        incidence_hyperedges = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]),
            size=(num_nodes, num_simplices),
        ).coalesce()

        # PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            incidence_hyperedges=incidence_hyperedges,
            edge_timestamps=edge_timestamps,
            num_nodes=num_nodes,
            n_x=num_nodes,  # TopoBench often requires n_x keys
        )

        # Collate into slices for InMemoryDataset
        data, slices = self.collate([data])
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed dataset saved to {self.processed_paths[0]}")
