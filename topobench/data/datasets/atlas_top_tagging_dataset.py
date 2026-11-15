"""ATLAS Top Tagging Dataset - Modified to read directly from compressed files."""

import glob
import gzip
import os
import os.path as osp
from collections.abc import Callable

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from tqdm import tqdm


class ATLASTopTaggingDataset(InMemoryDataset):
    """ATLAS Top Tagging Open Dataset for boosted top quark identification.

    The dataset contains jets from simulated LHC collisions with constituent
    four-vectors, high-level features, and labels (signal=1, background=0).

    Parameters
    ----------
    root : str
        Where to download (or look for) the dataset.
    split : str
        Which split to use: 'train' or 'test'. Default is 'train'.
    subset : float
        Fraction of dataset to be used (0.0 to 1.0). Default is 0.01 (1%).
        Due to large dataset size, using a small subset is recommended.
    max_constituents : int
        Maximum number of constituents to use per jet. Default is 80.
    use_high_level : bool
        Whether to include high-level features. Default is True.
    verbose : bool
        Whether to display more info while processing graphs.
    transform : callable
        A function/transform that takes in an `torch_geometric.data.Data`
        object and returns a transformed version.
    pre_transform : callable
        A function/transform that takes in an `torch_geometric.data.Data`
        object and returns a transformed version.
    pre_filter : callable
        A function that takes in an `torch_geometric.data.Data` object and
        returns a boolean value, indicating whether the data object should
        be included in the final dataset.
    post_filter : callable
        A function that takes in an `torch_geometric.data.Data` object and
        returns a boolean value, indicating whether the data object should
        be included in the final dataset.
    **kwargs
        Additional arguments.
    """

    # URLs for the nominal training and testing datasets
    URLS = {
        "train": "https://opendata.cern.ch/record/80030/files/train_nominal.tar.gz",
        "test": "https://opendata.cern.ch/record/80030/files/test_nominal.tar.gz",
    }

    # Branch names for constituent features
    CONSTITUENT_BRANCHES = [
        "fjet_clus_pt",
        "fjet_clus_eta",
        "fjet_clus_phi",
        "fjet_clus_E",
    ]

    # Branch names for high-level features
    HIGH_LEVEL_BRANCHES = [
        "fjet_C2",
        "fjet_D2",
        "fjet_ECF1",
        "fjet_ECF2",
        "fjet_ECF3",
        "fjet_L2",
        "fjet_L3",
        "fjet_Qw",
        "fjet_Split12",
        "fjet_Split23",
        "fjet_Tau1_wta",
        "fjet_Tau2_wta",
        "fjet_Tau3_wta",
        "fjet_Tau4_wta",
        "fjet_ThrustMaj",
    ]

    # Branch names for jet-level features
    JET_BRANCHES = ["fjet_pt", "fjet_eta", "fjet_phi", "fjet_m"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        subset: float = 0.01,
        max_constituents: int = 80,
        use_high_level: bool = True,
        verbose: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        post_filter: Callable | None = None,
        **kwargs,
    ):
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        assert 0.0 < subset <= 1.0, "subset must be between 0.0 and 1.0"

        self.split = split
        self.subset = subset
        self.max_constituents = max_constituents
        self.use_high_level = use_high_level
        self.verbose = verbose
        self.post_filter = post_filter
        self.kwargs = kwargs

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices, loaded_config = torch.load(
            self.processed_paths[0], weights_only=False
        )

        print(
            f"Loaded dataset: split={loaded_config['split']}, "
            f"subset={loaded_config['subset']:.3f}, "
            f"max_constituents={loaded_config['max_constituents']}"
        )

        # Check if loaded config matches requested config
        if (
            loaded_config["split"] != self.split
            or loaded_config["subset"] != self.subset
            or loaded_config["max_constituents"] != self.max_constituents
            or loaded_config["use_high_level"] != self.use_high_level
        ):
            print(
                "The loaded dataset has different settings from the ones requested. "
                "Processing graphs again."
            )
            self.process()
            self.data, self.slices, loaded_config = torch.load(
                self.processed_paths[0], weights_only=False
            )

    def __repr__(self):
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}(split={self.split}, "
            f"n_jets={len(self)}, subset={self.subset:.3f})"
        )

    def _total_files_for_split(self) -> int:
        """Return total number of files for the current split.

        Returns
        -------
        int
            Total number of files for 'train' or 'test' split.
        """
        # Approximate file counts per split (adjust if you know the exact numbers)
        return 930 if self.split == "train" else 100

    def _expected_filenames(self) -> list:
        """Return list of expected raw file names for current split and subset.

        Returns
        -------
        list
            List of expected raw file names.
        """
        total = self._total_files_for_split()
        n = max(1, min(total, int(round(total * float(self.subset)))))
        prefix = f"{self.split}_nominal_"
        return [f"{prefix}{i:03d}.h5.gz" for i in range(n)]

    @property
    def raw_file_names(self):
        """Return list of raw file names to download.

        Returns
        -------
        list
            List of raw file names.
        """
        split_dir = f"{self.split}_nominal"
        return [osp.join(split_dir, fn) for fn in self._expected_filenames()]

    @property
    def processed_file_names(self):
        """Return name of processed dataset file.

        Returns
        -------
        list
            Single-element list with processed file name.
        """
        filename = f"atlas_top_tagging_{self.split}_subset{self.subset:.3f}_maxconst{self.max_constituents}.pt"
        return [filename]

    @property
    def pre_processed_path(self):
        """Return path to preprocessed data file.

        Returns
        -------
        str
            Path to p .pt file.
        """
        return osp.join(self.raw_dir, f"preprocessed_{self.split}.pt")

    def download(self):
        """Download the ATLAS Top Tagging dataset files.

        Downloads individual .h5.gz files from CERN OpenData.
        Based on the subset parameter, downloads only the needed files.
        """
        import time
        import urllib.request

        print(f"\n{'=' * 60}")
        print(
            f"DOWNLOADING ATLAS TOP TAGGING DATASET ({self.split.upper()} SPLIT)"
        )
        print(f"{'=' * 60}")

        # Base URL for HTTP access (converted from root:// protocol)
        base_url = "https://opendata.cern.ch/eos/opendata/atlas/datascience/CERN-EP-2024-159"

        # Create split directory
        split_dir = osp.join(self.raw_dir, f"{self.split}_nominal")
        os.makedirs(split_dir, exist_ok=True)

        # Determine how many files to download based on subset
        # Each file contains ~100k jets, total ~930 files for train, ~100 for test
        total_files = 930 if self.split == "train" else 100

        num_files_to_download = max(1, int(total_files * self.subset))

        print(f"Subset: {self.subset:.3f} ({self.subset * 100:.1f}%)")
        print(
            f"Downloading {num_files_to_download} out of {total_files} files"
        )
        print(
            f"This will download approximately {num_files_to_download * 0.3:.1f} GB"
        )
        print(f"{'=' * 60}\n")

        # Download files
        downloaded = 0
        for i in range(num_files_to_download):
            # File naming: train_nominal_000.h5.gz, train_nominal_001.h5.gz, etc.
            file_num = str(i).zfill(3)
            filename = f"{self.split}_nominal_{file_num}.h5.gz"
            file_url = f"{base_url}/{filename}"
            file_path = osp.join(split_dir, filename)

            # Skip if already downloaded (check .gz file only, not decompressed)
            if osp.exists(file_path):
                print(
                    f"[{i + 1}/{num_files_to_download}] Already exists: {filename}"
                )
                downloaded += 1
                continue

            # Download file
            print(
                f"[{i + 1}/{num_files_to_download}] Downloading: {filename}..."
            )
            try:
                urllib.request.urlretrieve(file_url, file_path)
                downloaded += 1
                print(
                    f"  ✅ Downloaded ({downloaded}/{num_files_to_download})"
                )
                time.sleep(0.5)  # Be nice to the server
            except Exception as e:
                print(f"  ⚠️  Failed to download {filename}: {e}")
                print(f"  You can manually download from: {file_url}")
                # Continue with other files
                continue

        print(f"\n{'=' * 60}")
        print(f"Download complete: {downloaded}/{num_files_to_download} files")
        print(f"Location: {split_dir}")
        print(f"{'=' * 60}\n")

        if downloaded == 0:
            raise FileNotFoundError(
                f"Failed to download any files. Please check your internet connection\n"
                f"or manually download files from:\n"
                f"{base_url}/{self.split}_nominal_*.h5.gz"
            )

    def _load_h5_file_flexible(
        self,
        file_path: str,
        use_compressed: bool = True,
        num_jets: int | None = None,
    ):
        """Load data from either compressed (.h5.gz) or uncompressed (.h5) HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file (.h5 or .h5.gz).
        use_compressed : bool
            If True, reads directly from .gz file. If False, reads from .h5 file.
        num_jets : int, optional
            Number of jets to load from this file. If None, loads all.

        Returns
        -------
        dict
            Dictionary containing all the loaded arrays.
        """
        data_dict = {}

        if use_compressed:
            # Open compressed file directly without decompressing to disk
            with (
                gzip.open(file_path, "rb") as gz_file,
                h5py.File(gz_file, "r") as f,
            ):
                data_dict = self._extract_data_from_h5(f, num_jets)
        else:
            # Open regular uncompressed file
            with h5py.File(file_path, "r") as f:
                data_dict = self._extract_data_from_h5(f, num_jets)

        return data_dict

    def _extract_data_from_h5(self, f: h5py.File, num_jets: int | None = None):
        """Extract data from an open HDF5 file handle.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle.
        num_jets : int, optional
            Number of jets to load. If None, loads all.

        Returns
        -------
        dict
            Dictionary containing all the loaded arrays.
        """
        data_dict = {}

        # Load labels
        labels = f["labels"][:]
        if num_jets is not None:
            labels = labels[:num_jets]

        total_jets = len(labels)
        if num_jets is not None:
            total_jets = min(total_jets, num_jets)

        data_dict["labels"] = labels[:total_jets]

        # Load constituent features
        for branch in self.CONSTITUENT_BRANCHES:
            data_dict[branch] = f[branch][:total_jets]

        # Load high-level features if requested
        if self.use_high_level:
            for branch in self.HIGH_LEVEL_BRANCHES:
                if branch in f:
                    data_dict[branch] = f[branch][:total_jets]

        # Load jet-level features
        for branch in self.JET_BRANCHES:
            if branch in f:
                data_dict[branch] = f[branch][:total_jets]

        # Load training weights if available (only in train split)
        if "training_weights" in f:
            data_dict["training_weights"] = f["training_weights"][:total_jets]

        return data_dict

    def _load_h5_file(self, h5_path: str, num_jets: int | None = None):
        """Load data from a single HDF5 file (backwards compatibility).

        Parameters
        ----------
        h5_path : str
            Path to the HDF5 file.
        num_jets : int, optional
            Number of jets to load from this file. If None, loads all.

        Returns
        -------
        dict
            Dictionary containing all the loaded arrays.
        """
        # Determine if file is compressed
        is_compressed = h5_path.endswith(".gz")
        return self._load_h5_file_flexible(
            h5_path, use_compressed=is_compressed, num_jets=num_jets
        )

    def _preprocess(self):
        """Preprocessing the raw HDF5 files by reading directly from compressed files."""
        print(f"\n[Preprocessing] Building dataset for {self.split} split...")
        print(f"[Preprocessing] Using subset={self.subset:.3f} of the data")

        # Look for .h5.gz files (compressed)
        gz_pattern = osp.join(self.raw_dir, f"{self.split}_nominal", "*.h5.gz")
        gz_files = sorted(glob.glob(gz_pattern))

        if len(gz_files) == 0:
            # Fallback: look for already decompressed .h5 files
            h5_pattern = osp.join(
                self.raw_dir, f"{self.split}_nominal", "*.h5"
            )
            h5_files = sorted(glob.glob(h5_pattern))

            if len(h5_files) == 0:
                raise FileNotFoundError(
                    f"No .h5.gz or .h5 files found in {self.raw_dir}/{self.split}_nominal/"
                )

            print(
                f"[Preprocessing] Found {len(h5_files)} decompressed .h5 files"
            )
            files_to_process = h5_files
            use_compressed = False
        else:
            print(
                f"[Preprocessing] Found {len(gz_files)} compressed .h5.gz files"
            )
            print(
                "[Preprocessing] Reading directly from compressed files (saves disk space)..."
            )
            files_to_process = gz_files
            use_compressed = True

        # Calculate how many jets to load based on subset
        all_data = []
        total_jets_needed = None
        jets_loaded = 0

        # Load data from files until we reach the subset amount
        desc_msg = (
            "Loading from compressed files"
            if use_compressed
            else "Loading HDF5 files"
        )
        for file_path in tqdm(files_to_process, desc=desc_msg):
            if (
                total_jets_needed is not None
                and jets_loaded >= total_jets_needed
            ):
                break

            # Load data from this file (handles both .gz and uncompressed)
            data_dict = self._load_h5_file_flexible(
                file_path, use_compressed=use_compressed
            )

            # On first file, determine total jets needed
            if total_jets_needed is None:
                # Estimate total jets across all files
                # (assuming roughly equal distribution)
                jets_in_file = len(data_dict["labels"])
                estimated_total = jets_in_file * len(files_to_process)
                total_jets_needed = int(estimated_total * self.subset)
                print(
                    f"\n[Preprocessing] Target: {total_jets_needed} jets "
                    f"({self.subset * 100:.1f}% of estimated {estimated_total})"
                )

            all_data.append(data_dict)
            jets_loaded += len(data_dict["labels"])

            if self.verbose:
                print(
                    f"[Preprocessing] Loaded {jets_loaded}/{total_jets_needed} jets"
                )

        print(
            f"[Preprocessing] Loaded {jets_loaded} jets from {len(all_data)} files"
        )

        # Concatenate all data
        print("[Preprocessing] Concatenating data from all files...")
        combined_data = {}
        for key in all_data[0]:
            combined_data[key] = np.concatenate(
                [d[key] for d in all_data], axis=0
            )

        # Trim to exact subset size if we loaded more than needed
        if (
            total_jets_needed is not None
            and len(combined_data["labels"]) > total_jets_needed
        ):
            for key in combined_data:
                combined_data[key] = combined_data[key][:total_jets_needed]

        print(
            f"[Preprocessing] Final dataset size: {len(combined_data['labels'])} jets"
        )

        # Convert to PyTorch tensors and save
        print("[Preprocessing] Converting to PyTorch tensors...")
        for key in combined_data:
            combined_data[key] = torch.from_numpy(combined_data[key]).float()

        torch.save(combined_data, self.pre_processed_path)
        print(
            f"[Preprocessing] Saved preprocessed data to {self.pre_processed_path}"
        )

    def process(self):
        """Processing the dataset into PyTorch Geometric Data objects."""
        # If raw data was not preprocessed, do it now
        if not osp.exists(self.pre_processed_path):
            self._preprocess()

        # Load preprocessed data
        print("[Processing] Loading preprocessed data...")
        data_dict = torch.load(self.pre_processed_path, weights_only=False)

        num_jets = len(data_dict["labels"])
        print(f"[Processing] Converting {num_jets} jets to graph format...")

        data_list = []

        for i in tqdm(range(num_jets), desc="Building graphs"):
            # Extract constituent features
            pt = data_dict["fjet_clus_pt"][i]
            eta = data_dict["fjet_clus_eta"][i]
            phi = data_dict["fjet_clus_phi"][i]
            energy = data_dict["fjet_clus_E"][i]

            # Find valid constituents (non-zero pt)
            valid_mask = pt > 0
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                continue  # Skip jets with no constituents

            # Limit to max_constituents
            if num_valid > self.max_constituents:
                valid_mask = torch.zeros_like(pt, dtype=torch.bool)
                valid_mask[: self.max_constituents] = True
                num_valid = self.max_constituents

            # Build node feature matrix [num_constituents, 4]
            x = torch.stack(
                [
                    pt[valid_mask],
                    eta[valid_mask],
                    phi[valid_mask],
                    energy[valid_mask],
                ],
                dim=1,
            )

            # Creare k-NN graph connectivity
            edge_index = knn_graph(x, k=5, loop=False)

            # Add high-level features as graph-level attributes if requested
            graph_attrs = {}
            if self.use_high_level:
                hl_features = [
                    data_dict[branch][i].unsqueeze(0)
                    for branch in self.HIGH_LEVEL_BRANCHES
                    if branch in data_dict
                ]
                if len(hl_features) > 0:
                    graph_attrs["high_level_features"] = torch.cat(
                        hl_features, dim=0
                    )

            # Add jet-level features
            jet_features = [
                data_dict[branch][i].unsqueeze(0)
                for branch in self.JET_BRANCHES
                if branch in data_dict
            ]
            if len(jet_features) > 0:
                graph_attrs["jet_features"] = torch.cat(jet_features, dim=0)

            # Add training weight if available
            if "training_weights" in data_dict:
                graph_attrs["training_weight"] = data_dict["training_weights"][
                    i
                ]

            # Create Data object
            y = data_dict["labels"][i].long()

            data = Data(
                x=x,
                y=y,
                edge_index=edge_index,  # No edges initially, can be added via transforms
                edge_attr=None,
                num_nodes=num_valid,
                **graph_attrs,
            )

            data_list.append(data)

        print(f"[Processing] Created {len(data_list)} graphs")

        # Apply filters
        if self.pre_filter is not None:
            print("[Processing] Applying pre-filter...")
            data_list = [
                data for data in tqdm(data_list) if self.pre_filter(data)
            ]
            print(f"[Processing] {len(data_list)} graphs after pre-filter")

        if self.pre_transform is not None:
            print("[Processing] Applying pre-transform...")
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        if self.post_filter is not None:
            print("[Processing] Applying post-filter...")
            data_list = [
                data for data in tqdm(data_list) if self.post_filter(data)
            ]
            print(f"[Processing] {len(data_list)} graphs after post-filter")

        # Collate and save
        print("[Processing] Collating and saving...")
        data, slices = self.collate(data_list)

        # Save configuration along with data
        config = {
            "split": self.split,
            "subset": self.subset,
            "max_constituents": self.max_constituents,
            "use_high_level": self.use_high_level,
        }

        torch.save((data, slices, config), self.processed_paths[0])
        print(
            f"[Processing] Saved processed dataset to {self.processed_paths[0]}"
        )

    # Properties
    @property
    def num_classes(self) -> int:
        """Return number of classes.

        Returns
        -------
        int
            Number of classes (2).
        """
        return 2

    @property
    def num_features(self) -> int:
        """Return number of node features.

        Returns
        -------
        int
            Number of features (4).
        """
        return 4

    @property
    def num_high_level_features(self) -> int:
        """Return number of high-level features.

        Returns
        -------
        int
            Number of high-level features (15 or 0).
        """
        return len(self.HIGH_LEVEL_BRANCHES) if self.use_high_level else 0

    def stats(self):
        """Print dataset statistics."""
        print("\n*** ATLAS Top Tagging Dataset ***\n")
        print(f"Split: {self.split}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of jets: {len(self)}")
        print(f"Number of constituent features: {self.num_features}")
        print(f"Number of high-level features: {self.num_high_level_features}")
        print(f"Max constituents per jet: {self.max_constituents}")

        # Calculate class distribution
        labels = [self.get(i).y.item() for i in range(len(self))]
        num_signal = sum(labels)
        num_background = len(labels) - num_signal
        print(
            f"Signal jets: {num_signal} ({num_signal / len(labels) * 100:.1f}%)"
        )
        print(
            f"Background jets: {num_background} ({num_background / len(labels) * 100:.1f}%)"
        )

        # Average number of constituents
        avg_constituents = sum(
            [self.get(i).num_nodes for i in range(len(self))]
        ) / len(self)
        print(f"Average constituents per jet: {avg_constituents:.1f}")
