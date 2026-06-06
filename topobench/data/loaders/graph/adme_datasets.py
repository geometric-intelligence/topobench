"""Loaders for TDC (Therapeutics Data Commons) ADME datasets with SMILES to graph conversion."""

import os
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

try:
    from ogb.utils.mol import smiles2graph
    from tdc.single_pred import ADME

    _ADME_DEPS_AVAILABLE = True
except ImportError:
    _ADME_DEPS_AVAILABLE = False

from topobench.data.loaders.base import AbstractLoader


class ADMEDatasetLoader(AbstractLoader):
    """Load TDC ADME datasets with SMILES to graph conversion using OGB featurization.

    This loader:
    1. Loads ADME datasets from TDC (Therapeutics Data Commons)
    2. Converts SMILES strings to PyG graphs using OGB's standard featurization
    3. Uses fixed scaffold splits from TDC
    4. Returns graphs compatible with OGB molecular property prediction

    Node features (9-dimensional):
        - Atomic number
        - Chirality
        - Degree
        - Formal charge
        - Number of hydrogens
        - Number of radical electrons
        - Hybridization
        - Is aromatic
        - Is in ring

    Edge features (3-dimensional):
        - Bond type
        - Bond stereochemistry
        - Is conjugated

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the ADME dataset
            - data_type: Type of the dataset (e.g., "ADME")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> InMemoryDataset:
        """Load the ADME dataset with predefined scaffold splits.

        Returns
        -------
        InMemoryDataset
            The dataset with converted graphs and predefined splits.

        Raises
        ------
        RuntimeError
            If dataset loading or SMILES conversion fails.
        ValueError
            If invalid SMILES strings are encountered.
        ImportError
            If `PyTDC` or `rdkit` (via `ogb`) are not installed.
        """
        if not _ADME_DEPS_AVAILABLE:
            raise ImportError(
                "ADME datasets require additional dependencies. "
                "Install them with: pip install PyTDC rdkit"
            )

        class _ADMEDataset(InMemoryDataset):
            """Internal InMemoryDataset for ADME data.

            Parameters
            ----------
            root : str
                Root directory where the dataset should be stored.
            data_name : str
                Name of the ADME dataset.
            split_idx : dict
                Dictionary mapping split names ("train", "valid", "test")
                to index tensors.
            graph_list : list[torch_geometric.data.Data]
                List of pre-computed PyG graphs to be collated.
            """

            def __init__(self, root, data_name, split_idx, graph_list):
                self.data_name = data_name
                self.split_idx = split_idx
                self._graph_list = graph_list
                super().__init__(root)
                self.data, self.slices = torch.load(self.processed_paths[0])

            @property
            def processed_file_names(self):
                """Return the list of processed dataset file names.

                Returns
                -------
                list[str]
                    File names used to cache the processed dataset.
                """
                return [f"{self.data_name}.pt"]

            def process(self):
                """Collate the pre-built graphs and save them to disk."""
                self.data, self.slices = self.collate(self._graph_list)
                torch.save((self.data, self.slices), self.processed_paths[0])

            def __repr__(self):
                return f"ADMEDataset({self.data_name}, {len(self)})"

        # Define which datasets are classification vs regression
        CLASSIFICATION_DATASETS = {
            # Absorption
            "PAMPA_NCATS",
            "HIA_Hou",
            "Pgp_Broccatelli",
            "Bioavailability_Ma",
            # Distribution
            "BBB_Martins",
            # Metabolism - CYP Inhibition
            "CYP1A2_Veith",
            "CYP2C9_Veith",
            "CYP2C19_Veith",
            "CYP2D6_Veith",
            "CYP3A4_Veith",
            # Metabolism - CYP Substrate
            "CYP2C9_Substrate_CarbonMangels",
            "CYP2D6_Substrate_CarbonMangels",
            "CYP3A4_Substrate_CarbonMangels",
        }

        REGRESSION_DATASETS = {
            # Absorption
            "Caco2_Wang",
            "Lipophilicity_AstraZeneca",
            "Solubility_AqSolDB",
            "HydrationFreeEnergy_FreeSolv",
            # Distribution
            "PPBR_AZ",
            "VDss_Lombardo",
            # Excretion
            "Half_Life_Obach",
            "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ",
        }

        # Determine task type
        dataset_name = self.parameters.data_name
        if dataset_name in CLASSIFICATION_DATASETS:
            is_classification = True
        elif dataset_name in REGRESSION_DATASETS:
            is_classification = False
        else:
            raise ValueError(
                f"Unknown ADME dataset: {dataset_name}. "
                f"Please add it to CLASSIFICATION_DATASETS or REGRESSION_DATASETS."
            )

        # Create raw data directory for TDC to download to
        raw_dir = os.path.join(self.root_data_dir, dataset_name, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Load data from TDC with scaffold split, specify path for downloads
        data = ADME(name=dataset_name, path=raw_dir)
        split = data.get_split()

        # Convert splits to graphs
        graph_list = []
        train_data = split["train"]
        valid_data = split["valid"]
        test_data = split["test"]

        # Process each split
        for split_data in [train_data, valid_data, test_data]:
            for _, row in split_data.iterrows():
                smiles = row["Drug"]
                label = row["Y"]

                # Convert SMILES to graph using OGB's standard featurization
                graph_dict = smiles2graph(smiles)

                # Create PyG Data object
                if is_classification:
                    label_tensor = torch.tensor(int(label), dtype=torch.long)
                else:
                    label_tensor = torch.tensor([label], dtype=torch.float)

                pyg_graph = Data(
                    x=torch.tensor(graph_dict["node_feat"], dtype=torch.float),
                    edge_index=torch.tensor(
                        graph_dict["edge_index"], dtype=torch.long
                    ),
                    edge_attr=torch.tensor(
                        graph_dict["edge_feat"], dtype=torch.float
                    ),
                    y=label_tensor,
                    num_nodes=graph_dict["num_nodes"],
                )

                graph_list.append(pyg_graph)

        # Prepare split indices
        split_idx = {
            "train": torch.arange(len(train_data)),
            "valid": torch.arange(
                len(train_data), len(train_data) + len(valid_data)
            ),
            "test": torch.arange(
                len(train_data) + len(valid_data),
                len(train_data) + len(valid_data) + len(test_data),
            ),
        }

        # Create dataset - point to data/graph/ADME/{dataset_name}
        dataset_root = os.path.join(self.root_data_dir, dataset_name)
        dataset = _ADMEDataset(
            root=dataset_root,
            data_name=self.parameters.data_name,
            split_idx=split_idx,
            graph_list=graph_list,
        )

        # Attach split_idx to the dataset for compatibility with framework
        dataset.split_idx = split_idx

        return dataset

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
            Format: {root_data_dir}/{dataset_name}/.
            Example: data/graph/ADME/BBB_Martins/.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)
