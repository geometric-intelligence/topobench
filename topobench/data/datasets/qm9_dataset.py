"""Dataset class for QM9 molecular dataset."""

from typing import ClassVar

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import QM9 as PyGQM9
from torch_geometric.io import fs


class QM9Dataset(InMemoryDataset):
    r"""Dataset class for QM9 molecular property prediction dataset.

    QM9 is a comprehensive dataset of quantum mechanical properties for 134k
    stable small organic molecules made up of CHONF (C, H, O, N, F). Each
    molecule has up to 9 heavy atoms and includes 19 regression targets.

    This class wraps PyTorch Geometric's QM9 dataset and adapts it for
    TopoBench's molecular topology learning framework.

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
    targets : list
        List of available target properties for regression.
    """

    # QM9 target properties (19 regression targets)
    TARGETS: ClassVar = [
        "mu",           # Dipole moment (D)
        "alpha",        # Isotropic polarizability (Bohr^3)
        "homo",         # Highest occupied molecular orbital energy (eV)
        "lumo",         # Lowest unoccupied molecular orbital energy (eV)
        "gap",          # HOMO-LUMO gap (eV)
        "r2",           # Electronic spatial extent (Bohr^2)
        "zpve",         # Zero point vibrational energy (eV)
        "u0",           # Internal energy at 0K (eV)
        "u298",         # Internal energy at 298.15K (eV)
        "h298",         # Enthalpy at 298.15K (eV)
        "g298",         # Free energy at 298.15K (eV)
        "cv",           # Heat capacity at 298.15K (cal/(mol*K))
        "u0_atom",      # Atomization energy at 0K (eV)
        "u298_atom",    # Atomization energy at 298.15K (eV)
        "h298_atom",    # Atomization enthalpy at 298.15K (eV)
        "g298_atom",    # Atomization free energy at 298.15K (eV)
        "A",            # Rotational constant A (GHz)
        "B",            # Rotational constant B (GHz)
        "C",            # Rotational constant C (GHz)
    ]

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Get target property index (default to HOMO-LUMO gap)
        self.target_property = parameters.get("target_property", "gap")
        if self.target_property not in self.TARGETS:
            raise ValueError(f"Target property {self.target_property} not in {self.TARGETS}")
        self.target_idx = self.TARGETS.index(self.target_property)

        # Subset size for testing (None for full dataset)
        self.subset_size = parameters.get("subset_size", None)

        super().__init__(root)

        # Load processed data
        out = fs.torch_load(self.processed_paths[0])
        if len(out) == 2:
            data, self.slices = out
            self.sizes = {"total": len(self.slices)}
        elif len(out) == 3:
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
        return f"{self.name}(root={self.root}, target={self.target_property}, size={len(self)})"

    @property
    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def processed_file_names(self):
        return [f"qm9_{self.target_property}.pt"]

    def download(self):
        """Download QM9 dataset using PyTorch Geometric's downloader."""
        # PyG's QM9 dataset handles downloading automatically
        # We just need to ensure the raw files are in the right place
        PyGQM9(root=self.root, transform=None)
        # Files are automatically downloaded to self.raw_dir

    def process(self):
        """Process QM9 dataset for TopoBench molecular topology learning."""
        # Load PyG QM9 dataset
        pyg_dataset = PyGQM9(root=self.root, transform=None)
        
        data_list = []
        
        # Process subset if specified, otherwise full dataset
        dataset_size = len(pyg_dataset) if self.subset_size is None else min(self.subset_size, len(pyg_dataset))
        
        for i in range(dataset_size):
            data = pyg_dataset[i]
            
            # Extract features and target
            processed_data = Data(
                # Node features: atomic numbers
                x=data.z.float().unsqueeze(1),  # [num_atoms, 1]
                
                # Edge features: bond connectivity
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                
                # 3D coordinates for potential geometric features
                pos=data.pos,
                
                # Target property (single regression target)
                y=data.y[0, self.target_idx].unsqueeze(0),  # [1]
                
                # Additional molecular properties
                z=data.z,  # Atomic numbers
                
                # Molecule-level features
                num_atoms=data.z.size(0),
            )
            
            data_list.append(processed_data)

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices, {"total": len(data_list)}), self.processed_paths[0])

    @property
    def num_node_features(self) -> int:
        """Number of node features per atom."""
        return 1  # Atomic number only

    @property
    def num_edge_features(self) -> int:
        """Number of edge features per bond."""
        return 4  # Bond type features from PyG QM9

    @property
    def num_classes(self) -> int:
        """Number of classes (regression task)."""
        return 1  # Single regression target

    def get_target_statistics(self) -> dict:
        """Get statistics for the target property."""
        all_targets = [self[i].y.item() for i in range(len(self))]
        return {
            "mean": torch.tensor(all_targets).mean().item(),
            "std": torch.tensor(all_targets).std().item(),
            "min": torch.tensor(all_targets).min().item(),
            "max": torch.tensor(all_targets).max().item(),
        }
