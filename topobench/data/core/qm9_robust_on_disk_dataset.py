import os

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

from .robust_on_disk_dataset import RobustOnDiskDataset


class QM9RobustOnDiskDataset(RobustOnDiskDataset):
    """
    QM9 dataset using RobustOnDiskDataset.
    Fully compatible with QM9 features, creates chunks from raw files.
    """

    def __init__(self, root: str, chunk_size: int = 512, **kwargs):
        self.qm9 = QM9(root=root)
        super().__init__(root=root, chunk_size=chunk_size, **kwargs)

    @property
    def raw_file_names(self):
        return self.qm9.raw_file_names

    def prepare_raw_data(self):
        return [os.path.join(self.root, "raw", f) for f in self.raw_file_names]

    def process_raw_file(self, raw_path: str):
        # Load .pt file and wrap in Data if needed
        data = torch.load(raw_path)
        if not isinstance(data, Data):
            data = Data(x=data)
        return [data]
