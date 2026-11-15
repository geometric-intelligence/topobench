import os
import torch
import pytest
from torch_geometric.data import Data
from topobench.data.core.qm9_robust_on_disk_dataset import QM9RobustOnDiskDataset


@pytest.fixture
def tmp_dataset_10000(tmp_path, monkeypatch):
    """
    Fixture creating 10000 synthetic samples with chunk_size=500.
    """

    class FakeQM9:
        raw_file_names = [f"mol_{i}.xyz" for i in range(10000)]

        def __init__(self, root):
            self.root = root
            os.makedirs(os.path.join(root, "raw"), exist_ok=True)
            self.raw_paths = []
            for name in self.raw_file_names:
                path = os.path.join(root, "raw", name)
                tensor = torch.rand(5, 11)
                torch.save(tensor, path)
                self.raw_paths.append(path)

    monkeypatch.setattr(
        "topobench.data.core.qm9_robust_on_disk_dataset.QM9", lambda root: FakeQM9(root)
    )

    dataset = QM9RobustOnDiskDataset(root=str(tmp_path / "qm9_10000"), chunk_size=500)
    return dataset


def test_dataset_length_10000(tmp_dataset_10000):
    assert tmp_dataset_10000.len() == 10000


def test_chunks_are_saved_10000(tmp_dataset_10000):
    chunk_files = [
        f for f in os.listdir(tmp_dataset_10000.processed_dir)
        if f.startswith("chunk_") and f.endswith(".pt")
    ]
    # 10000 / 500 = 20 chunks
    assert len(chunk_files) == 20


def test_dataset_get_10000(tmp_dataset_10000):
    for i in range(0, 10000, 1000):  # test every 1000th sample for speed
        data = tmp_dataset_10000.get(i)
        assert isinstance(data, Data)
        assert data.x.shape == (5, 11)
