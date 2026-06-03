import os
import torch
import pytest
from torch_geometric.data import Data
from topobench.data.core.qm9_robust_on_disk_dataset import QM9RobustOnDiskDataset


@pytest.fixture
def tmp_dataset_100000(tmp_path, monkeypatch):
    """
    Fixture creating 100000 synthetic samples with chunk_size=5000.
    """

    class FakeQM9:
        raw_file_names = [f"mol_{i}.xyz" for i in range(100000)]

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

    dataset = QM9RobustOnDiskDataset(root=str(tmp_path / "qm9_100000"), chunk_size=5000)
    return dataset


def test_dataset_length_100000(tmp_dataset_100000):
    assert tmp_dataset_100000.len() == 100000


def test_chunks_are_saved_100000(tmp_dataset_100000):
    chunk_files = [
        f for f in os.listdir(tmp_dataset_100000.processed_dir)
        if f.startswith("chunk_") and f.endswith(".pt")
    ]
    # 100000 / 5000 = 20 chunks
    assert len(chunk_files) == 20


def test_dataset_get_100000(tmp_dataset_100000):
    for i in range(0, 100000, 10000):  # test every 10000th sample
        data = tmp_dataset_100000.get(i)
        assert isinstance(data, Data)
        assert data.x.shape == (5, 11)
