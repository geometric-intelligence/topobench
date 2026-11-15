import os
import torch
import pytest
from torch_geometric.data import Data
from topobench.data.core.qm9_robust_on_disk_dataset import QM9RobustOnDiskDataset


@pytest.fixture
def tmp_dataset_1000(tmp_path, monkeypatch):
    """
    Fixture creating 1000 synthetic samples with chunk_size=50.
    """

    class FakeQM9:
        raw_file_names = [f"mol_{i}.xyz" for i in range(1000)]

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

    dataset = QM9RobustOnDiskDataset(root=str(tmp_path / "qm9_1000"), chunk_size=50)
    return dataset


def test_dataset_length_1000(tmp_dataset_1000):
    assert tmp_dataset_1000.len() == 1000


def test_chunks_are_saved_1000(tmp_dataset_1000):
    chunk_files = [
        f for f in os.listdir(tmp_dataset_1000.processed_dir)
        if f.startswith("chunk_") and f.endswith(".pt")
    ]
    # 1000 / 50 = 20 chunks
    assert len(chunk_files) == 20


def test_dataset_get_1000(tmp_dataset_1000):
    for i in range(0, 1000, 100):  # test every 100th sample for speed
        data = tmp_dataset_1000.get(i)
        assert isinstance(data, Data)
        assert data.x.shape == (5, 11)
