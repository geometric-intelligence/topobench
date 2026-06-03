import os
import torch
import pytest
from torch_geometric.data import Data
from topobench.data.core.qm9_robust_on_disk_dataset import QM9RobustOnDiskDataset


@pytest.fixture
def tmp_dataset(tmp_path, monkeypatch):
    """
    Fixture creating 20 synthetic samples and monkeypatching QM9.
    """

    # Fake QM9 for testing
    class FakeQM9:
        raw_file_names = [f"mol_{i}.xyz" for i in range(20)]

        def __init__(self, root):
            self.root = root
            os.makedirs(os.path.join(root, "raw"), exist_ok=True)
            self.raw_paths = []
            self.processed_paths = []
            for name in self.raw_file_names:
                path = os.path.join(root, "raw", name)
                tensor = torch.rand(5, 11)
                torch.save(tensor, path)
                self.raw_paths.append(path)
                self.processed_paths.append(path)

    # Monkeypatch QM9
    monkeypatch.setattr(
        "topobench.data.core.qm9_robust_on_disk_dataset.QM9", lambda root: FakeQM9(root)
    )

    dataset = QM9RobustOnDiskDataset(root=str(tmp_path / "qm9"), chunk_size=5)
    return dataset


def test_dataset_length(tmp_dataset):
    assert tmp_dataset.len() == 20


def test_dataset_get(tmp_dataset):
    for i in range(20):
        data = tmp_dataset.get(i)
        assert isinstance(data, Data)
        assert data.x.shape == (5, 11)


def test_chunks_are_saved(tmp_dataset):
    chunk_files = [
        f for f in os.listdir(tmp_dataset.processed_dir)
        if f.startswith("chunk_") and f.endswith(".pt")
    ]
    assert len(chunk_files) == 4  # 20 samples / chunk_size=5


def test_index_out_of_range(tmp_dataset):
    with pytest.raises(IndexError):
        tmp_dataset.get(999)
