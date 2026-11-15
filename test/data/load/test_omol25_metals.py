"""Tests for OMol25 metals dataset and loader."""

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.datasets.omol25_metals_dataset import OMol25MetalsDataset
from topobench.data.loaders.hypergraph.omol25_metals_dataset_loader import (
    OMol25MetalsDatasetLoader,
)


def _create_dummy_omol25_metals_root(
    base_dir: Path,
    num_graphs: int = 4,
    num_nodes: int = 5,
    num_node_features: int = 3,
) -> Path:
    """Create a tiny synthetic OMol25Metals processed directory.

    Parameters
    ----------
    base_dir : Path
        Base directory where the OMol25Metals subdirectory will be created.
    num_graphs : int, optional
        Number of synthetic graphs to create (default: 4).
    num_nodes : int, optional
        Number of nodes per graph (default: 5).
    num_node_features : int, optional
        Number of node features (default: 3).

    Returns
    -------
    Path
        Path to the created OMol25Metals root directory.
    """
    root = base_dir / "omol25_metals"
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    data_list = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes, num_node_features)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            dtype=torch.long,
        )
        # Scalar regression target
        y = torch.tensor([float(i)], dtype=torch.float32)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), processed / "data.pt")
    return root


def test_omol25_metals_dataset_basic(tmp_path):
    """Check that the dataset loads and returns correct shapes.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest fixture.
    """
    root = _create_dummy_omol25_metals_root(tmp_path)
    dataset = OMol25MetalsDataset(root=str(root))

    assert len(dataset) == 4
    item = dataset[0]

    assert item.x.shape == (5, 3)
    assert item.y.shape == (1,)
    assert item.edge_index.shape[0] == 2


def test_omol25_metals_loader_splits(tmp_path):
    """Check that the loader produces non-empty train/val/test splits.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest fixture.
    """
    base_dir = tmp_path
    _create_dummy_omol25_metals_root(base_dir)

    loader = OMol25MetalsDatasetLoader(
        parameters=DictConfig({
            "data_domain": "hypergraph",
            "data_type": "omol25_metals",
            "data_name": "omol25_metals",
            "data_dir": str(base_dir),
        })
    )

    split_params = {
        "learning_setting": "inductive",
        "split_type": "random_in_train",
        "data_seed": 0,
        "train_prop": 0.5,
        "val_prop": 0.25,
    }

    splits = loader.get_splits(split_params)

    assert set(splits.keys()) == {"train", "val", "test"}
    total = sum(len(ds) for ds in splits.values())
    assert total == 4
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_omol25_metals_loader_dataloaders(tmp_path):
    """Check that the loader returns working dataloaders.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest fixture.
    """
    base_dir = tmp_path
    _create_dummy_omol25_metals_root(base_dir)

    loader = OMol25MetalsDatasetLoader(
        parameters=DictConfig({
            "data_domain": "hypergraph",
            "data_type": "omol25_metals",
            "data_name": "omol25_metals",
            "data_dir": str(base_dir),
        })
    )

    split_params = {
        "learning_setting": "inductive",
        "split_type": "random_in_train",
        "data_seed": 0,
        "train_prop": 0.5,
        "val_prop": 0.25,
    }

    dataloader_params = {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }

    loaders = loader.get_dataloaders(split_params, dataloader_params)

    for split_name in ("train", "val", "test"):
        dl = loaders[split_name]
        batch = next(iter(dl))

        # Basic sanity checks on batch structure
        assert batch.x.ndim == 2
        assert batch.y.ndim >= 1
        assert batch.edge_index.shape[0] == 2
