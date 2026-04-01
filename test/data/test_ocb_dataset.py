"""Tests for the synthetic OCB datasets and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.data.datasets.ocb_dataset import (
    NUM_NODE_TYPES,
    OCB101Dataset,
    OCB301Dataset,
    OCBDataset,
)
from topobench.data.loaders.graph.ocb_loader import OCBDatasetLoader

PROCESSED_FILES = {"OCB101": "data.pt", "OCB301": "data_301.pt"}
RAW_FILES = {
    "OCB101": ["ckt_bench_101.pkl", "perform101.csv"],
    "OCB301": ["ckt_bench_301.pkl.zip", "perform301.csv"],
}
DATASET_CLASSES = {"OCB101": OCB101Dataset, "OCB301": OCB301Dataset}
SYNTHETIC_NUM_GRAPHS = 20


def _write_dummy_processed(
    base_dir: Path, dataset_name: str, num_graphs: int = SYNTHETIC_NUM_GRAPHS
):
    """Create a synthetic processed dataset for tests.

    Parameters
    ----------
    base_dir : Path
        Root directory under which the dataset folders will be created.
    dataset_name : str
        Name of the dataset (e.g., ``\"OCB101\"`` or ``\"OCB301\"``).
    num_graphs : int, optional
        Number of synthetic graphs to generate, by default ``SYNTHETIC_NUM_GRAPHS``.
    """
    processed_dir = base_dir / dataset_name / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / PROCESSED_FILES[dataset_name]
    data_list = []
    for idx in range(num_graphs):
        num_nodes = idx % 3 + 1
        x = torch.full(
            (num_nodes, NUM_NODE_TYPES + 1), fill_value=float(idx)
        )
        if num_nodes > 1:
            edge_index = torch.vstack(
                [
                    torch.arange(0, num_nodes - 1, dtype=torch.long),
                    torch.arange(1, num_nodes, dtype=torch.long),
                ]
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([float(idx)], dtype=torch.float),
            vid=torch.arange(num_nodes, dtype=torch.long),
            valid=torch.tensor([1], dtype=torch.long),
        )
        data_list.append(data)
    raw_dir = base_dir / dataset_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for raw_file in RAW_FILES[dataset_name]:
        (raw_dir / raw_file).touch()

    data, slices = OCBDataset.collate(data_list)
    torch.save((data, slices), processed_path)


@pytest.fixture(params=["OCB101", "OCB301"])
def ocb_dataset(tmp_path, request):
    """Return a synthetic dataset and its loader directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory in which to place the fake dataset.
    request : FixtureRequest
        Pytest request object containing the parametrized dataset name.

    Returns
    -------
    tuple
        ``(dataset_name, dataset, dataset_dir)`` for the synthetic dataset.
    """
    dataset_name: str = request.param
    base_dir = tmp_path / "graph" / "circuits"
    _write_dummy_processed(base_dir, dataset_name)
    cfg = OmegaConf.create(
        {
            "data_domain": "graph",
            "data_type": "circuits",
            "data_name": dataset_name,
            "data_dir": str(base_dir),
        }
    )
    loader = OCBDatasetLoader(parameters=cfg)
    dataset, dataset_dir = loader.load()
    return dataset_name, dataset, Path(dataset_dir)


def test_ocb_loader_dispatches_correct_dataset(ocb_dataset):
    """Ensure the loader instantiates the requested dataset.

    Parameters
    ----------
    ocb_dataset : tuple
        Fixture returning ``(dataset_name, dataset, dataset_dir)``.
    """
    dataset_name, dataset, dataset_dir = ocb_dataset
    expected_cls = DATASET_CLASSES[dataset_name]
    assert isinstance(dataset, expected_cls)
    assert dataset_dir.name == dataset_name
    assert len(dataset) == SYNTHETIC_NUM_GRAPHS
    assert dataset[0].x.shape[1] == NUM_NODE_TYPES + 1


def test_ocb_dataset_statistics_match_manual_computation(ocb_dataset):
    """Validate get_target_statistics output for synthetic data.

    Parameters
    ----------
    ocb_dataset : tuple
        Fixture returning ``(dataset_name, dataset, dataset_dir)``.
    """
    _, dataset, _ = ocb_dataset
    stats = dataset.get_target_statistics()
    assert set(stats) == {"mean", "std", "min", "max"}
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    assert pytest.approx(float(all_targets.mean())) == stats["mean"]
    assert pytest.approx(float(all_targets.std())) == stats["std"]
    assert stats["min"] == pytest.approx(float(all_targets.min()))
    assert stats["max"] == pytest.approx(float(all_targets.max()))


def test_ocb_loader_requires_data_name(tmp_path: Path):
    """Missing data_name should raise a ValueError.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for placing fake data inputs.
    """
    cfg = OmegaConf.create(
        {
            "data_domain": "graph",
            "data_type": "circuits",
            "data_dir": str(tmp_path),
        }
    )
    loader = OCBDatasetLoader(parameters=cfg)
    with pytest.raises(ValueError):
        loader.load_dataset()


def test_ocb_loader_rejects_unknown_dataset(tmp_path: Path):
    """Unknown datasets surface a helpful error.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for placing fake data inputs.
    """
    cfg = OmegaConf.create(
        {
            "data_domain": "graph",
            "data_type": "circuits",
            "data_dir": str(tmp_path),
            "data_name": "OCB999",
        }
    )
    loader = OCBDatasetLoader(parameters=cfg)
    with pytest.raises(ValueError):
        loader.load_dataset()
