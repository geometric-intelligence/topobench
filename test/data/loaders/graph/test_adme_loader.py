"""Tests for ``ADMEDatasetLoader`` (no network).

The real ``load_dataset`` downloads from TDC over the network, which we
don't want to hit in CI. These tests focus on the parts of the loader
that we can exercise locally: construction, ``get_data_dir``, the
classification/regression bookkeeping, and the unknown-name branch.

The ``test_load_dataset_*`` tests mock out ``tdc.single_pred.ADME`` and
``ogb.utils.smiles2graph`` so no network requests are made.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from topobench.data.loaders.graph.adme_datasets import ADMEDatasetLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path, data_name="BBB_Martins"):
    return OmegaConf.create(
        {
            "data_dir": str(tmp_path),
            "data_name": data_name,
            "data_type": "ADME",
        }
    )


def _fake_smiles2graph(_smiles):
    """Minimal graph dict returned by ogb's smiles2graph."""
    return {
        "num_nodes": 3,
        "node_feat": [[1, 0, 0, 0, 0, 0, 0, 0, 0]] * 3,
        "edge_index": [[0, 1], [1, 0]],
        "edge_feat": [[1, 0, 0], [1, 0, 0]],
    }


def _fake_tdc_adme(name, path):
    """Return a mock ADME object whose get_split() yields tiny DataFrames."""
    df_train = pd.DataFrame({"Drug": ["C", "CC"], "Y": [1, 0]})
    df_valid = pd.DataFrame({"Drug": ["CCC"], "Y": [1]})
    df_test = pd.DataFrame({"Drug": ["CCCC"], "Y": [0]})
    mock = MagicMock()
    mock.get_split.return_value = {
        "train": df_train,
        "valid": df_valid,
        "test": df_test,
    }
    return mock


# ---------------------------------------------------------------------------
# Basic unit tests (no mocking needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def loader(tmp_path):
    return ADMEDatasetLoader(_make_cfg(tmp_path))


def test_repr(loader):
    assert "ADMEDatasetLoader" in repr(loader)


def test_get_data_dir_combines_root_and_name(loader, tmp_path):
    assert loader.get_data_dir() == os.path.join(str(tmp_path), "BBB_Martins")


def test_load_dataset_rejects_unknown_name(tmp_path):
    cfg = _make_cfg(tmp_path, data_name="TotallyMadeUp")
    with pytest.raises(ValueError, match="Unknown ADME dataset"):
        ADMEDatasetLoader(cfg).load_dataset()


# ---------------------------------------------------------------------------
# Mocked load_dataset tests
# ---------------------------------------------------------------------------

@patch("topobench.data.loaders.graph.adme_datasets.smiles2graph", side_effect=_fake_smiles2graph)
@patch("topobench.data.loaders.graph.adme_datasets.ADME", side_effect=_fake_tdc_adme)
def test_load_dataset_classification(mock_adme, mock_s2g, tmp_path):
    """load_dataset builds int labels for a classification dataset."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "BBB_Martins"))
    dataset = loader.load_dataset()

    assert len(dataset) == 4  # 2 train + 1 valid + 1 test
    # Classification labels should be long tensors
    assert dataset[0].y.dtype == torch.long
    assert hasattr(dataset, "split_idx")
    assert "train" in dataset.split_idx
    assert len(dataset.split_idx["train"]) == 2


@patch("topobench.data.loaders.graph.adme_datasets.smiles2graph", side_effect=_fake_smiles2graph)
@patch("topobench.data.loaders.graph.adme_datasets.ADME", side_effect=_fake_tdc_adme)
def test_load_dataset_regression(mock_adme, mock_s2g, tmp_path):
    """load_dataset builds float labels for a regression dataset."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "Caco2_Wang"))
    dataset = loader.load_dataset()

    assert len(dataset) == 4
    assert dataset[0].y.dtype == torch.float
    assert dataset[0].y.shape == (1,)


@patch("topobench.data.loaders.graph.adme_datasets.smiles2graph", side_effect=_fake_smiles2graph)
@patch("topobench.data.loaders.graph.adme_datasets.ADME", side_effect=_fake_tdc_adme)
def test_load_dataset_node_and_edge_features(mock_adme, mock_s2g, tmp_path):
    """Each graph has 9-dim node features and 3-dim edge features."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "BBB_Martins"))
    dataset = loader.load_dataset()

    graph = dataset[0]
    assert graph.x.shape[1] == 9
    assert graph.edge_attr.shape[1] == 3


@patch("topobench.data.loaders.graph.adme_datasets.smiles2graph", side_effect=_fake_smiles2graph)
@patch("topobench.data.loaders.graph.adme_datasets.ADME", side_effect=_fake_tdc_adme)
def test_load_dataset_split_indices_partition_dataset(mock_adme, mock_s2g, tmp_path):
    """train/valid/test split indices together cover the whole dataset."""
    loader = ADMEDatasetLoader(_make_cfg(tmp_path, "BBB_Martins"))
    dataset = loader.load_dataset()

    idx = dataset.split_idx
    all_indices = torch.cat([idx["train"], idx["valid"], idx["test"]])
    assert len(all_indices) == len(dataset)
