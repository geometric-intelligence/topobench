"""Unit tests for HICDataset."""

import os
from pathlib import Path

import pytest
import torch

from topobench.data.datasets.hic_dataset import HICDataset


def _write_hic_raw(root: Path, name: str, content: str) -> Path:
    """Create a synthetic HIC raw file."""
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{name}.txt"
    raw_path.write_text(content)
    return raw_path


@pytest.fixture
def dataset_root(tmp_path) -> Path:
    """Temporary root directory for HIC datasets."""
    return tmp_path / "hic_datasets"


def test_raw_and_processed_dirs_and_filenames(dataset_root):
    name = "RHG_3"
    ds = HICDataset.__new__(HICDataset)
    ds.root = str(dataset_root)
    ds.name = name

    assert ds.raw_dir == os.path.join(str(dataset_root), name, "raw")
    assert ds.processed_dir == os.path.join(str(dataset_root), name, "processed")
    assert ds.raw_file_names == [f"{name}.txt"]
    assert ds.processed_file_names == ["data.pt"]


def test_parse_vertex_labels_single_and_multi():
    line = "1 2/3 4/5/6 7"
    parsed = HICDataset._parse_vertex_labels(line)

    assert parsed == [1, [2, 3], [4, 5, 6], 7]


def test_clique_expand_edges_basic():
    hyperedges = [
        [0, 1, 2],
        [2, 3],
        [4],
    ]
    num_v = 5

    edge_index = HICDataset._clique_expand_edges(hyperedges, num_v)

    assert edge_index.shape[0] == 2
    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_v

    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False))
    expected = {
        (0, 1), (1, 0),
        (0, 2), (2, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
    }
    assert expected.issubset(pairs)


def test_build_incidence_basic():
    hyperedges = [
        [0, 1],
        [1, 2, 3],
    ]
    he_index, num_h = HICDataset._build_incidence(hyperedges, 4)

    assert num_h == 2
    assert he_index.shape[0] == 2

    nodes = he_index[0].tolist()
    hed_ids = he_index[1].tolist()

    assert 0 in hed_ids
    assert 1 in hed_ids
    assert all(v in nodes for v in [0, 1, 2, 3])


def test_download_invalid_dataset_raises(dataset_root):
    ds = HICDataset.__new__(HICDataset)
    ds.root = str(dataset_root)
    ds.name = "UNKNOWN"

    with pytest.raises(ValueError):
        ds.download()


def test_download_builds_correct_url(monkeypatch, dataset_root):
    calls = {}

    def fake_download_url(url, folder):
        calls["url"] = url
        calls["folder"] = folder

    monkeypatch.setattr(
        "topobench.data.datasets.hic_dataset.download_url",
        fake_download_url,
    )

    ds = HICDataset.__new__(HICDataset)
    ds.root = str(dataset_root)
    ds.name = "RHG_3"

    ds.download()

    assert "RHG/RHG_3.txt" in calls["url"]
    assert calls["folder"].endswith("RHG_3/raw")


def test_read_all_graphs_and_process_hypergraph(dataset_root):
    name = "RHG_3"
    content = """2
3 2 0
1 2 3
0 1
1 2
2 1 1
4 5
0 1
"""
    raw_path = _write_hic_raw(dataset_root, name, content)

    ds = HICDataset(root=str(dataset_root), name=name)
    assert len(ds) == 2

    entries, v_universe, g_universe = ds._read_all_graphs(str(raw_path))

    assert len(entries) == 2
    assert v_universe == {1, 2, 3, 4, 5}
    assert g_universe == {0, 1}

    data0 = ds[0]
    assert data0.num_nodes == 3
    assert data0.num_hyperedges == 2
    assert data0.x.shape[1] == len(v_universe)
    assert data0.y.shape == torch.Size([1])


def test_process_simple_graph(dataset_root):
    name = "MUTAG"
    content = """1
4 2 1
10 20 30 40
0 1
1 2 3
"""
    _write_hic_raw(dataset_root, name, content)

    ds = HICDataset(root=str(dataset_root), name=name)
    data = ds[0]

    assert data.num_nodes == 4
    assert data.y.shape == torch.Size([])


def test_process_multi_label_graph_targets(dataset_root):
    name = "IMDB_dir_genre_m"
    content = """1
3 1 0 2
1/2 3/4 5/6
0 1 2
"""
    _write_hic_raw(dataset_root, name, content)

    ds = HICDataset(root=str(dataset_root), name=name)
    data = ds[0]

    assert data.y.tolist().count(1) == 2


def test_use_degree_as_tag_replaces_vertex_labels(dataset_root):
    name = "RHG_10"
    content = """1
3 3 0
10 20 30
0 1
0 2
1 2
"""
    _write_hic_raw(dataset_root, name, content)

    ds = HICDataset(root=str(dataset_root), name=name, use_degree_as_tag=True)
    data = ds[0]

    assert torch.allclose(data.x[0], data.x[2])
    assert data.x.shape[1] >= 2


def test_len_and_getitem_multiple_graphs(dataset_root):
    name = "RHG_table"
    content = """2
2 1 0
1 2
0 1
3 1 1
3 4 5
0 1 2
"""
    _write_hic_raw(dataset_root, name, content)

    ds = HICDataset(root=str(dataset_root), name=name)

    assert len(ds) == 2
    assert ds[0].num_nodes == 2
    assert ds[1].num_nodes == 3
