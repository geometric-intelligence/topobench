"""Tests for the TRAWLTransform data manipulation."""

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from test._utils.trawl import complex_data, prepare
from topobench.transforms.data_manipulations.trawl import (
    TRAWLTransform,
    hopse_encodings,
)


def test_hasse_and_overlap_are_different():
    hasse = prepare(graph="hasse")
    overlap = prepare(graph="cell_overlap")
    assert len(hasse.trawl_edges) == 12
    assert hasse.trawl_relations.item() == 2
    assert overlap.trawl_relations.item() == 1
    assert torch.all(overlap.trawl_edges[:, :2] >= 3)
    assert hasse.trawl_counts.tolist() == [[3, 2, 1]]


def test_adjacency_and_multihop():
    graph = prepare(
        neighborhoods=[
            "up_adjacency-0",
            "down_adjacency-1",
            "2-up_incidence-0",
        ]
    )
    edges = graph.trawl_edges
    assert {(0, 1), (1, 0), (1, 2), (2, 1)} == set(
        map(tuple, edges[edges[:, 2] == 0, :2].tolist())
    )
    assert (3, 4) in set(map(tuple, edges[edges[:, 2] == 1, :2].tolist()))
    assert len(edges[edges[:, 2] == 2]) == 6


def test_default_neighborhoods_follow_max_rank():
    assert TRAWLTransform(max_rank=2).neighborhoods == [
        "up_incidence-0",
        "up_incidence-1",
    ]
    assert TRAWLTransform(max_rank=1).neighborhoods == ["up_incidence-0"]
    assert TRAWLTransform(max_rank=0).neighborhoods == ["up_adjacency-0"]


def test_hypergraph_and_plain_graph():
    hyper = Data(
        x=torch.randn(3, 2),
        incidence_hyperedges=torch.tensor(
            [[1.0, 0], [1, 1], [0, 1]]
        ).to_sparse(),
        num_nodes=3,
        y=torch.tensor([0]),
    )
    adapted = prepare(hyper, max_rank=1)
    assert adapted.trawl_counts.tolist() == [[3, 2]]
    plain = Data(
        x=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=3,
        y=torch.tensor([0]),
    )
    adapted = prepare(plain, max_rank=0)
    assert adapted.trawl_edges.shape == (2, 3)


def test_spectral_and_color_refinement():
    matrix = sp.csr_matrix([[0.0, 1], [1, 0]])
    pe = hopse_encodings(
        matrix,
        LapPE={"max_pe_dim": 4},
        RWSE={"max_pe_dim": 2},
        ElectrostaticPE={},
        HKdiagSE={"kernel_param_HKdiagSE": [1, 2]},
    )
    # 2 local + 4 LapPE (zero-padded) + 2 RWSE + 7 electrostatic + 1 heat.
    assert pe.shape == (2, 16)
    assert np.isfinite(pe).all()
    graph = complex_data()
    a, b = (
        prepare(graph.clone()),
        prepare(graph.clone(), color_refinement=True),
    )
    assert torch.equal(a.trawl_edges, b.trawl_edges)
    assert b.trawl_pe.shape[1] == a.trawl_pe.shape[1] * 6


def test_hopse_encodings_zero_inactive_states():
    hopse = {
        "local": True,
        "LapPE": {"max_pe_dim": 2},
        "RWSE": {"max_pe_dim": 3},
        "ElectrostaticPE": {},
        "HKdiagSE": {"kernel_param_HKdiagSE": [1, 3]},
    }
    data = TRAWLTransform(encodings=hopse)(complex_data())
    # Per relation: 2 local + 2 LapPE + 3 RWSE + 7 electrostatic + 2 heat.
    assert data.trawl_pe.shape == (len(data.trawl_colors), 2 * 16)
    assert torch.isfinite(data.trawl_pe).all()
    # Rank-2 cells are absent from the 0-1 relation and get zeros there.
    assert not data.trawl_pe[-1, :16].any() and data.trawl_pe[-1, 16:].any()


def test_encodings_on_large_sparse_graph():
    n = 10000
    rows = np.arange(n)
    matrix = sp.coo_matrix(
        (
            np.ones(2 * n),
            (
                np.concatenate([rows, rows]),
                np.concatenate([(rows + 1) % n, (rows - 1) % n]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    encoding = hopse_encodings(matrix, RWSE={"max_pe_dim": 2})
    assert encoding.shape == (n, 4)
    assert np.isfinite(encoding).all()


def test_finalize_dataset_aligns_absent_rank_widths():
    first = prepare(complex_data(empty=True))
    second = complex_data()
    second.x_2 = torch.randn(1, 4)
    second = prepare(second)
    TRAWLTransform().finalize_dataset([first, second])
    assert first.trawl_signal_2.shape == (0, 4)
    assert first.x_2.shape == (0, 4)
