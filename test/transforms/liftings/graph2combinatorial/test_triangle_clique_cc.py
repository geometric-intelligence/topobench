"""Tests for explicit triangle-clique combinatorial lifting."""

import torch
from torch_geometric.data import Data

from topobench.transforms.liftings.graph2combinatorial.triangle_clique_cc import (
    GraphTriangleCliqueCCLifting,
)


def _undirected_edge_index(edges):
    directed = []
    for source, target in edges:
        directed.append((source, target))
        directed.append((target, source))
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def test_triangle_clique_lifting_adds_one_rank2_cell_per_graph_triangle():
    """Every graph triangle becomes an explicit rank-2 cell."""
    data = Data(
        x=torch.eye(4),
        edge_index=_undirected_edge_index([(0, 1), (1, 2), (0, 2), (2, 3)]),
    )
    lifting = GraphTriangleCliqueCCLifting(
        complex_dim=2,
        neighborhoods=[
            "up_incidence-0",
            "up_incidence-1",
            "down_incidence-2",
            "up_adjacency-2",
        ],
    )

    lifted = lifting(data)

    assert lifted.x_0.shape == (4, 4)
    assert lifted.x_1.shape == (4, 4)
    assert lifted.x_2.shape == (1, 4)
    assert lifted.incidence_2.shape == (4, 1)
    assert lifted.incidence_2.coalesce().values().abs().sum() == 3
    assert lifted["up_incidence-0"].shape == (4, 4)
    assert lifted["up_incidence-1"].shape == (1, 4)
    assert lifted["down_incidence-2"].shape == (4, 1)
    assert lifted["up_adjacency-2"].shape == (1, 1)
    assert lifted.structure_0.shape == (4, 4)
    assert lifted.structure_1.shape == (4, 4)
    assert lifted.structure_2.shape == (1, 4)
    torch.testing.assert_close(
        lifted.structure_2[:, 0],
        torch.log1p(torch.tensor([3.0])),
    )


def test_triangle_clique_lifting_handles_graphs_without_triangles():
    """Triangle-free graphs keep an empty but valid rank-2 representation."""
    data = Data(
        x=torch.ones(3, 2),
        edge_index=_undirected_edge_index([(0, 1), (1, 2)]),
    )
    lifting = GraphTriangleCliqueCCLifting(
        complex_dim=2,
        neighborhoods=["up_incidence-1", "down_incidence-2"],
    )

    lifted = lifting(data)

    assert lifted.x_2.shape == (0, 2)
    assert lifted.incidence_2.shape == (2, 0)
    assert lifted.structure_2.shape == (0, 4)
