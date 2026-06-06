"""Tests for PrecomputeKHopFeatures transform."""

import pytest
import torch
import torch_geometric
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.precompute_khop_features import (
    PrecomputeKHopFeatures,
)


def _make_triangle_complex(feat_dim: int = 1):
    """Return a minimal simplicial complex: 3 nodes + 3 edges (triangle).

    incidence_1 shape: [3 nodes, 3 edges]
    x_0: node features  [3, feat_dim]
    x_1: edge features  [3, feat_dim]
    """
    inc1 = torch.tensor(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]
    ).to_sparse()
    return Data(
        incidence_1=inc1,
        x_0=torch.ones(3, feat_dim),
        x_1=torch.ones(3, feat_dim),
    )


def _make_dim2_complex():
    """Return a complex with dim=2: triangle + one 2-cell.

    incidence_1: [3 nodes, 3 edges]
    incidence_2: [3 edges, 1 face]
    """
    inc1 = torch.tensor(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]
    ).to_sparse()
    inc2 = torch.tensor([[1.0], [-1.0], [1.0]]).to_sparse()
    return Data(
        incidence_1=inc1,
        incidence_2=inc2,
        x_0=torch.ones(3, 1),
        x_1=torch.ones(3, 1),
    )


class TestPrecomputeKHopFeatures:
    """Unit tests for PrecomputeKHopFeatures."""

    def test_repr(self):
        """__repr__ includes class name and key parameters."""
        t = PrecomputeKHopFeatures(max_hop=2, complex_dim=1, use_initial_features=True)
        r = repr(t)
        assert "PrecomputeKHopFeatures" in r
        assert "complex_dim=1" in r

    def test_max_hop_stored_minus_one(self):
        """Internal max_hop is decremented by 1 (0-hop = identity)."""
        t = PrecomputeKHopFeatures(max_hop=3, complex_dim=1, use_initial_features=True)
        assert t.max_hop == 2

    def test_forward_dim1_use_initial_features(self):
        """forward() produces x0_0 and x0_1 for complex_dim=1."""
        data = _make_triangle_complex(feat_dim=2)
        t = PrecomputeKHopFeatures(max_hop=2, complex_dim=1, use_initial_features=True)

        out = t(data)

        assert hasattr(out, "x0_0")
        assert hasattr(out, "x0_1")
        assert out["x0_0"].shape[0] == 3  # 3 nodes
        assert out["x0_1"].shape[0] == 3  # 3 edges

    def test_forward_dim1_no_initial_features(self):
        """forward() with use_initial_features=False uses ones as 0-hop."""
        data = _make_triangle_complex(feat_dim=2)
        t = PrecomputeKHopFeatures(max_hop=2, complex_dim=1, use_initial_features=False)

        out = t(data)

        # 0-hop should be all-ones regardless of input features
        assert torch.all(out["x0_0"] == 1.0)

    def test_forward_preserves_original_fields(self):
        """forward() keeps the original data fields (incidence, x_0, x_1)."""
        data = _make_triangle_complex()
        t = PrecomputeKHopFeatures(max_hop=2, complex_dim=1, use_initial_features=True)

        out = t(data)

        assert hasattr(out, "x_0")
        assert hasattr(out, "x_1")
        assert hasattr(out, "incidence_1")

    def test_forward_dim2(self):
        """forward() handles complex_dim=2 (nodes, edges, faces)."""
        data = _make_dim2_complex()
        t = PrecomputeKHopFeatures(max_hop=2, complex_dim=2, use_initial_features=True)

        out = t(data)

        assert hasattr(out, "x0_0")
        assert hasattr(out, "x1_0")

    def test_hop1_produces_only_zeroth_hop(self):
        """max_hop=1 means T=0, so only x{k}_0 entries are generated."""
        data = _make_triangle_complex()
        t = PrecomputeKHopFeatures(max_hop=1, complex_dim=1, use_initial_features=True)

        out = t(data)

        # x0_0 must be set (0-hop = initial features)
        assert hasattr(out, "x0_0")
        assert out["x0_0"].shape[0] == 3
        # With T=0 the loop over t=1..T never runs, so no x0_1 attribute
        assert not hasattr(out, "x0_1")

    def test_output_is_data_object(self):
        """forward() returns a torch_geometric Data object."""
        data = _make_triangle_complex()
        t = PrecomputeKHopFeatures(max_hop=2, complex_dim=1, use_initial_features=True)

        out = t(data)

        assert isinstance(out, torch_geometric.data.Data)
