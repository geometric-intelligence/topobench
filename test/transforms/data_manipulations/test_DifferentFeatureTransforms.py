"""Tests for DifferentGausFeatures, DifferentGausFeaturesSANN, DifferentZeroFeaturesSANN."""

import pytest
import torch
import torch_geometric
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.different_gaus_features import (
    DifferentGausFeatures,
)
from topobench.transforms.data_manipulations.different_gaus_features_sann import (
    DifferentGausFeaturesSANN,
)
from topobench.transforms.data_manipulations.different_zero_features_sann import (
    DifferentZeroFeaturesSANN,
)


def _make_complex(n_nodes=4, n_edges=3, feat_dim=2):
    """Return a Data object with incidence_1 and x_0."""
    inc = torch.zeros(n_nodes, n_edges).to_sparse()
    return Data(incidence_1=inc, x_0=torch.randn(n_nodes, feat_dim))


def _make_sann_complex(n_nodes=4, n_edges=3):
    """Return a Data object with incidence_0 and incidence_1 for SANN tests."""
    inc0 = torch.zeros(n_nodes, n_edges).to_sparse()
    inc1 = torch.zeros(n_edges, 2).to_sparse()
    return Data(incidence_0=inc0, incidence_1=inc1)


# ---------------------------------------------------------------------------
# DifferentGausFeatures
# ---------------------------------------------------------------------------

class TestDifferentGausFeatures:
    """Tests for DifferentGausFeatures transform."""

    def _make_transform(self, dimensions=(1,), feat_size=4):
        return DifferentGausFeatures(
            mean=0.0, std=1.0, num_features=feat_size, dimensions=dimensions
        )

    def test_repr(self):
        """__repr__ includes class name and parameters."""
        t = self._make_transform()
        r = repr(t)
        assert "DifferentGausFeatures" in r
        assert "mean" in r

    def test_forward_single_dimension(self):
        """forward() sets x_1 with Gaussian features."""
        t = self._make_transform(dimensions=[1], feat_size=5)
        data = _make_complex(n_nodes=4, n_edges=3)

        out = t(data)

        assert hasattr(out, "x_1")
        assert out.x_1.shape == (3, 5)

    def test_forward_multiple_dimensions(self):
        """forward() sets features for each requested dimension."""
        inc0 = torch.zeros(4, 3).to_sparse()
        inc1 = torch.zeros(3, 2).to_sparse()
        data = Data(incidence_0=inc0, incidence_1=inc1)
        t = self._make_transform(dimensions=[0, 1], feat_size=3)

        out = t(data)

        assert hasattr(out, "x_0")
        assert hasattr(out, "x_1")
        assert out.x_0.shape == (3, 3)  # incidence_0 has 3 cols
        assert out.x_1.shape == (2, 3)  # incidence_1 has 2 cols

    def test_call_delegates_to_forward(self):
        """__call__ is equivalent to forward."""
        t = self._make_transform()
        data = _make_complex()
        out = t(data)
        assert hasattr(out, "x_1")

    def test_init_stores_params(self):
        """__init__ stores mean, std, feature_vector_size, dimensions."""
        t = DifferentGausFeatures(mean=1.0, std=2.0, num_features=8, dimensions=[1, 2])
        assert t.mean == 1.0
        assert t.std == 2.0
        assert t.feature_vector_size == 8

    def test_default_feature_size_minus_one(self):
        """Without num_features kwarg, feature_vector_size defaults to -1."""
        t = DifferentGausFeatures(mean=0.0, std=1.0, dimensions=[1])
        assert t.feature_vector_size == -1


# ---------------------------------------------------------------------------
# DifferentGausFeaturesSANN
# ---------------------------------------------------------------------------

class TestDifferentGausFeaturesSANN:
    """Tests for DifferentGausFeaturesSANN transform."""

    def _make_transform(self, dimensions=2, max_hop=2, feat_size=4):
        return DifferentGausFeaturesSANN(
            mean=0.0, std=1.0, num_features=feat_size,
            dimensions=dimensions, max_hop=max_hop,
        )

    def test_repr(self):
        """__repr__ includes class name."""
        r = repr(self._make_transform())
        assert "DifferentGausFeaturesSANN" in r

    def test_forward_sets_hop_features(self):
        """forward() sets x{dim}_{t} for each dim and hop."""
        t = self._make_transform(dimensions=2, max_hop=3, feat_size=5)
        data = _make_sann_complex(n_nodes=4, n_edges=3)

        out = t(data)

        for dim in range(2):
            for hop in range(3):
                key = f"x{dim}_{hop}"
                assert hasattr(out, key), f"Missing {key}"

    def test_forward_feature_shapes(self):
        """feature tensors have shape [n_cells, feat_size]."""
        t = self._make_transform(dimensions=1, max_hop=2, feat_size=4)
        inc0 = torch.zeros(5, 4).to_sparse()
        data = Data(incidence_0=inc0)

        out = t(data)

        assert out.x0_0.shape == (4, 4)  # incidence_0 has 4 cols
        assert out.x0_1.shape == (4, 4)

    def test_init_stores_params(self):
        """__init__ stores all relevant attributes."""
        t = self._make_transform(dimensions=3, max_hop=4, feat_size=7)
        assert t.max_hop == 4
        assert t.dimensions == 3
        assert t.feature_vector_size == 7


# ---------------------------------------------------------------------------
# DifferentZeroFeaturesSANN
# ---------------------------------------------------------------------------

class TestDifferentZeroFeaturesSANN:
    """Tests for DifferentZeroFeaturesSANN transform."""

    def _make_transform(self, dimensions=2, max_hop=2, feat_size=4):
        return DifferentZeroFeaturesSANN(
            mean=0.0, std=1.0, num_features=feat_size,
            dimensions=dimensions, max_hop=max_hop,
        )

    def test_repr(self):
        r = repr(self._make_transform())
        assert "DifferentZeroFeaturesSANN" in r

    def test_forward_produces_zeros(self):
        """forward() sets x{dim}_{t} tensors that are all zero."""
        t = self._make_transform(dimensions=1, max_hop=2, feat_size=3)
        inc0 = torch.zeros(4, 5).to_sparse()
        data = Data(incidence_0=inc0)

        out = t(data)

        assert out.x0_0.shape == (5, 3)
        assert torch.all(out.x0_0 == 0.0)
        assert torch.all(out.x0_1 == 0.0)

    def test_forward_multiple_hops(self):
        """All requested hop-features are present."""
        t = self._make_transform(dimensions=2, max_hop=3, feat_size=2)
        data = _make_sann_complex()

        out = t(data)

        for dim in range(2):
            for hop in range(3):
                key = f"x{dim}_{hop}"
                assert hasattr(out, key), f"Missing {key}"

    def test_init_stores_params(self):
        t = self._make_transform(dimensions=2, max_hop=3, feat_size=6)
        assert t.max_hop == 3
        assert t.feature_vector_size == 6
