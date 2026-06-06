"""Tests for the Heat Kernel Diagonal Structural Encoding (HKdiagSE)."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.hkdiag_encodings import HKdiagSE


@pytest.fixture
def small_graph():
    """A tiny cycle graph with features."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    )
    x = torch.randn(4, 3)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


class TestHKdiagSE:
    """Tests for ``HKdiagSE``."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            HKdiagSE(kernel_param_HKdiagSE=(1, 4), method="other")

    def test_pe_dim_from_scalar(self):
        t = HKdiagSE(kernel_param_HKdiagSE=4)
        assert t.pe_dim == 4

    @pytest.mark.parametrize("method", ["exact", "fast"])
    def test_forward_concatenates(self, small_graph, method):
        n_feat = small_graph.x.shape[1]
        t = HKdiagSE(
            kernel_param_HKdiagSE=(1, 3),
            method=method,
            concat_to_x=True,
        )
        out = t(small_graph)
        assert out.x.shape[0] == small_graph.num_nodes
        assert out.x.shape[1] >= n_feat

    def test_forward_stores_separate_attribute(self, small_graph):
        t = HKdiagSE(kernel_param_HKdiagSE=(1, 3), concat_to_x=False)
        out = t(small_graph)
        assert hasattr(out, "HKdiagSE")
        assert out.HKdiagSE.shape[0] == small_graph.num_nodes

    def test_debug_mode(self, small_graph, capsys):
        """Test HKdiagSE with debug=True."""
        t = HKdiagSE(kernel_param_HKdiagSE=(1, 3), debug=True)
        out = t(small_graph)
        captured = capsys.readouterr()
        assert "HKdiagSE Debug Report" in captured.out
        assert out.x.shape[0] == small_graph.num_nodes
