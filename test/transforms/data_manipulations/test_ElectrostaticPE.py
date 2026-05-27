"""Tests for the Electrostatic Positional Encoding (ElectrostaticPE)."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.electrostatic_encodings import (
    ElectrostaticPE,
)


@pytest.fixture
def small_graph():
    """Small connected graph with random features."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    x = torch.randn(4, 2)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


class TestElectrostaticPE:
    """Tests for ``ElectrostaticPE``."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ElectrostaticPE(method="oracle")

    def test_pe_dim_is_seven(self):
        t = ElectrostaticPE()
        assert t.pe_dim == 7

    @pytest.mark.parametrize("method", ["numpy", "gpu"])
    def test_forward_appends_seven_dims(self, small_graph, method):
        n_feat = small_graph.x.shape[1]
        t = ElectrostaticPE(method=method, concat_to_x=True)
        out = t(small_graph)
        assert out.x.shape == (small_graph.num_nodes, n_feat + 7)

    def test_forward_stores_separate_attribute(self, small_graph):
        t = ElectrostaticPE(concat_to_x=False)
        out = t(small_graph)
        assert hasattr(out, "ElectrostaticPE")
        assert out.ElectrostaticPE.shape == (small_graph.num_nodes, 7)

    def test_debug_mode(self, small_graph, capsys):
        """Test ElectrostaticPE with debug=True."""
        t = ElectrostaticPE(debug=True)
        out = t(small_graph)
        captured = capsys.readouterr()
        assert "ElectrostaticPE Debug Report" in captured.out
        assert out.x.shape == (small_graph.num_nodes, small_graph.x.shape[1] + 7)
