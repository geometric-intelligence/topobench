"""Tests for the K-hop Feature Encoding (KHopFE) transform."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.khop_feature_encodings import (
    KHopFE,
)


@pytest.fixture
def small_graph():
    """A small connected graph with random features."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    x = torch.randn(4, 3)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


class TestKHopFE:
    """Tests for ``KHopFE``."""

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            KHopFE(max_hop=3, method="dummy")

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            KHopFE(max_hop=3, aggregation="weird")

    def test_max_hop_is_off_by_one(self):
        """The constructor stores ``max_hop - 1`` (hop 0 is the features)."""
        t = KHopFE(max_hop=3)
        assert t.max_hop == 2

    def test_forward_requires_x(self):
        t = KHopFE(max_hop=2)
        data = Data(
            edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=2
        )
        with pytest.raises(ValueError, match="KHopFE requires node features"):
            t(data)

    @pytest.mark.parametrize("method", ["dense", "sparse"])
    def test_forward_concatenates(self, small_graph, method):
        n_feat = small_graph.x.shape[1]
        t = KHopFE(max_hop=3, method=method, concat_to_x=True)
        out = t(small_graph)
        assert out.x.shape[0] == small_graph.num_nodes
        assert out.x.shape[1] >= n_feat

    def test_forward_stores_separate_attribute(self, small_graph):
        t = KHopFE(max_hop=2, concat_to_x=False)
        out = t(small_graph)
        assert hasattr(out, "KHopFE")
        assert out.KHopFE.shape[0] == small_graph.num_nodes

    def test_debug_mode(self, small_graph, capsys):
        """Test KHopFE with debug=True."""
        t = KHopFE(max_hop=2, debug=True)
        out = t(small_graph)
        captured = capsys.readouterr()
        assert "KHopFE Debug Report" in captured.out
        assert out.x.shape[0] == small_graph.num_nodes
