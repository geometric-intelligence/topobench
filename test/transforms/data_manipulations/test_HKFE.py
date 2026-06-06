"""Tests for the Heat Kernel Feature Encoding (HKFE) transform."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.hk_feature_encodings import HKFE


@pytest.fixture
def small_graph():
    """Return a tiny line-graph Data object with ``data.x``.

    Returns
    -------
    torch_geometric.data.Data
        A small 4-node line graph with 3 random features per node.
    """
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    x = torch.randn(4, 3)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


class TestHKFE:
    """Tests for ``HKFE``."""

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            HKFE(kernel_param_HKFE=(1, 4), aggregation="bogus")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            HKFE(kernel_param_HKFE=(1, 4), method="oracle")

    def test_fe_dim_from_tuple(self):
        t = HKFE(kernel_param_HKFE=(1, 5))
        assert t.fe_dim == 4

    def test_fe_dim_from_scalar(self):
        t = HKFE(kernel_param_HKFE=4)
        assert t.fe_dim == 4

    def test_forward_requires_x(self):
        t = HKFE(kernel_param_HKFE=(1, 3))
        with pytest.raises(ValueError, match="HKFE requires node features"):
            t(Data(edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=2))

    def test_forward_returns_concatenated_features(self, small_graph):
        n_feat = small_graph.x.shape[1]
        t = HKFE(kernel_param_HKFE=(1, 3), concat_to_x=True)
        out = t(small_graph)
        assert out.x.shape == (small_graph.num_nodes, n_feat + t.fe_dim)

    def test_forward_stores_separate_attribute(self, small_graph):
        t = HKFE(kernel_param_HKFE=(1, 3), concat_to_x=False)
        out = t(small_graph)
        assert hasattr(out, "HKFE")
        assert out.HKFE.shape == (small_graph.num_nodes, t.fe_dim)

    def test_forward_handles_empty_edge_index(self):
        x = torch.randn(3, 2)
        data = Data(
            x=x, edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=3
        )
        t = HKFE(kernel_param_HKFE=(1, 3), concat_to_x=False)
        out = t(data)
        assert torch.all(out.HKFE == 0.0)

    def test_debug_mode(self, small_graph, capsys):
        """Test HKFE with debug=True."""
        t = HKFE(kernel_param_HKFE=(1, 3), debug=True)
        out = t(small_graph)
        captured = capsys.readouterr()
        assert "HKFE Debug Report" in captured.out
        assert out.x.shape == (small_graph.num_nodes, small_graph.x.shape[1] + t.fe_dim)
