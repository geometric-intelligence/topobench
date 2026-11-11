"""Tests for the JacobiConv graph backbone."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.jacobiconv import JacobiConv
from topobench.nn.wrappers import GNNWrapper


class TestJacobiConv:
    def test_forward_and_backward(self, simple_graph_0):
        x = torch.randn(simple_graph_0.num_nodes, 16, requires_grad=True)
        model = JacobiConv(16, 32, 24, polynomial_order=4, dropout=0.0)
        out = model(x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, 24)
        assert torch.isfinite(out).all()
        out.mean().backward()
        assert x.grad is not None
        assert model.filter_coefficients.grad is not None

    def test_order_zero(self, simple_graph_0):
        x = torch.randn(simple_graph_0.num_nodes, 8)
        model = JacobiConv(8, 8, 8, polynomial_order=0, dropout=0.0)
        assert model(x, simple_graph_0.edge_index).shape == x.shape

    def test_edgeless_and_batched(self, simple_graph_0, simple_graph_1):
        model = JacobiConv(8, 8, 8, polynomial_order=3, dropout=0.0)
        x = torch.randn(4, 8)
        assert model(x, torch.empty(2, 0, dtype=torch.long)).shape == x.shape
        batch = Batch.from_data_list([simple_graph_0, simple_graph_1])
        bx = torch.randn(batch.num_nodes, 8)
        assert model(bx, batch.edge_index, batch=batch.batch).shape == bx.shape

    def test_wrapper_compatibility(self, simple_graph_0):
        x = torch.randn(simple_graph_0.num_nodes, 16)
        data = Data(x=x, x_0=x, edge_index=simple_graph_0.edge_index,
                    y=simple_graph_0.y,
                    batch_0=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long))
        wrapper = GNNWrapper(JacobiConv(16, 16, 16, dropout=0.0), out_channels=16,
                             num_cell_dimensions=1, residual_connections=False)
        assert wrapper(data)["x_0"].shape == x.shape

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            JacobiConv(hidden_channels=8)
        with pytest.raises(ValueError):
            JacobiConv(8, 8, polynomial_order=-1)
        with pytest.raises(ValueError):
            JacobiConv(8, 8, jacobi_alpha=-1.0)
