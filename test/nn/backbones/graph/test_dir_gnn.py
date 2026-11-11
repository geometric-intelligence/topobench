"""Unit tests for DirGNN backbone (Rossi et al., LoG 2024)."""

import pytest
import torch
from topobench.nn.backbones.graph.dir_gnn import DirGNN, DirSageConv


@pytest.fixture
def small_graph():
    """A tiny directed graph: 10 nodes, 20 random edges."""
    x = torch.randn(10, 8)
    edge_index = torch.randint(0, 10, (2, 20))
    return x, edge_index


class TestDirSageConv:
    def test_output_shape(self, small_graph):
        x, edge_index = small_graph
        conv = DirSageConv(in_channels=8, out_channels=16)
        out = conv(x, edge_index)
        assert out.shape == (10, 16), f"Expected (10, 16), got {out.shape}"

    def test_alpha_zero(self, small_graph):
        """alpha=0: only out-neighbour aggregation contributes."""
        x, edge_index = small_graph
        conv = DirSageConv(in_channels=8, out_channels=16, alpha=0.0)
        out = conv(x, edge_index)
        assert out.shape == (10, 16)

    def test_alpha_one(self, small_graph):
        """alpha=1: only in-neighbour aggregation contributes."""
        x, edge_index = small_graph
        conv = DirSageConv(in_channels=8, out_channels=16, alpha=1.0)
        out = conv(x, edge_index)
        assert out.shape == (10, 16)

    def test_no_edges(self):
        """Graph with no edges: should fall back to self-transform."""
        x = torch.randn(5, 8)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        conv = DirSageConv(in_channels=8, out_channels=16)
        out = conv(x, edge_index)
        assert out.shape == (5, 16)

    def test_reset_parameters(self, small_graph):
        x, edge_index = small_graph
        conv = DirSageConv(in_channels=8, out_channels=16)
        conv.reset_parameters()
        out = conv(x, edge_index)
        assert not torch.isnan(out).any()


class TestDirGNN:
    def test_output_shape_2layers(self, small_graph):
        x, edge_index = small_graph
        model = DirGNN(in_channels=8, hidden_channels=32, out_channels=4, num_layers=2)
        out = model(x, edge_index)
        assert out.shape == (10, 4)

    def test_output_shape_3layers(self, small_graph):
        x, edge_index = small_graph
        model = DirGNN(in_channels=8, hidden_channels=32, out_channels=4, num_layers=3)
        out = model(x, edge_index)
        assert out.shape == (10, 4)

    def test_output_shape_1layer(self, small_graph):
        x, edge_index = small_graph
        model = DirGNN(in_channels=8, hidden_channels=32, out_channels=4, num_layers=1)
        out = model(x, edge_index)
        assert out.shape == (10, 4)

    def test_no_nan_output(self, small_graph):
        x, edge_index = small_graph
        model = DirGNN(in_channels=8, hidden_channels=32, out_channels=4)
        out = model(x, edge_index)
        assert not torch.isnan(out).any()

    def test_train_eval_mode(self, small_graph):
        """Dropout should behave differently in train vs eval."""
        x, edge_index = small_graph
        model = DirGNN(in_channels=8, hidden_channels=32, out_channels=4, dropout=0.9)
        model.train()
        out_train = model(x, edge_index)
        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)
        # Shapes must match regardless of mode
        assert out_train.shape == out_eval.shape

    def test_gradient_flow(self, small_graph):
        """Loss should produce gradients for all parameters."""
        x, edge_index = small_graph
        model = DirGNN(in_channels=8, hidden_channels=32, out_channels=4)
        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
