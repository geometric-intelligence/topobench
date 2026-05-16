"""Unit tests for the Bundle Neural Network graph backbone."""

import pytest
import torch
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.bunn import BuNN, BuNNLayer


def _undirected_edge_index(data):
    """Return an undirected edge index for stable diffusion tests."""
    return to_undirected(data.edge_index, num_nodes=data.num_nodes)


class TestBuNNLayer:
    """Test the BuNN diffusion layer."""

    def test_bundle_maps_are_orthogonal(self):
        """Node-wise bundle maps should be valid 2D rotations."""
        layer = BuNNLayer(
            hidden_channels=16,
            num_bundles=4,
            bundle_dim=2,
            dropout=0.0,
        )
        x = torch.randn(5, 16)

        maps = layer._compute_bundle_maps(x)
        identity = torch.eye(2).expand(5, 4, 2, 2)
        product = maps @ maps.transpose(-1, -2)

        assert maps.shape == (5, 4, 2, 2)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_invalid_bundle_dimension_raises(self):
        """Only 2D bundles are supported by the direct parameterization."""
        with pytest.raises(ValueError, match="bundle_dim=2"):
            BuNNLayer(hidden_channels=16, num_bundles=2, bundle_dim=4)

    def test_invalid_hidden_width_raises(self):
        """The hidden width must split cleanly into bundle channels."""
        with pytest.raises(ValueError, match="divisible"):
            BuNNLayer(hidden_channels=18, num_bundles=4, bundle_dim=2)

    def test_random_walk_laplacian_handles_edgeless_graph(self):
        """Isolated nodes should have a zero random-walk Laplacian."""
        x = torch.randn(3, 8)
        edge_index = torch.empty(2, 0, dtype=torch.long)

        out = BuNNLayer._random_walk_laplacian(x, edge_index)

        assert torch.equal(out, torch.zeros_like(x))


class TestBuNN:
    """Test the full BuNN graph encoder."""

    def test_initialization(self):
        """BuNN stores its main architectural parameters."""
        model = BuNN(
            in_channels=8,
            hidden_channels=16,
            num_layers=3,
            num_bundles=4,
            t=0.5,
            taylor_degree=2,
        )

        assert model.in_channels == 8
        assert model.hidden_channels == 16
        assert model.num_layers == 3
        assert model.num_bundles == 4
        assert model.t == 0.5
        assert model.taylor_degree == 2
        assert len(model.layers) == 3

    def test_forward_shape(self, simple_graph_0):
        """BuNN returns one hidden vector per node."""
        edge_index = _undirected_edge_index(simple_graph_0)
        x = torch.randn(simple_graph_0.num_nodes, 8)
        model = BuNN(
            in_channels=8,
            hidden_channels=16,
            num_layers=2,
            num_bundles=4,
            taylor_degree=2,
            dropout=0.0,
        )

        out = model(x=x, edge_index=edge_index)

        assert out.shape == (simple_graph_0.num_nodes, 16)
        assert torch.isfinite(out).all()

    def test_forward_with_edge_weights(self, simple_graph_0):
        """BuNN accepts scalar edge weights from the graph wrapper."""
        edge_index = _undirected_edge_index(simple_graph_0)
        edge_weight = torch.ones(edge_index.shape[1])
        x = torch.randn(simple_graph_0.num_nodes, 16)
        model = BuNN(
            in_channels=16,
            hidden_channels=16,
            num_layers=1,
            num_bundles=4,
            dropout=0.0,
        )

        out = model(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_forward_backward(self, simple_graph_0):
        """Gradients should flow through bundle maps and diffusion."""
        edge_index = _undirected_edge_index(simple_graph_0)
        x = torch.randn(simple_graph_0.num_nodes, 8, requires_grad=True)
        model = BuNN(
            in_channels=8,
            hidden_channels=16,
            num_layers=1,
            num_bundles=4,
            dropout=0.0,
        )

        loss = model(x=x, edge_index=edge_index).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_invalid_num_layers_raises(self):
        """At least one BuNN layer is required."""
        with pytest.raises(ValueError, match="num_layers"):
            BuNN(in_channels=8, hidden_channels=16, num_layers=0)
