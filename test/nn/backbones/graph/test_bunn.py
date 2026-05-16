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
        """Node-wise bundle maps should cover both O(2) components."""
        layer = BuNNLayer(
            hidden_channels=16,
            num_bundles=4,
            bundle_dim=2,
            dropout=0.0,
        )
        x = torch.randn(5, 16)

        maps = layer._compute_bundle_maps(x)
        identity = torch.eye(2, dtype=maps.dtype, device=maps.device).expand(
            5, 4, 2, 2
        )
        product = maps @ maps.transpose(-1, -2)
        determinants = torch.linalg.det(maps)
        expected_determinants = maps.new_tensor([1.0, 1.0, -1.0, -1.0])

        assert maps.shape == (5, 4, 2, 2)
        assert torch.allclose(product, identity, atol=1e-5)
        assert torch.allclose(
            determinants,
            expected_determinants.expand_as(determinants),
            atol=1e-5,
        )

    def test_invalid_bundle_dimension_raises(self):
        """Only 2D bundles are supported by the direct parameterization."""
        with pytest.raises(ValueError, match="bundle_dim=2"):
            BuNNLayer(hidden_channels=16, num_bundles=2, bundle_dim=4)

    def test_invalid_hidden_width_raises(self):
        """The hidden width must split cleanly into bundle channels."""
        with pytest.raises(ValueError, match="divisible"):
            BuNNLayer(hidden_channels=18, num_bundles=4, bundle_dim=2)

    def test_invalid_diffusion_time_raises(self):
        """Diffusion time is a non-negative heat-equation parameter."""
        with pytest.raises(ValueError, match="non-negative"):
            BuNNLayer(hidden_channels=16, num_bundles=4, bundle_dim=2, t=-0.1)

    def test_reflection_parameterization_requires_even_bundles(self):
        """The paper-style O(2) split needs matched rotations/reflections."""
        with pytest.raises(ValueError, match="even"):
            BuNNLayer(hidden_channels=18, num_bundles=3, bundle_dim=2)

    def test_random_walk_laplacian_handles_edgeless_graph(self):
        """Isolated nodes should have a zero random-walk Laplacian."""
        x = torch.randn(3, 8)
        edge_index = torch.empty(2, 0, dtype=torch.long)

        out = BuNNLayer._random_walk_laplacian(x, edge_index)

        assert torch.equal(out, torch.zeros_like(x))

    def test_random_walk_laplacian_symmetrizes_edges(self):
        """A one-way edge should be treated as an undirected graph edge."""
        x = torch.tensor([[0.0], [2.0]])
        edge_index = torch.tensor([[0], [1]])

        out = BuNNLayer._random_walk_laplacian(x, edge_index)

        assert torch.allclose(out, torch.tensor([[-2.0], [2.0]]))

    def test_random_walk_laplacian_ignores_self_loops(self):
        """Self-loops are not part of the paper's simple graph Laplacian."""
        x = torch.tensor([[0.0], [2.0]])
        edge_index = torch.tensor([[0, 0], [0, 1]])

        out = BuNNLayer._random_walk_laplacian(x, edge_index)

        assert torch.allclose(out, torch.tensor([[-2.0], [2.0]]))

    def test_edge_weight_length_must_match_edges(self):
        """Weighted Laplacians need one scalar weight per edge."""
        x = torch.tensor([[0.0], [2.0]])
        edge_index = torch.tensor([[0], [1]])
        edge_weight = torch.tensor([1.0, 1.0])

        with pytest.raises(ValueError, match="one scalar per edge"):
            BuNNLayer._random_walk_laplacian(x, edge_index, edge_weight)

    def test_zero_time_identity_update_recovers_input(self):
        """Synchronization and desynchronization should cancel for identity W."""
        layer = BuNNLayer(
            hidden_channels=8,
            num_bundles=2,
            bundle_dim=2,
            t=0.0,
            act="identity",
            dropout=0.0,
            residual=False,
            norm=None,
        )
        with torch.no_grad():
            layer.channel_mixer.weight.copy_(torch.eye(8))
            layer.channel_mixer.bias.zero_()
        x = torch.randn(4, 8)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        out = layer(x, edge_index)

        assert torch.allclose(out, x, atol=1e-5)


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
        assert model.layers[0].angle_network[0].out_features == 8

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
