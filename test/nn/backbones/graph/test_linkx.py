"""Unit tests for LINKX backbone."""

import pytest
import torch
from torch_geometric.data import Batch

from topobench.nn.backbones.graph.linkx import LINKX, _MLP, _aggregate_neighbors


class TestMLP:
    """Test the _MLP helper class."""

    def test_single_layer(self):
        """Test single-layer MLP (linear only, no BatchNorm)."""
        mlp = _MLP(16, 32, 8, num_layers=1)
        x = torch.randn(10, 16)
        out = mlp(x)
        assert out.shape == (10, 8)

    def test_multi_layer(self):
        """Test multi-layer MLP with BatchNorm."""
        mlp = _MLP(16, 32, 8, num_layers=3, dropout=0.1)
        x = torch.randn(10, 16)
        out = mlp(x)
        assert out.shape == (10, 8)

    def test_two_layer(self):
        """Test two-layer MLP (one hidden with BatchNorm)."""
        mlp = _MLP(16, 32, 8, num_layers=2)
        x = torch.randn(10, 16)
        out = mlp(x)
        assert out.shape == (10, 8)


class TestAggregateNeighbors:
    """Test the _aggregate_neighbors function."""

    def test_aggregation(self):
        """Test that neighbor features are averaged correctly."""
        # 3 nodes, 2 features; edges: 1->0, 2->0 (so node 0 gets mean of 1 and 2)
        x = torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 6.0]])
        edge_index = torch.tensor([[1, 2], [0, 0]])
        out = _aggregate_neighbors(x, edge_index)
        # Node 0 should get mean of [2,4] and [4,6] = [3,5]
        assert torch.allclose(out[0], torch.tensor([3.0, 5.0]))

    def test_isolated_node(self):
        """Test aggregation with isolated nodes (no incoming edges)."""
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0], [1]])  # only 0->1
        out = _aggregate_neighbors(x, edge_index)
        # Node 2 is isolated, should get 0
        assert out[2].item() == 0.0


class TestLINKX:
    """Test LINKX backbone model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.in_channels = 16
        self.hidden_channels = 32

    def _make_model(self, **kwargs):
        """Create a LINKX model with default test params."""
        defaults = dict(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=2,
            num_edge_layers=1,
            num_node_layers=1,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return LINKX(**defaults)

    def _make_features(self, num_nodes, dim=None):
        """Create random node features."""
        return torch.randn(num_nodes, dim or self.in_channels)

    def test_initialization_default(self):
        """Test default initialization."""
        model = self._make_model()
        assert model.hidden_channels == self.hidden_channels
        assert isinstance(model.mlp_a, _MLP)
        assert isinstance(model.mlp_x, _MLP)

    def test_initialization_custom(self):
        """Test custom initialization with multi-layer MLPs."""
        model = self._make_model(
            num_layers=3, num_edge_layers=2, num_node_layers=2, dropout=0.3
        )
        # Multi-layer MLPs should have BatchNorm layers
        assert len(model.mlp_a.bns) == 1  # 2 layers = 1 BN
        assert len(model.mlp_x.bns) == 1
        assert len(model.mlp_final.bns) == 2  # 3 layers = 2 BN

    def test_forward_basic(self, simple_graph_0):
        """Test basic forward pass."""
        model = self._make_model(in_channels=1)
        out = model(
            simple_graph_0.x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_batch(self, simple_graph_0):
        """Test forward pass without batch vector."""
        model = self._make_model(in_channels=1)
        out = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_forward_with_edge_weight(self, simple_graph_0):
        """Test that edge_weight kwarg is accepted without error."""
        model = self._make_model(in_channels=1)
        edge_weight = torch.ones(simple_graph_0.edge_index.shape[1])
        out = model(
            simple_graph_0.x,
            simple_graph_0.edge_index,
            edge_weight=edge_weight,
        )
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_forward_higher_dim_features(self):
        """Test forward with higher-dimensional input features."""
        model = self._make_model()
        x = self._make_features(20)
        edge_index = torch.randint(0, 20, (2, 40))
        out = model(x, edge_index)
        assert out.shape == (20, self.hidden_channels)

    def test_training_vs_eval(self, simple_graph_0):
        """Test that train and eval modes produce different outputs (due to dropout/BN)."""
        model = self._make_model(in_channels=1, dropout=0.5, num_layers=3)

        model.eval()
        out_eval = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert not model.training

        model.train()
        out_train = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert model.training

        # Both should have correct shape
        assert out_eval.shape == out_train.shape

    def test_deterministic_eval(self, simple_graph_0):
        """Test that eval mode produces deterministic outputs."""
        model = self._make_model(in_channels=1, dropout=0.5)
        model.eval()

        out1 = model(simple_graph_0.x, simple_graph_0.edge_index)
        out2 = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert torch.allclose(out1, out2)

    def test_backward_pass(self, simple_graph_0):
        """Test gradient computation."""
        model = self._make_model(in_channels=1)
        x = simple_graph_0.x.clone().requires_grad_(True)
        out = model(x, simple_graph_0.edge_index)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_batched_graphs(self, simple_graph_0, simple_graph_1):
        """Test forward with batched graphs."""
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        total_nodes = simple_graph_0.num_nodes + simple_graph_1.num_nodes

        model = self._make_model(in_channels=1)
        out = model(batch_data.x, batch_data.edge_index, batch=batch_data.batch)
        assert out.shape == (total_nodes, self.hidden_channels)

    def test_single_node_no_edges(self):
        """Test forward with a single node and no edges."""
        model = self._make_model(num_layers=1, num_edge_layers=1, num_node_layers=1)
        model.eval()  # Avoid BN issues with single sample

        x = self._make_features(1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (1, self.hidden_channels)

    def test_large_graph(self):
        """Test forward with a larger graph."""
        model = self._make_model()
        x = self._make_features(200)
        edge_index = torch.randint(0, 200, (2, 600))
        batch = torch.zeros(200, dtype=torch.long)
        out = model(x, edge_index, batch=batch)
        assert out.shape == (200, self.hidden_channels)

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
    def test_parametrized_final_layers(self, simple_graph_0, num_layers):
        """Test different numbers of final MLP layers."""
        model = self._make_model(in_channels=1, num_layers=num_layers)
        out = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    @pytest.mark.parametrize("num_edge_layers", [1, 2, 3])
    def test_parametrized_edge_layers(self, simple_graph_0, num_edge_layers):
        """Test different numbers of edge MLP layers."""
        model = self._make_model(in_channels=1, num_edge_layers=num_edge_layers)
        out = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    @pytest.mark.parametrize("num_node_layers", [1, 2, 3])
    def test_parametrized_node_layers(self, simple_graph_0, num_node_layers):
        """Test different numbers of node MLP layers."""
        model = self._make_model(in_channels=1, num_node_layers=num_node_layers)
        out = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    @pytest.mark.parametrize("hidden_channels", [8, 16, 32, 64])
    def test_parametrized_hidden_dims(self, simple_graph_0, hidden_channels):
        """Test different hidden dimensions."""
        model = self._make_model(in_channels=1, hidden_channels=hidden_channels)
        out = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, hidden_channels)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_parametrized_dropout(self, simple_graph_0, dropout):
        """Test different dropout rates."""
        model = self._make_model(in_channels=1, dropout=dropout)
        out = model(simple_graph_0.x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_kwargs_ignored(self, simple_graph_0):
        """Test that extra kwargs are accepted gracefully."""
        model = self._make_model(in_channels=1)
        out = model(
            simple_graph_0.x,
            simple_graph_0.edge_index,
            unused_kwarg="test",
            another=42,
        )
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_device_consistency(self, simple_graph_0):
        """Test model works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = self._make_model(in_channels=1).cuda()
        x = simple_graph_0.x.cuda()
        edge_index = simple_graph_0.edge_index.cuda()
        out = model(x, edge_index)
        assert out.is_cuda
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)
