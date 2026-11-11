"""Unit tests for the SGFormer graph backbone."""

import pytest
import torch
from torch_geometric.data import Batch

from topobench.nn.backbones import MODEL_CLASSES, SGFormer


class TestSGFormer:
    """Test SGFormer model behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.in_channels = 8
        self.hidden_channels = 16
        self.out_channels = 16

    def _features(self, num_nodes):
        return torch.randn(num_nodes, self.in_channels)

    def test_model_export(self):
        """Test dynamic model discovery."""
        assert SGFormer is MODEL_CLASSES["SGFormer"]

    def test_initialization_default(self):
        """Test default initialization."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
        )

        assert model.in_channels == self.in_channels
        assert model.hidden_channels == self.hidden_channels
        assert model.out_channels == self.hidden_channels
        assert model.aggregate == "add"

    def test_initialization_custom(self):
        """Test custom initialization."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            trans_num_layers=2,
            trans_num_heads=2,
            aggregate="cat",
            gnn_use_bn=False,
        )

        assert model.out_channels == self.out_channels
        assert model.aggregate == "cat"

    def test_invalid_aggregate(self):
        """Test invalid aggregate type."""
        with pytest.raises(ValueError, match="Invalid aggregate type"):
            SGFormer(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                aggregate="mean",
            )

    def test_forward_basic(self, simple_graph_0):
        """Test a basic forward pass."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            gnn_use_bn=False,
        )
        x = self._features(simple_graph_0.num_nodes)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)

        out = model(x, simple_graph_0.edge_index, batch=batch)

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_without_batch(self, simple_graph_0):
        """Test forward pass without a batch vector."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            gnn_use_bn=False,
        )
        out = model(self._features(simple_graph_0.num_nodes), simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_forward_cat_aggregate(self, simple_graph_0):
        """Test concatenation branch aggregation."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            aggregate="cat",
            gnn_use_bn=False,
        )
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        out = model(
            self._features(simple_graph_0.num_nodes),
            simple_graph_0.edge_index,
            batch=batch,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_batched_graphs(self, simple_graph_0, simple_graph_1):
        """Test forward pass on a batch of graphs."""
        data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            trans_num_layers=2,
            gnn_use_bn=False,
        )
        out = model(
            self._features(data.num_nodes),
            data.edge_index,
            batch=data.batch,
        )

        assert out.shape == (data.num_nodes, self.out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_backward_pass(self, simple_graph_0):
        """Test gradient computation."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            gnn_use_bn=False,
        )
        x = self._features(simple_graph_0.num_nodes).requires_grad_(True)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)

        loss = model(x, simple_graph_0.edge_index, batch=batch).sum()
        loss.backward()

        assert x.grad is not None
        assert any(
            param.grad is not None
            for param in model.parameters()
            if param.requires_grad
        )

    def test_single_node_training_batch_with_layer_norm(self):
        """Test the LayerNorm graph branch on a one-node training batch."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            gnn_use_bn=False,
        )
        model.train()
        x = self._features(1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)

        out = model(x, edge_index, batch=batch)

        assert out.shape == (1, self.out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_edge_weight_reference_path_ignored(self, simple_graph_0):
        """Test that edge weights are ignored unless explicitly enabled."""
        model = SGFormer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            gnn_use_bn=False,
            gnn_use_edge_weight=False,
            trans_dropout=0.0,
            gnn_dropout=0.0,
        )
        model.eval()
        x = self._features(simple_graph_0.num_nodes)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        edge_weight = torch.rand(simple_graph_0.edge_index.size(1))

        unweighted = model(x, simple_graph_0.edge_index, batch=batch)
        weighted = model(
            x,
            simple_graph_0.edge_index,
            batch=batch,
            edge_weight=edge_weight,
        )

        assert torch.allclose(unweighted, weighted)
