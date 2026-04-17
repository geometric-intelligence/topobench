"""Unit tests for NSD (Neural Sheaf Diffusion) Encoder."""

import pytest
import torch
from torch_geometric.data import Batch

from topobench.nn.backbones.graph.nsd import NSDEncoder


class TestNSDEncoder:
    """Test NSDEncoder model."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.num_nodes = 10
        self.num_edges = 20

    def _prepare_features(self, num_nodes, feat_dim=None):
        """Helper to create features matching the model's expected dimension.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        feat_dim : int, optional
            Feature dimension. If None, uses self.input_dim.

        Returns
        -------
        torch.Tensor
            Random feature tensor of shape [num_nodes, feat_dim].
        """
        if feat_dim is None:
            feat_dim = self.input_dim
        return torch.randn(num_nodes, feat_dim)

    def test_initialization_default(self):
        """Test default initialization of NSDEncoder."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )

        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_layers == 2  # default
        assert model.sheaf_type == "diag"  # default
        assert model.d == 2  # default
        assert model.sheaf_model is not None

    def test_initialization_custom(self):
        """Test custom initialization of NSDEncoder."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
            sheaf_type="bundle",
            d=4,
            dropout=0.2,
            input_dropout=0.15,
            sheaf_act="elu",
            orth="matrix_exp"
        )

        assert model.num_layers == 3
        assert model.sheaf_type == "bundle"
        assert model.d == 4
        assert model.sheaf_model is not None

    def test_initialization_diag_sheaf(self):
        """Test initialization with diagonal sheaf type."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="diag",
            d=2
        )

        assert model.sheaf_type == "diag"
        assert model.d == 2

    def test_initialization_bundle_sheaf(self):
        """Test initialization with bundle sheaf type."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="bundle",
            d=4
        )

        assert model.sheaf_type == "bundle"
        assert model.d == 4

    def test_initialization_general_sheaf(self):
        """Test initialization with general sheaf type."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="general",
            d=3
        )

        assert model.sheaf_type == "general"
        assert model.d == 3

    def test_initialization_invalid_sheaf_type(self):
        """Test that invalid sheaf type raises error."""
        with pytest.raises(ValueError, match="Unknown sheaf type"):
            NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                sheaf_type="invalid_type"
            )

    def test_initialization_invalid_sheaf_act(self):
        """Test that invalid sheaf activation raises error."""
        # The error occurs during initialization when the sheaf model is created
        with pytest.raises(ValueError, match="Unsupported act"):
            NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                sheaf_act="relu"  # Not supported, only 'id', 'tanh', 'elu' are valid
            )

    def test_initialization_diag_d_validation(self):
        """Test that diag sheaf type validates d >= 1."""
        # Should work with d=1
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            sheaf_type="diag",
            d=1
        )
        assert model.d == 1

        # Should fail with d < 1
        with pytest.raises(AssertionError):
            NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                sheaf_type="diag",
                d=0
            )

    def test_initialization_bundle_d_validation(self):
        """Test that bundle sheaf type validates d > 1."""
        # Should work with d=2
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            sheaf_type="bundle",
            d=2
        )
        assert model.d == 2

        # Should fail with d <= 1
        with pytest.raises(AssertionError):
            NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                sheaf_type="bundle",
                d=1
            )

    def test_initialization_general_d_validation(self):
        """Test that general sheaf type validates d > 1."""
        # Should work with d=2
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            sheaf_type="general",
            d=2
        )
        assert model.d == 2

        # Should fail with d <= 1
        with pytest.raises(AssertionError):
            NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                sheaf_type="general",
                d=1
            )

    def test_forward_basic(self, simple_graph_0):
        """Test basic forward pass.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_batch(self, simple_graph_0):
        """Test forward pass without batch vector.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_with_edge_attr(self, simple_graph_0):
        """Test forward pass with edge attributes (should be ignored).

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        # Create dummy edge attributes (should be ignored by NSD)
        edge_attr = torch.randn(simple_graph_0.edge_index.shape[1], 4)

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_attr=edge_attr,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_with_edge_weight(self, simple_graph_0):
        """Test forward pass with edge weights (should be ignored).

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        # Create dummy edge weights (should be ignored by NSD)
        edge_weight = torch.randn(simple_graph_0.edge_index.shape[1])

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_weight=edge_weight,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_diag_sheaf(self, simple_graph_0):
        """Test forward pass with diagonal sheaf.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="diag",
            d=2
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_bundle_sheaf(self, simple_graph_0):
        """Test forward pass with bundle sheaf.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="bundle",
            d=4
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_general_sheaf(self, simple_graph_0):
        """Test forward pass with general sheaf.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="general",
            d=3
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_different_num_layers(self, simple_graph_0):
        """Test forward pass with different number of layers.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        for num_layers in [1, 2, 4, 6]:
            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=num_layers
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
            assert model.num_layers == num_layers

    def test_forward_different_d_values(self, simple_graph_0):
        """Test forward pass with different d values.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        for d in [2, 4, 8]:
            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                d=d
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
            assert model.d == d

    def test_forward_different_dropout(self, simple_graph_0):
        """Test forward pass with different dropout rates.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        for dropout in [0.0, 0.1, 0.3, 0.5]:
            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=dropout
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_different_input_dropout(self, simple_graph_0):
        """Test forward pass with different input dropout rates.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        for input_dropout in [0.0, 0.1, 0.3, 0.5]:
            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                input_dropout=input_dropout
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_different_sheaf_activations(self, simple_graph_0):
        """Test forward pass with different sheaf activations.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        for sheaf_act in ["tanh", "elu", "id"]:
            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                sheaf_act=sheaf_act
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_different_orth_methods(self, simple_graph_0):
        """Test forward pass with different orthogonalization methods.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        for orth in ["cayley", "matrix_exp"]:
            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                orth=orth
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_training_mode(self, simple_graph_0):
        """Test forward pass in training mode.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        model.train()

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert model.training

    def test_eval_mode(self, simple_graph_0):
        """Test forward pass in evaluation mode.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        model.eval()

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not model.training

    def test_backward_pass(self, simple_graph_0):
        """Test backward pass and gradient computation.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        x = self._prepare_features(simple_graph_0.num_nodes).requires_grad_(True)
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        # At least some parameters should have gradients
        has_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        assert has_grad, "No gradients computed for any model parameters"

    def test_batched_graphs(self, simple_graph_0, simple_graph_1):
        """Test forward pass with batched graphs.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            First test graph fixture.
        simple_graph_1 : torch_geometric.data.Data
            Second test graph fixture.
        """
        expected_nodes = simple_graph_0.num_nodes + simple_graph_1.num_nodes
        x = self._prepare_features(expected_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        # Create batch
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])

        out = model(
            x=x,
            edge_index=batch_data.edge_index,
            batch=batch_data.batch
        )

        assert out.shape == (expected_nodes, self.hidden_dim)

    def test_empty_graph(self):
        """Test forward pass with empty graph (single node, no edges)."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        # Set to eval mode to avoid batch norm issues with single node
        model.eval()

        x = self._prepare_features(1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)

        out = model(x=x, edge_index=edge_index, batch=batch)

        assert out.shape == (1, self.hidden_dim)

    def test_large_graph(self):
        """Test forward pass with a larger graph."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        num_nodes = 100
        num_edges = 300
        x = self._prepare_features(num_nodes)

        # Create edges in one direction only, then to_undirected will handle making it bidirectional
        # This ensures proper pairing that NSD expects
        edge_list = []
        for _ in range(num_edges):
            src = torch.randint(0, num_nodes, (1,)).item()
            tgt = torch.randint(0, num_nodes, (1,)).item()
            # Ensure src < tgt to avoid duplicates when making undirected
            if src > tgt:
                src, tgt = tgt, src
            if src != tgt:  # Avoid self-loops
                edge_list.append([src, tgt])

        # Remove duplicates
        edge_list = list(set(tuple(e) for e in edge_list))

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            # Fallback: create at least one edge
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        batch = torch.zeros(num_nodes, dtype=torch.long)

        out = model(x=x, edge_index=edge_index, batch=batch)

        assert out.shape == (num_nodes, self.hidden_dim)

    def test_deterministic_output_eval_mode(self, simple_graph_0):
        """Test that output is deterministic in eval mode.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout=0.5
        )
        model.eval()

        # Run forward pass twice
        out1 = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        out2 = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert torch.allclose(out1, out2)

    def test_different_hidden_dims(self, simple_graph_0):
        """Test with different hidden dimensions.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        for hidden_dim in [16, 32, 64, 128]:
            x = self._prepare_features(simple_graph_0.num_nodes)

            model = NSDEncoder(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                num_layers=2
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, hidden_dim)

    def test_different_input_dims(self, simple_graph_0):
        """Test with different input dimensions.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        for input_dim in [8, 16, 32, 64]:
            x = self._prepare_features(simple_graph_0.num_nodes, input_dim)

            model = NSDEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            )

            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )

            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_model_device_consistency(self, simple_graph_0):
        """Test that model respects device placement.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            device="cuda"
        )
        model = model.cuda()

        x = self._prepare_features(simple_graph_0.num_nodes).cuda()
        edge_index = simple_graph_0.edge_index.cuda()
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long).cuda()

        out = model(x=x, edge_index=edge_index, batch=batch)

        assert out.is_cuda
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
    def test_parametrized_num_layers(self, simple_graph_0, num_layers):
        """Parametrized test for different number of layers.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        num_layers : int
            Number of NSD layers to test.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert model.num_layers == num_layers

    @pytest.mark.parametrize("sheaf_type", ["diag", "bundle", "general"])
    def test_parametrized_sheaf_type(self, simple_graph_0, sheaf_type):
        """Parametrized test for different sheaf types.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        sheaf_type : str
            Type of sheaf to test.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        # Use d=2 for all types (minimum valid for all)
        d = 2 if sheaf_type != "diag" else 1
        if sheaf_type in ["bundle", "general"]:
            d = 4  # Use larger d for non-diag types

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type=sheaf_type,
            d=d
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert model.sheaf_type == sheaf_type

    @pytest.mark.parametrize("d", [2, 4, 8])
    def test_parametrized_d_values(self, simple_graph_0, d):
        """Parametrized test for different d values.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        d : int
            Stalk dimension to test.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            d=d
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert model.d == d

    @pytest.mark.parametrize("orth", ["cayley", "matrix_exp"])
    def test_parametrized_orth_method(self, simple_graph_0, orth):
        """Parametrized test for different orthogonalization methods.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        orth : str
            Orthogonalization method to test.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            orth=orth
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_kwargs_ignored(self, simple_graph_0):
        """Test that additional kwargs are ignored gracefully.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
            unused_kwarg="test",
            another_unused=123
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_get_sheaf_model(self):
        """Test get_sheaf_model method."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )

        sheaf_model = model.get_sheaf_model()

        assert sheaf_model is not None
        assert sheaf_model == model.sheaf_model

    def test_sheaf_config_correctness(self):
        """Test that sheaf config is set up correctly."""
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=3,
            d=4,
            dropout=0.2,
            input_dropout=0.15,
            sheaf_act="elu",
            orth="matrix_exp"
        )

        config = model.sheaf_config

        assert config["d"] == 4
        assert config["layers"] == 3
        assert config["hidden_channels"] == 64 // 4  # hidden_dim // d
        assert config["input_dim"] == self.input_dim
        assert config["output_dim"] == 64
        assert config["dropout"] == 0.2
        assert config["input_dropout"] == 0.15
        assert config["sheaf_act"] == "elu"
        assert config["orth"] == "matrix_exp"

    def test_multiple_forward_passes_same_graph(self, simple_graph_0):
        """Test multiple forward passes on the same graph.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        model.eval()

        # Multiple forward passes
        for _ in range(5):
            out = model(
                x=x,
                edge_index=simple_graph_0.edge_index,
                batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
            )
            assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_gradient_flow(self, simple_graph_0):
        """Test that gradients flow through the entire network.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        model.train()

        x = self._prepare_features(simple_graph_0.num_nodes).requires_grad_(True)
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        # Compute loss
        loss = out.mean()
        loss.backward()

        # Check gradients in different parts of the model
        param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(param_grads) > 0, "No parameters have gradients"

        # Check that gradients are not all zeros
        non_zero_grads = [g for g in param_grads if g.abs().sum() > 0]
        assert len(non_zero_grads) > 0, "All gradients are zero"

    def test_bundle_sheaf_with_matrix_exp(self, simple_graph_0):
        """Test bundle sheaf with matrix_exp orthogonalization.

        This tests the orthogonal.py matrix_exp path (line 39).

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="bundle",
            d=4,
            orth="matrix_exp"
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_general_sheaf_with_matrix_exp(self, simple_graph_0):
        """Test general sheaf with matrix_exp orthogonalization.

        This ensures complete coverage of matrix_exp path in orthogonal.py.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)

        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="general",
            d=3,
            orth="matrix_exp"
        )

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_bundle_sheaf_gradient_flow(self, simple_graph_0):
        """Test gradient flow through bundle sheaf.

        This tests the bundle laplacian builder normalise method (lines 153-176).

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="bundle",
            d=4
        )
        model.train()

        x = self._prepare_features(simple_graph_0.num_nodes).requires_grad_(True)
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        has_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        assert has_grad, "No gradients computed for any model parameters"

    def test_general_sheaf_gradient_flow(self, simple_graph_0):
        """Test gradient flow through general sheaf.

        This ensures coverage of general sheaf laplacian methods.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = NSDEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            sheaf_type="general",
            d=3
        )
        model.train()

        x = self._prepare_features(simple_graph_0.num_nodes).requires_grad_(True)
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        )

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        has_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
        assert has_grad, "No gradients computed for any model parameters"
