"""Unit tests for the GFT graph backbone."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.gft import GFT, _TreeVocabulary
from topobench.nn.wrappers import GNNWrapper


class TestGFT:
    """Test the TopoBench GFT core encoder."""

    def setup_method(self):
        """Set up test dimensions."""
        self.in_channels = 16
        self.hidden_channels = 32
        self.out_channels = 24

    def _features(self, num_nodes, channels=None):
        """Create node features."""
        channels = self.in_channels if channels is None else channels
        return torch.randn(num_nodes, channels)

    def test_initialization_default(self):
        """Test default initialization."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
        )

        assert model.in_channels == self.in_channels
        assert model.hidden_channels == self.hidden_channels
        assert model.out_channels == self.hidden_channels
        assert model.num_layers == 2
        assert model.backbone == "sage"
        assert model.vocabulary.codebook.shape == (4, 128, 32)

    def test_initialization_custom(self):
        """Test custom initialization."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=3,
            backbone="gcn",
            normalize="layer",
            tree_depth=3,
            codebook_size=17,
            codebook_heads=2,
            code_dim=8,
        )

        assert model.out_channels == self.out_channels
        assert model.num_layers == 3
        assert model.tree_depth == 3
        assert model.vocabulary.codebook.shape == (2, 17, 8)

    def test_initialization_aliases(self):
        """Test input_dim and hidden_dim aliases."""
        model = GFT(input_dim=self.in_channels, hidden_dim=20)

        assert model.in_channels == self.in_channels
        assert model.hidden_channels == 20

    def test_invalid_backbone(self):
        """Test that invalid message-passing backbone names fail."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                backbone="invalid",
            )

    def test_invalid_normalization(self):
        """Test that invalid normalization names fail."""
        with pytest.raises(ValueError, match="Unsupported normalize"):
            GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                normalize="invalid",
            )

    def test_invalid_core_dimensions(self):
        """Test validation for required and non-negative dimensions."""
        with pytest.raises(ValueError, match="in_channels or input_dim"):
            GFT(hidden_channels=self.hidden_channels)

        with pytest.raises(ValueError, match="num_layers"):
            GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                num_layers=0,
            )

        with pytest.raises(ValueError, match="tree_depth"):
            GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                tree_depth=-1,
            )

        with pytest.raises(ValueError, match="Unsupported activation"):
            GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                activation="sigmoid",
            )

    def test_tree_vocabulary_validation(self):
        """Test vocabulary parameter validation."""
        with pytest.raises(ValueError, match="codebook_size"):
            _TreeVocabulary(
                dim=8,
                codebook_size=0,
                code_dim=4,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="code_dim"):
            _TreeVocabulary(
                dim=8,
                codebook_size=4,
                code_dim=0,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="num_heads"):
            _TreeVocabulary(
                dim=8,
                codebook_size=4,
                code_dim=4,
                num_heads=0,
            )

    def test_tree_vocabulary_fixed_euclidean_lookup(self):
        """Test fixed codebooks and Euclidean nearest-code lookup."""
        vocabulary = _TreeVocabulary(
            dim=8,
            codebook_size=5,
            code_dim=4,
            num_heads=2,
            use_cosine_sim=False,
            learnable_codebook=False,
        )
        assert "codebook" in dict(vocabulary.named_buffers())
        assert "codebook" not in dict(vocabulary.named_parameters())

        out, token_ids, commitment_loss = vocabulary(torch.randn(6, 8))

        assert out.shape == (6, 8)
        assert token_ids.shape == (6, 2)
        assert commitment_loss.dim() == 0

    def test_tree_vocabulary_empty_input(self):
        """Test empty graph tokenization."""
        vocabulary = _TreeVocabulary(
            dim=8,
            codebook_size=5,
            code_dim=4,
            num_heads=2,
        )

        out, token_ids, commitment_loss = vocabulary(torch.empty(0, 8))

        assert out.shape == (0, 8)
        assert token_ids.shape == (0, 2)
        assert commitment_loss.item() == 0

    def test_forward_basic(self, simple_graph_0):
        """Test a basic forward pass."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            normalize="none",
        )
        x = self._features(simple_graph_0.num_nodes)

        out = model(x=x, edge_index=simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert model.last_tree_token_ids.shape == (
            simple_graph_0.num_nodes,
            model.codebook_heads,
        )

    def test_forward_with_aux(self, simple_graph_0):
        """Test optional tree-token diagnostics."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            codebook_heads=2,
            code_dim=8,
        )
        x = self._features(simple_graph_0.num_nodes)

        out, aux = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            return_aux=True,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert aux["tree_token_ids"].shape == (simple_graph_0.num_nodes, 2)
        assert aux["commitment_loss"].dim() == 0

    def test_forward_with_edge_weight(self, simple_graph_0):
        """Test scalar edge weights are accepted by the forward API."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
        )
        x = self._features(simple_graph_0.num_nodes)
        edge_weight = torch.rand(simple_graph_0.edge_index.size(1))

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_weight=edge_weight,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_forward_with_scalar_edge_attr_shapes(self, simple_graph_0):
        """Test scalar edge attributes are coerced to GCN edge weights."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            backbone="gcn",
            normalize="none",
        )
        x = self._features(simple_graph_0.num_nodes)
        one_dim_attr = torch.rand(simple_graph_0.edge_index.size(1))
        two_dim_attr = one_dim_attr.view(-1, 1)

        out_one_dim = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_attr=one_dim_attr,
        )
        out_two_dim = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_attr=two_dim_attr,
        )

        assert out_one_dim.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert out_two_dim.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_forward_with_ignored_vector_edge_attr(self, simple_graph_0):
        """Test vector edge attributes are ignored without failing."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
        )
        x = self._features(simple_graph_0.num_nodes)
        edge_attr = torch.randn(simple_graph_0.edge_index.size(1), 4)

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_attr=edge_attr,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_edgeless_graph(self):
        """Test tree descriptors on a graph with no edges."""
        num_nodes = 4
        edge_index = torch.empty(2, 0, dtype=torch.long)
        x = self._features(num_nodes)
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            tree_depth=2,
            normalize="none",
        )

        out = model(x=x, edge_index=edge_index)

        assert out.shape == (num_nodes, self.out_channels)

    def test_activation_and_normalization_variants(self, simple_graph_0):
        """Test supported activation and normalization variants."""
        x = self._features(simple_graph_0.num_nodes)

        for activation in ["gelu", "elu", "leaky_relu"]:
            model = GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                activation=activation,
                normalize="none",
            )
            out = model(x=x, edge_index=simple_graph_0.edge_index)
            assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

        for normalize in ["batch", "batch_norm", "layer", "layer_norm", None]:
            model = GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                normalize=normalize,
            )
            out = model(x=x, edge_index=simple_graph_0.edge_index)
            assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_different_local_backbones(self, simple_graph_0):
        """Test supported message-passing backbones."""
        x = self._features(simple_graph_0.num_nodes)

        for backbone in ["sage", "gcn", "gin", "gat"]:
            model = GFT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                backbone=backbone,
                normalize="none",
            )
            out = model(x=x, edge_index=simple_graph_0.edge_index)
            assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_wrapper_compatibility(self, simple_graph_0):
        """Test compatibility with the standard graph wrapper."""
        channels = self.in_channels
        x = self._features(simple_graph_0.num_nodes, channels)
        batch = Data(
            x=x,
            x_0=x,
            edge_index=simple_graph_0.edge_index,
            y=simple_graph_0.y,
            batch_0=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        model = GFT(
            in_channels=channels,
            hidden_channels=channels,
            out_channels=channels,
            normalize="none",
        )
        wrapper = GNNWrapper(
            model,
            out_channels=channels,
            num_cell_dimensions=1,
            residual_connections=False,
        )

        model_out = wrapper(batch)

        assert model_out["x_0"].shape == x.shape
        assert model_out["batch_0"].shape == (simple_graph_0.num_nodes,)
        assert torch.equal(model_out["labels"], simple_graph_0.y)

    def test_batched_graphs(self, simple_graph_0, simple_graph_1):
        """Test batched graph inputs."""
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        x = self._features(batch_data.num_nodes)
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            normalize="none",
        )

        out = model(
            x=x,
            edge_index=batch_data.edge_index,
            batch=batch_data.batch,
        )

        assert out.shape == (batch_data.num_nodes, self.out_channels)

    def test_backward_pass(self, simple_graph_0):
        """Test gradients flow through encoder and tree-vocabulary path."""
        model = GFT(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            normalize="none",
        )
        x = self._features(simple_graph_0.num_nodes).requires_grad_(True)

        out = model(x=x, edge_index=simple_graph_0.edge_index)
        out.mean().backward()

        assert x.grad is not None
        assert any(
            param.grad is not None
            for param in model.parameters()
            if param.requires_grad
        )
