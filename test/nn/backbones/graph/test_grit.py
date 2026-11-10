"""Unit tests for the GRIT backbone (Graph Inductive Bias Transformer)."""

import hydra
import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph import BACKBONE_CLASSES
from topobench.nn.backbones.graph.grit import (
    GRITAttention,
    GRITBackbone,
    GRITTransformerLayer,
    full_edge_index,
)
from topobench.transforms.data_manipulations.rrwp_positional_encodings import (
    AddRRWP,
)

HIDDEN_DIM = 16
WALK_LENGTH = 4


def _make_graph(num_nodes: int, seed: int = 0) -> Data:
    """Create a small random connected graph with RRWP encodings.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    seed : int, optional
        Random seed for the node features (default: 0).

    Returns
    -------
    torch_geometric.data.Data
        Graph with features, labels and RRWP attributes.
    """
    generator = torch.Generator().manual_seed(seed)
    src = torch.arange(num_nodes - 1)
    dst = src + 1
    edge_index = torch.cat(
        [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
    )
    data = Data(
        x=torch.randn(num_nodes, HIDDEN_DIM, generator=generator),
        edge_index=edge_index,
        y=torch.zeros(1, dtype=torch.long),
    )
    return AddRRWP(walk_length=WALK_LENGTH)(data)


def _forward(model: GRITBackbone, data: Data, **overrides) -> torch.Tensor:
    """Run the backbone on a (batched) data object.

    Parameters
    ----------
    model : GRITBackbone
        The backbone to evaluate.
    data : torch_geometric.data.Data
        Graph or batch with RRWP attributes.
    **overrides : dict
        Keyword arguments overriding the batch attributes.

    Returns
    -------
    torch.Tensor
        Node representations.
    """
    kwargs = {
        "batch": data.get("batch", None),
        "edge_attr": data.get("edge_attr", None),
        "rrwp": data.get("rrwp", None),
        "rrwp_index": data.get("rrwp_index", None),
        "rrwp_val": data.get("rrwp_val", None),
        "log_deg": data.get("log_deg", None),
    }
    kwargs.update(overrides)
    return model(data.x, data.edge_index, **kwargs)


class TestFullEdgeIndex:
    """Test the full_edge_index utility."""

    def test_single_graph(self):
        """Test all pairs of a single graph."""
        batch = torch.zeros(3, dtype=torch.long)
        index = full_edge_index(batch)
        assert index.shape == (2, 9)

    def test_no_cross_graph_pairs(self):
        """Test that no pair connects two different graphs."""
        batch = torch.tensor([0, 0, 1, 1, 1])
        index = full_edge_index(batch)

        assert index.shape == (2, 4 + 9)
        assert torch.equal(batch[index[0]], batch[index[1]])


class TestGRITAttention:
    """Test the GRIT attention mechanism."""

    def test_invalid_num_heads(self):
        """Test that indivisible head counts are rejected."""
        with pytest.raises(ValueError, match="divisible"):
            GRITAttention(hidden_dim=10, num_heads=4)

    def test_signed_sqrt(self):
        """Test the signed square root activation."""
        x = torch.tensor([4.0, -9.0, 0.0])
        expected = torch.tensor([2.0, -3.0, 0.0])
        assert torch.allclose(GRITAttention.signed_sqrt(x), expected)

    def test_forward_shapes(self):
        """Test output shapes of the attention forward pass."""
        num_nodes, num_pairs, num_heads = 5, 12, 4
        attention = GRITAttention(HIDDEN_DIM, num_heads)
        x = torch.randn(num_nodes, HIDDEN_DIM)
        pair_index = torch.randint(0, num_nodes, (2, num_pairs))
        pair_attr = torch.randn(num_pairs, HIDDEN_DIM)

        x_out, pair_out = attention(x, pair_index, pair_attr)

        assert x_out.shape == (num_nodes, num_heads, HIDDEN_DIM // num_heads)
        assert pair_out.shape == (num_pairs, HIDDEN_DIM)
        assert torch.isfinite(x_out).all()
        assert torch.isfinite(pair_out).all()

    def test_no_clamp_and_no_edge_enhance(self):
        """Test the forward pass without clamping and edge enhancement."""
        attention = GRITAttention(
            HIDDEN_DIM, num_heads=2, clamp=None, edge_enhance=False
        )
        assert attention.clamp is None
        assert not hasattr(attention, "W_Ev")

        x = torch.randn(4, HIDDEN_DIM)
        pair_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        pair_attr = torch.randn(4, HIDDEN_DIM)
        x_out, _ = attention(x, pair_index, pair_attr)
        assert x_out.shape == (4, 2, HIDDEN_DIM // 2)


class TestGRITTransformerLayer:
    """Test the GRIT transformer layer."""

    def _inputs(self, num_nodes=6, num_pairs=20):
        """Create random layer inputs.

        Parameters
        ----------
        num_nodes : int, optional
            Number of nodes (default: 6).
        num_pairs : int, optional
            Number of attention pairs (default: 20).

        Returns
        -------
        tuple
            Node features, pair index, pair attributes and log-degrees.
        """
        x = torch.randn(num_nodes, HIDDEN_DIM)
        pair_index = torch.randint(0, num_nodes, (2, num_pairs))
        pair_attr = torch.randn(num_pairs, HIDDEN_DIM)
        log_deg = torch.rand(num_nodes, 1)
        return x, pair_index, pair_attr, log_deg

    def test_forward_shapes(self):
        """Test output shapes of the layer."""
        layer = GRITTransformerLayer(HIDDEN_DIM, num_heads=4)
        x, pair_index, pair_attr, log_deg = self._inputs()

        x_out, pair_out = layer(x, pair_index, pair_attr, log_deg)

        assert x_out.shape == x.shape
        assert pair_out.shape == pair_attr.shape
        assert not torch.equal(pair_out, pair_attr)

    def test_no_degree_scaler(self):
        """Test the layer without the adaptive degree scaler."""
        layer = GRITTransformerLayer(HIDDEN_DIM, num_heads=4, deg_scaler=False)
        assert not hasattr(layer, "deg_coef")

        x, pair_index, pair_attr, log_deg = self._inputs()
        x_out, _ = layer(x, pair_index, pair_attr, log_deg)
        assert x_out.shape == x.shape

    def test_frozen_pair_representations(self):
        """Test that update_pair_rep=False keeps pair representations."""
        layer = GRITTransformerLayer(
            HIDDEN_DIM, num_heads=4, update_pair_rep=False
        )
        x, pair_index, pair_attr, log_deg = self._inputs()

        _, pair_out = layer(x, pair_index, pair_attr, log_deg)
        assert torch.equal(pair_out, pair_attr)


class TestGRITBackbone:
    """Test the GRIT backbone."""

    def test_registration(self):
        """Test that the backbone is auto-discovered by TopoBench."""
        assert "GRITBackbone" in BACKBONE_CLASSES

    def test_initialization(self):
        """Test default and custom initialization."""
        model = GRITBackbone(hidden_dim=HIDDEN_DIM)
        assert len(model.layers) == 4
        assert model.walk_length == 8

        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=2, walk_length=WALK_LENGTH
        )
        assert len(model.layers) == 2
        assert model.abs_pe_encoder.in_features == WALK_LENGTH

    def test_forward_with_precomputed_rrwp(self):
        """Test the forward pass with precomputed RRWP encodings."""
        data = _make_graph(7)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=2, walk_length=WALK_LENGTH
        )

        out = _forward(model, data)

        assert out.shape == (7, HIDDEN_DIM)
        assert torch.isfinite(out).all()

    def test_forward_without_batch_vector(self):
        """Test that a missing batch vector defaults to a single graph."""
        data = _make_graph(5)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=1, walk_length=WALK_LENGTH
        )
        out = model(
            data.x,
            data.edge_index,
            rrwp=data.rrwp,
            rrwp_index=data.rrwp_index,
            rrwp_val=data.rrwp_val,
            log_deg=data.log_deg,
        )
        assert out.shape == (5, HIDDEN_DIM)

    def test_fallback_matches_precomputed(self):
        """Test on-the-fly RRWP equals the precomputed transform output."""
        batch = Batch.from_data_list([_make_graph(6, 0), _make_graph(4, 1)])
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=2, walk_length=WALK_LENGTH
        ).eval()

        out_precomputed = _forward(model, batch)
        out_on_the_fly = _forward(
            model,
            batch,
            rrwp=None,
            rrwp_index=None,
            rrwp_val=None,
            log_deg=None,
        )

        assert torch.allclose(out_precomputed, out_on_the_fly, atol=1e-6)

    def test_batching_matches_individual_graphs(self):
        """Test that batched processing equals per-graph processing."""
        data_0, data_1 = _make_graph(6, 0), _make_graph(4, 1)
        batch = Batch.from_data_list([data_0, data_1])
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=2, walk_length=WALK_LENGTH
        ).eval()

        out_batched = _forward(model, batch)
        out_individual = torch.cat(
            [_forward(model, data_0), _forward(model, data_1)]
        )

        assert torch.allclose(out_batched, out_individual, atol=1e-5)

    def test_walk_length_mismatch(self):
        """Test that mismatched RRWP channels raise an error."""
        data = _make_graph(5)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=1, walk_length=WALK_LENGTH + 1
        )
        with pytest.raises(ValueError, match="walk_length"):
            _forward(model, data)

    def test_sparse_attention(self):
        """Test the forward pass without full-graph padding."""
        data = _make_graph(6)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM,
            num_layers=1,
            walk_length=WALK_LENGTH,
            pad_to_full_graph=False,
        )
        out = _forward(model, data)
        assert out.shape == (6, HIDDEN_DIM)

    def test_edge_attributes_used_when_configured(self):
        """Test that configured edge attributes change the output."""
        data = _make_graph(5)
        data.edge_attr = torch.randn(data.edge_index.size(1), 3)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM,
            num_layers=1,
            walk_length=WALK_LENGTH,
            edge_dim=3,
        ).eval()

        out_with_edges = _forward(model, data)
        out_without_edges = _forward(model, data, edge_attr=None)

        assert out_with_edges.shape == (5, HIDDEN_DIM)
        assert not torch.allclose(out_with_edges, out_without_edges)

    def test_edge_attributes_ignored_by_default(self):
        """Test that edge attributes are ignored when edge_dim is None."""
        data = _make_graph(5)
        data.edge_attr = torch.randn(data.edge_index.size(1), 3)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=1, walk_length=WALK_LENGTH
        ).eval()

        assert model.edge_attr_encoder is None
        out_with_edges = _forward(model, data)
        out_without_edges = _forward(model, data, edge_attr=None)
        assert torch.allclose(out_with_edges, out_without_edges)

    def test_backward_pass(self):
        """Test gradient flow through the backbone."""
        data = _make_graph(6)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=2, walk_length=WALK_LENGTH
        )

        out = _forward(model, data)
        out.sum().backward()

        # The pair-representation output of the last layer does not
        # influence the node output, so its projection has no gradient;
        # every other parameter must receive one.
        last_layer_pair_params = {
            id(p)
            for name, p in model.layers[-1].named_parameters()
            if "W_O_pair" in name or "norm1_pair" in name
        }
        for name, param in model.named_parameters():
            if id(param) in last_layer_pair_params:
                continue
            assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_in_eval_mode(self):
        """Test deterministic outputs in eval mode despite dropout."""
        data = _make_graph(6)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            walk_length=WALK_LENGTH,
            dropout=0.5,
            attn_dropout=0.5,
        ).eval()

        assert torch.allclose(_forward(model, data), _forward(model, data))

    def test_extra_kwargs_ignored(self):
        """Test that unused keyword arguments are ignored gracefully."""
        data = _make_graph(5)
        model = GRITBackbone(
            hidden_dim=HIDDEN_DIM, num_layers=1, walk_length=WALK_LENGTH
        )
        out = _forward(model, data, unused_kwarg="test")
        assert out.shape == (5, HIDDEN_DIM)


class TestGRITHydraConfig:
    """Test Hydra configuration loading for the GRIT model."""

    def setup_method(self):
        """Reset the global Hydra state."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_config_composition_and_instantiation(self):
        """Test that the grit model config composes and instantiates."""
        with hydra.initialize(
            config_path="../../../../configs", job_name="test_grit", version_base="1.3"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=["model=graph/grit", "dataset=graph/MUTAG"],
                return_hydra_config=True,
            )

            # The model-default transform is picked up and kept in sync
            # with the backbone's walk_length.
            assert "grit_rrwp" in cfg.transforms
            assert cfg.transforms.grit_rrwp.transform_name == "AddRRWP"
            assert (
                cfg.transforms.grit_rrwp.walk_length
                == cfg.model.backbone.walk_length
            )

            backbone = hydra.utils.instantiate(cfg.model.backbone)
            # The exports manager loads backbones under a separate module
            # identity, so compare by class name instead of isinstance.
            assert type(backbone).__name__ == GRITBackbone.__name__
            assert (
                backbone.hidden_dim == cfg.model.feature_encoder.out_channels
            )
