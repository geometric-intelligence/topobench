"""Unit tests for the OGFormer backbone (topobench/nn/backbones/graph/ogformer.py)."""

import hydra
import pytest
import torch

from topobench.loss.model import OGFormerLoss
from topobench.nn.backbones.graph import OGFormer
from topobench.nn.backbones.graph.ogformer import (
    OGFormerAttention,
    OGFormerLayer,
    OGFormerPropagation,
    standardize_rows,
    symmetric_normalize,
)


@pytest.fixture
def two_graph_batch():
    """Create a deterministic batch of two graphs.

    Returns
    -------
    dict
        Dictionary with node features ``x``, block-structured
        ``edge_index`` and a ``batch`` vector for two 5-node graphs.
    """
    torch.manual_seed(42)
    x = torch.randn(10, 12)
    edges_g1 = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 3]])
    edges_g2 = edges_g1 + 5
    edge_index = torch.cat([edges_g1, edges_g2], dim=1)
    batch = torch.tensor([0] * 5 + [1] * 5)
    return {"x": x, "edge_index": edge_index, "batch": batch}


def test_standardize_rows():
    """Test that standardize_rows yields zero-mean, unit-std rows."""
    torch.manual_seed(0)
    x = torch.randn(6, 32) * 3.0 + 1.0
    x_hat = standardize_rows(x)
    assert torch.allclose(x_hat.mean(dim=1), torch.zeros(6), atol=1e-6)
    assert torch.allclose(x_hat.std(dim=1), torch.ones(6), atol=1e-4)


def test_symmetric_normalize():
    """Test symmetric normalization, including zero-degree rows."""
    adjacency = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    normalized = symmetric_normalize(adjacency)
    expected = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert torch.allclose(normalized, expected)
    assert not torch.isnan(normalized).any()


def test_build_dense_adjacency():
    """Test the dense adjacency construction is binary and symmetric."""
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 2]])
    adjacency = OGFormer.build_dense_adjacency(edge_index, 4, torch.float32)
    assert adjacency.shape == (4, 4)
    assert torch.equal(adjacency, adjacency.T)
    # Duplicated edges are not accumulated
    assert adjacency.max() == 1.0
    assert adjacency[3].sum() == 0.0


def test_ogformer_attention_scores():
    """Test the attention module output shapes and score properties."""
    torch.manual_seed(0)
    attention = OGFormerAttention(12, 16, alpha=0.5)
    x = torch.randn(8, 12)
    se = torch.eye(8)
    q, r = attention(x, se)
    assert q.shape == (8, 16)
    assert r.shape == (8, 8)
    assert (q > 0).all() and (q < 1).all()  # Sigmoid queries
    # Kernel part is row L1-normalized; SE bias adds alpha on the diagonal
    assert torch.allclose(r.sum(dim=1), torch.full((8,), 1.0 + 0.5), atol=1e-5)


def test_ogformer_propagation_norms():
    """Test random-walk and symmetric normalization propagation paths."""
    torch.manual_seed(0)
    x = torch.randn(6, 12)
    r = torch.rand(6, 6)
    for sym_norm in (False, True):
        propagation = OGFormerPropagation(12, 16, sym_norm=sym_norm)
        out = propagation(r, x)
        assert out.shape == (6, 16)
        assert not torch.isnan(out).any()


def test_ogformer_layer_activation_flag():
    """Test that the last-layer variant skips the ReLU activation."""
    torch.manual_seed(0)
    x = torch.randn(6, 12)
    se = torch.eye(6)
    layer = OGFormerLayer(12, 16, alpha=0.8, apply_activation=False)
    h, q, r = layer(x, se)
    assert h.shape == (6, 16)
    assert q.shape == (6, 16)
    assert r.shape == (6, 6)
    # Without ReLU the pre-residual messages can be negative: h - q < 0 somewhere
    assert ((h - q) < 0).any()


def test_ogformer_forward_shapes_and_aux(two_graph_batch):
    """Test the backbone forward pass in training and eval modes.

    Parameters
    ----------
    two_graph_batch : dict
        Deterministic two-graph batch fixture.
    """
    model = OGFormer(12, 16, n_layers=2, alpha=[0.8, 0.9], dropout=0.2)
    assert model.out_channels == 16

    model.train()
    out, aux = model(
        two_graph_batch["x"],
        two_graph_batch["edge_index"],
        two_graph_batch["batch"],
    )
    assert out.shape == (10, 16)
    assert len(aux["queries"]) == 2
    assert len(aux["attention_scores"]) == 2
    assert aux["attention_scores"][0].shape == (10, 10)

    model.eval()
    out_eval, aux_eval = model(
        two_graph_batch["x"],
        two_graph_batch["edge_index"],
        two_graph_batch["batch"],
    )
    assert out_eval.shape == (10, 16)
    assert aux_eval is None


def test_ogformer_blocks_cross_graph_attention(two_graph_batch):
    """Test that attention never crosses graph boundaries in a batch.

    Parameters
    ----------
    two_graph_batch : dict
        Deterministic two-graph batch fixture.
    """
    model = OGFormer(12, 16, n_layers=2)
    model.train()
    _, aux = model(
        two_graph_batch["x"],
        two_graph_batch["edge_index"],
        two_graph_batch["batch"],
    )
    for attention in aux["attention_scores"]:
        assert attention[:5, 5:].abs().sum() == 0.0
        assert attention[5:, :5].abs().sum() == 0.0


def test_ogformer_permutation_equivariance():
    """Test that node permutations permute the output accordingly."""
    torch.manual_seed(1)
    x = torch.randn(9, 12)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]]
    )
    model = OGFormer(12, 16, n_layers=2)
    model.eval()

    perm = torch.randperm(9)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(9)

    out, _ = model(x, edge_index)
    out_perm, _ = model(x[perm], inverse[edge_index])
    assert torch.allclose(out, out_perm[inverse], atol=1e-5)


def test_ogformer_alpha_validation():
    """Test that a wrong number of per-layer alphas raises an error."""
    with pytest.raises(ValueError):
        OGFormer(12, 16, n_layers=2, alpha=[0.8, 0.9, 1.0])


def test_ogformer_reset_parameters():
    """Test that parameters can be re-initialized."""
    model = OGFormer(12, 16, n_layers=2)
    model.reset_parameters()
    for layer in model.layers:
        assert (layer.propagation.lin.bias == 0).all()


def test_ogformer_hydra_instantiation():
    """Test that the Hydra config instantiates the backbone and its loss."""
    from topobench.utils.config_resolvers import register_all_resolvers

    register_all_resolvers()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(
        version_base="1.3", config_path="../../../../configs", job_name="job"
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=["model=graph/ogformer", "dataset=graph/MUTAG"],
            return_hydra_config=True,
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        # The exports manager re-loads backbone modules, so compare by name
        assert type(backbone).__name__ == OGFormer.__name__
        assert len(backbone.layers) == cfg.model.backbone.n_layers
        assert backbone.out_channels == cfg.model.feature_encoder.out_channels
        loss = hydra.utils.instantiate(cfg.model.backbone.loss)
        assert isinstance(loss, OGFormerLoss)
