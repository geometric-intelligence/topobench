"""Unit tests for the PhenomNN backbone."""

import pytest
import torch

from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.hypergraph import PhenomNN
from topobench.nn.backbones.hypergraph.phenomnn import (
    PhenomNNConv,
    build_expansion_operators,
)


def _toy_incidence(sparse=True):
    """Build a small node-hyperedge incidence matrix.

    Hyperedges: {0, 1, 2} and {2, 3} over 4 nodes.

    Parameters
    ----------
    sparse : bool, optional
        Whether to return a sparse COO tensor (default: True).

    Returns
    -------
    torch.Tensor
        Incidence matrix of shape ``(4, 2)``.
    """
    dense = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    return dense.to_sparse_coo() if sparse else dense


def test_PhenomNN():
    """Test the PhenomNN backbone end to end with the auto-test harness."""
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    incidence = _toy_incidence(sparse=True)
    auto_test = NNModuleAutoTest(
        [
            {
                "module": PhenomNN,
                "init": (x.shape[1],),
                "forward": (x, incidence),
                "assert_shape": x.shape,
            },
        ]
    )
    auto_test.run()


def test_build_expansion_operators_properties():
    """Test shape, symmetry and finiteness of the expansion operators."""
    incidence = _toy_incidence(sparse=False)
    n = incidence.shape[0]
    a_beta, a_gamma, d_beta, d_gamma = build_expansion_operators(incidence)

    for adj in (a_beta, a_gamma):
        assert adj.shape == (n, n)
        assert torch.allclose(adj, adj.t(), atol=1e-6)  # symmetric
        assert torch.isfinite(adj).all()
    assert d_beta.shape == (n,)
    assert d_gamma.shape == (n,)
    # Node 3 only shares hyperedge {2,3}; node 0 shares {0,1,2}. The clique
    # adjacency must therefore connect 0 with 1 and 2 but not with 3.
    assert a_beta[0, 1] > 0
    assert a_beta[0, 2] > 0
    assert torch.isclose(a_beta[0, 3], torch.tensor(0.0))


def test_sparse_and_dense_incidence_match():
    """Test that sparse and dense incidence inputs give identical operators."""
    a_sparse = build_expansion_operators(_toy_incidence(sparse=True))
    a_dense = build_expansion_operators(_toy_incidence(sparse=False))
    for s, d in zip(a_sparse, a_dense):
        assert torch.allclose(s, d, atol=1e-6)


def test_full_variant_has_parameters():
    """Test that the full variant exposes learnable compatibility matrices."""
    model = PhenomNN(num_features=6, compatibility=True)
    names = [n for n, _ in model.named_parameters()]
    assert any("H_beta" in n for n in names)
    assert any("H_gamma" in n for n in names)


def test_simple_variant_is_parameter_free():
    """Test that the simple variant has no learnable backbone parameters."""
    model = PhenomNN(num_features=6, compatibility=False)
    assert sum(p.numel() for p in model.parameters()) == 0
    out, hyper = model(torch.randn(4, 6), _toy_incidence())
    assert out.shape == (4, 6)
    assert hyper is None


def test_forward_shapes_full():
    """Test forward output shapes for the full variant."""
    model = PhenomNN(num_features=6, compatibility=True, prop_step=4)
    out, hyper = model(torch.randn(4, 6), _toy_incidence())
    assert out.shape == (4, 6)
    assert hyper is None
    assert torch.isfinite(out).all()


def test_multiple_layers():
    """Test that num_layers controls the number of stacked blocks."""
    model = PhenomNN(num_features=6, num_layers=3)
    assert len(model.convs) == 3
    out, _ = model(torch.randn(4, 6), _toy_incidence())
    assert out.shape == (4, 6)


def test_alpha_default_resolution():
    """Test that alpha defaults to 1 / (1 + lam0 + lam1)."""
    model = PhenomNN(num_features=6, lam0=10.0, lam1=10.0, alpha=None)
    assert pytest.approx(model.convs[0].alpha) == 1.0 / 21.0
    # Explicit alpha is respected.
    model2 = PhenomNN(num_features=6, alpha=0.25)
    assert model2.convs[0].alpha == 0.25


def test_invalid_num_layers():
    """Test that a non-positive num_layers raises an assertion error."""
    with pytest.raises(AssertionError):
        PhenomNN(num_features=6, num_layers=0)


def test_invalid_prop_step():
    """Test that a non-positive prop_step raises an assertion error."""
    with pytest.raises(AssertionError):
        PhenomNN(num_features=6, prop_step=0)


def test_reset_parameters_reinitializes_identity():
    """Test that reset_parameters re-centers compatibility matrices on identity."""
    model = PhenomNN(num_features=6, compatibility=True)
    with torch.no_grad():
        model.convs[0].H_beta.add_(5.0)
    model.reset_parameters()
    # After reset, H_beta should be close to the identity (plus small noise).
    diff = model.convs[0].H_beta - torch.eye(6)
    assert diff.abs().mean() < 0.5


def test_conv_repr():
    """Test the PhenomNNConv repr string."""
    conv = PhenomNNConv(
        channels=6,
        prop_step=4,
        lam0=10.0,
        lam1=10.0,
        alpha=0.05,
        compatibility=False,
    )
    assert "PhenomNNConv" in repr(conv)
