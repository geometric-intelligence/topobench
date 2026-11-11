"""Unit tests for the EHNN backbone."""

import pytest
import torch

from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.hypergraph import EHNN
from topobench.nn.backbones.hypergraph.ehnn import (
    BiasV,
    InnerMLP,
    PositionalEncoding,
)


def _toy_incidence(sparse=True):
    """Build a small node-hyperedge incidence matrix.

    Hyperedges {0,1,2}, {2,3}, {3,4} over 5 nodes.

    Parameters
    ----------
    sparse : bool, optional
        Whether to return a sparse COO tensor (default: True).

    Returns
    -------
    torch.Tensor
        Incidence matrix of shape ``(5, 3)``.
    """
    dense = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return dense.to_sparse_coo() if sparse else dense


def test_EHNN():
    """Test the EHNN backbone end to end with the auto-test harness."""
    torch.manual_seed(0)
    x = torch.randn(5, 6)
    incidence = _toy_incidence(sparse=True)
    NNModuleAutoTest(
        [
            {
                "module": EHNN,
                "init": {
                    "num_features": 6,
                    "hidden_channels": 8,
                    "max_edge_order": 10,
                },
                "forward": (x, incidence),
                "assert_shape": (5, 8),
            }
        ]
    ).run()


def test_build_cache():
    """Test that the operator cache has correct orders and normalizers."""
    model = EHNN(num_features=6, hidden_channels=8, max_edge_order=10)
    cache = model._build_cache(_toy_incidence(sparse=False))
    assert torch.equal(
        cache["edge_orders"], torch.tensor([3, 2, 2])
    )  # hyperedge sizes
    assert torch.allclose(
        cache["prefix_normalizer"], torch.tensor([3.0, 2.0, 2.0])
    )
    # node degrees: nodes 2 and 3 are in two hyperedges.
    assert torch.allclose(
        cache["suffix_normalizer"], torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0])
    )
    assert cache["incidence"].is_sparse


def test_edge_order_clamping():
    """Test that hyperedge orders are clamped to max_edge_order."""
    model = EHNN(num_features=6, hidden_channels=8, max_edge_order=2)
    cache = model._build_cache(_toy_incidence(sparse=False))
    # The size-3 hyperedge must be clamped down to 2.
    assert cache["edge_orders"].max().item() == 2


def test_forward_shapes():
    """Test forward output shapes and that the hyperedge slot is None."""
    model = EHNN(num_features=6, hidden_channels=8, max_edge_order=10)
    out, hyper = model(torch.randn(5, 6), _toy_incidence())
    assert out.shape == (5, 8)
    assert hyper is None
    assert torch.isfinite(out).all()


def test_sparse_and_dense_incidence_run():
    """Test that both sparse and dense incidence inputs are accepted."""
    model = EHNN(num_features=6, hidden_channels=8, max_edge_order=10)
    model.eval()
    x = torch.randn(5, 6)
    out_sparse, _ = model(x, _toy_incidence(sparse=True))
    out_dense, _ = model(x, _toy_incidence(sparse=False))
    assert out_sparse.shape == out_dense.shape == (5, 8)


def test_invalid_max_edge_order():
    """Test that a non-positive max_edge_order raises an assertion error."""
    with pytest.raises(AssertionError):
        EHNN(num_features=6, hidden_channels=8, max_edge_order=0)


def test_reset_parameters():
    """Test that reset_parameters runs and keeps parameters finite."""
    model = EHNN(num_features=6, hidden_channels=8, max_edge_order=10)
    model.reset_parameters()
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_custom_hyper_dims():
    """Test EHNN with explicit pe_dim / hyper_dim and multiple hyper layers."""
    model = EHNN(
        num_features=6,
        hidden_channels=8,
        max_edge_order=10,
        pe_dim=4,
        hyper_dim=16,
        hyper_layers=2,
    )
    out, _ = model(torch.randn(5, 6), _toy_incidence())
    assert out.shape == (5, 8)


def test_inner_mlp_single_and_multi_layer():
    """Test the InnerMLP for both single- and multi-layer configurations."""
    x = torch.randn(4, 6)
    single = InnerMLP(6, 3, 5, n_layers=1, dropout=0.0)
    assert single(x).shape == (4, 3)
    multi = InnerMLP(6, 3, 5, n_layers=3, dropout=0.0)
    assert multi(x).shape == (4, 3)


def test_positional_encoding_shape():
    """Test that the positional encoding looks up the right shape."""
    pe = PositionalEncoding(dim=8, max_pos=5)
    idx = torch.tensor([0, 2, 4])
    assert pe(idx).shape == (3, 8)


def test_bias_v():
    """Test the learnable node bias module."""
    bias = BiasV(dim_out=8)
    x = torch.randn(5, 8)
    out = bias(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)  # a nonzero bias is added
