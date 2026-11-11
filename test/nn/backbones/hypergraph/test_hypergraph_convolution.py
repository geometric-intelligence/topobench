"""Unit tests for HyperGraphConvolution."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.hypergraph.hypergraph_convolution import (
    HyperGraphConvolution,
    SparseMM,
    incidence_to_hyperedges,
)
from topobench.nn.wrappers import HypergraphWrapper


def _square_incidence(num_nodes, seed=0):
    """Build a square sparse incidence matrix with no empty hyperedge.

    Parameters
    ----------
    num_nodes : int
        Number of nodes, also used as the number of hyperedges.
    seed : int, optional
        Seed for reproducibility, by default 0.

    Returns
    -------
    torch.Tensor
        Sparse incidence matrix of shape ``[num_nodes, num_nodes]``.
    """
    generator = torch.Generator().manual_seed(seed)
    incidence = (
        torch.rand(num_nodes, num_nodes, generator=generator) > 0.4
    ).float()
    # Guarantee every hyperedge has at least two members.
    incidence[0, :] = 1.0
    incidence[1, :] = 1.0
    return incidence.to_sparse_coo()


def test_incidence_to_hyperedges():
    """Unit test for incidence_to_hyperedges."""
    incidence = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ).to_sparse_coo()

    # Hyperedges 1 and 2 are singletons and must be dropped.
    hyperedges = incidence_to_hyperedges(incidence)
    assert set(hyperedges) == {0}
    assert sorted(hyperedges[0]) == [0, 1, 3]

    # Lowering min_size keeps them.
    hyperedges = incidence_to_hyperedges(incidence, min_size=1)
    assert set(hyperedges) == {0, 1, 2}
    assert sorted(hyperedges[1]) == [2]
    assert sorted(hyperedges[2]) == [0]

    # Every hyperedge is a singleton -> empty dict.
    assert incidence_to_hyperedges(torch.eye(4).to_sparse_coo()) == {}


def test_forward_without_reapproximation():
    """Unit test for the forward pass reusing the incidence matrix."""
    num_nodes, in_channels, out_channels = 6, 5, 3
    x_0 = torch.randn(num_nodes, in_channels)
    incidence = _square_incidence(num_nodes)

    model = HyperGraphConvolution(
        in_channels, out_channels, reapproximate=False
    )
    x_0_out, x_1_out = model(x_0, incidence)

    assert x_0_out.shape == (num_nodes, out_channels)
    assert x_1_out.shape == (num_nodes, out_channels)
    assert torch.isfinite(x_0_out).all()


@pytest.mark.parametrize("mediators", [True, False])
def test_forward_with_reapproximation(mediators):
    """Unit test for the forward pass rebuilding the Laplacian.

    Parameters
    ----------
    mediators : bool
        Whether the Laplacian approximation uses mediators.
    """
    num_nodes, num_hyperedges, in_channels, out_channels = 8, 5, 4, 3
    x_0 = torch.randn(num_nodes, in_channels)

    incidence = torch.zeros(num_nodes, num_hyperedges)
    for edge in range(num_hyperedges):
        members = torch.arange(edge, min(edge + 3, num_nodes))
        incidence[members, edge] = 1.0
    incidence = incidence.to_sparse_coo()

    model = HyperGraphConvolution(
        in_channels, out_channels, reapproximate=True
    )
    x_0_out, x_1_out = model(x_0, incidence, m=mediators)

    assert x_0_out.shape == (num_nodes, out_channels)
    assert x_1_out.shape == (num_hyperedges, out_channels)
    assert torch.isfinite(x_0_out).all()


def test_forward_with_only_singleton_hyperedges():
    """Unit test for the identity fallback when no hyperedge survives."""
    num_nodes, in_channels, out_channels = 4, 3, 2
    x_0 = torch.randn(num_nodes, in_channels)
    incidence = torch.eye(num_nodes).to_sparse_coo()

    model = HyperGraphConvolution(
        in_channels, out_channels, reapproximate=True
    )
    x_0_out, _ = model(x_0, incidence)

    expected = x_0 @ model.W + model.bias
    assert torch.allclose(x_0_out, expected, atol=1e-5)


def test_backward():
    """Unit test that gradients reach the layer parameters."""
    num_nodes, in_channels, out_channels = 6, 5, 3
    x_0 = torch.randn(num_nodes, in_channels)
    incidence = _square_incidence(num_nodes)

    model = HyperGraphConvolution(
        in_channels, out_channels, reapproximate=False
    )
    x_0_out, _ = model(x_0, incidence)
    x_0_out.sum().backward()

    assert model.W.grad is not None
    assert model.W.grad.shape == model.W.shape
    assert model.bias.grad is not None


def test_sparse_mm():
    """Unit test for SparseMM covering both backward branches."""
    m1 = torch.randn(3, 4, requires_grad=True)
    m2 = torch.randn(4, 2, requires_grad=True)

    out = SparseMM.apply(m1, m2)
    assert torch.allclose(out, m1 @ m2, atol=1e-6)

    out.sum().backward()
    assert m1.grad.shape == m1.shape
    assert m2.grad.shape == m2.shape


def test_reset_parameters():
    """Unit test that parameters are reinitialised in range."""
    model = HyperGraphConvolution(4, 16)
    model.reset_parameters()

    bound = 1.0 / (16**0.5)
    assert model.W.abs().max().item() <= bound
    assert model.bias.abs().max().item() <= bound


def test_repr():
    """Unit test for the string representation."""
    model = HyperGraphConvolution(4, 3)
    assert repr(model) == "HyperGraphConvolution (4 -> 3)"


def test_hypergraph_wrapper():
    """Unit test for HyperGraphConvolution behind its wrapper."""
    num_nodes, channels = 6, 4
    x_0 = torch.randn(num_nodes, channels)
    incidence = _square_incidence(num_nodes)

    batch = torch_geometric.data.Data(
        x_0=x_0,
        y=torch.randint(0, 2, (num_nodes,)),
        incidence_hyperedges=incidence,
        batch_0=torch.zeros(num_nodes, dtype=torch.long),
    )

    backbone = HyperGraphConvolution(channels, channels, reapproximate=False)
    wrapper = HypergraphWrapper(
        backbone, **{"out_channels": channels, "num_cell_dimensions": 1}
    )

    _ = wrapper.__repr__()
    model_out = wrapper(batch)

    assert model_out["x_0"].shape == x_0.shape
    assert model_out["hyperedge"].shape == (num_nodes, channels)
    assert "labels" in model_out
    assert "batch_0" in model_out
