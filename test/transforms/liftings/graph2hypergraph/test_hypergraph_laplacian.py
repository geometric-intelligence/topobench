"""Unit tests for the hypergraph Laplacian approximation."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from topobench.transforms.liftings.graph2hypergraph.hypergraph_laplacian import (
    Laplacian,
    adjacency,
    normalise,
    ssm2tst,
    symnormalise,
    update,
)


@pytest.mark.parametrize("mediators", [True, False])
def test_laplacian(mediators):
    """Unit test for Laplacian.

    Parameters
    ----------
    mediators : bool
        Whether the approximation uses mediators.
    """
    num_nodes = 6
    hyperedges = {0: [0, 1, 2], 1: [2, 3], 2: [3, 4, 5]}
    features = np.random.default_rng(0).normal(size=(num_nodes, 4))

    A = Laplacian(num_nodes, hyperedges, features, mediators)

    assert A.shape == (num_nodes, num_nodes)
    assert A.is_sparse
    dense = A.to_dense()
    assert torch.isfinite(dense).all()
    # Self loops are added before normalisation, so the diagonal is non-zero.
    assert (dense.diagonal() > 0).all()


def test_update():
    """Unit test for update."""
    weights = update(0, 1, 2, {}, c=3.0)

    assert set(weights) == {(0, 2), (1, 2), (2, 0), (2, 1)}
    for value in weights.values():
        assert value == pytest.approx(1 / 3)

    # Calling again accumulates on the existing keys.
    weights = update(0, 1, 2, weights, c=3.0)
    for value in weights.values():
        assert value == pytest.approx(2 / 3)


def test_adjacency():
    """Unit test for adjacency."""
    edges = [[0, 1], [1, 0], [0, 1]]  # duplicated pair is deduplicated
    weights = {(0, 1): 0.5, (1, 0): 0.5}

    A = adjacency(edges, weights, n=3)
    dense = A.to_dense()

    assert dense.shape == (3, 3)
    assert torch.allclose(dense, dense.t(), atol=1e-6)
    # Isolated node 2 only has its self loop, normalised to one.
    assert dense[2, 2].item() == pytest.approx(1.0, abs=1e-6)


def test_symnormalise():
    """Unit test for symnormalise."""
    M = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32))
    out = np.asarray(symnormalise(M).todense())

    assert np.allclose(out, np.eye(2), atol=1e-6)

    # A zero row yields a zero scaling factor rather than an infinity.
    M = sp.csr_matrix(np.array([[0.0, 0.0], [0.0, 4.0]], dtype=np.float32))
    out = np.asarray(symnormalise(M).todense())
    assert np.isfinite(out).all()


def test_normalise():
    """Unit test for normalise."""
    M = sp.csr_matrix(np.array([[1.0, 3.0], [0.0, 0.0]], dtype=np.float32))
    out = np.asarray(normalise(M).todense())

    assert out[0].sum() == pytest.approx(1.0)
    assert np.isfinite(out).all()


def test_ssm2tst():
    """Unit test for ssm2tst."""
    M = sp.coo_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    A = ssm2tst(M)

    assert A.is_sparse
    assert A.shape == (2, 2)
    assert torch.allclose(
        A.to_dense(), torch.tensor([[1.0, 0.0], [0.0, 2.0]]), atol=1e-6
    )
