"""Unit tests for PolynomialFilterTNN (polynomial filter in the Hodge Laplacian).

These tests exercise the backbone directly on small dense/sparse Hodge
Laplacians. They verify that the graph polynomial-filter bases act on the
Hodge Laplacian verbatim, that the ``rescale`` normalization lands the
Chebyshev domain in ``[0, 2]``, and that a signal on each simplicial rank
is filtered independently.
"""

import pytest
import torch

from topobench.nn.backbones.graph.poly_filter.bases import (
    Chebyshev,
    Jacobi,
    Monomial,
)
from topobench.nn.backbones.simplicial.hodge_utils import (
    spectral_upper_bound,
    spmm,
)
from topobench.nn.backbones.simplicial.poly_filter_tnn import (
    PolynomialFilterTNN,
)


def _sym_psd(n: int, seed: int) -> torch.Tensor:
    """Small symmetric PSD matrix standing in for a Hodge Laplacian."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g)
    return a @ a.T / n


def _dummy_complex(f: int = 4):
    """Signals + Hodge Laplacians for a 5-node / 7-edge / 3-triangle complex."""
    x_all = (torch.randn(5, f), torch.randn(7, f), torch.randn(3, f))
    laplacian_all = (_sym_psd(5, 1), _sym_psd(7, 2), _sym_psd(3, 3))
    return x_all, laplacian_all


class TestPolynomialFilterTNN:
    """Tests for the per-rank Hodge polynomial filter."""

    def test_forward_shapes_per_rank(self):
        x_all, laplacian_all = _dummy_complex()
        m = PolynomialFilterTNN(
            4, 8, 3, K=5, basis=Chebyshev(), laplacian_norm="rescale"
        )
        ys = m(x_all, laplacian_all)
        assert [tuple(y.shape) for y in ys] == [(5, 3), (7, 3), (3, 3)]
        assert all(torch.isfinite(y).all() for y in ys)

    def test_theta_is_learnable_per_rank(self):
        m = PolynomialFilterTNN(4, 8, 3, K=5, basis=Chebyshev())
        assert len(m.theta) == 3
        for t in m.theta:
            assert t.requires_grad and t.shape == (6,)

    def test_backward_flows_to_theta(self):
        x_all, laplacian_all = _dummy_complex()
        m = PolynomialFilterTNN(
            4, 8, 3, K=4, basis=Monomial(), laplacian_norm="raw"
        )
        loss = sum(y.sum() for y in m(x_all, laplacian_all))
        loss.backward()
        assert all(
            t.grad is not None and t.grad.abs().sum() > 0 for t in m.theta
        )

    @pytest.mark.parametrize(
        ("basis", "norm"),
        [
            (Chebyshev(), "rescale"),
            (Monomial(), "raw"),
            (Jacobi(alpha=1.0, beta=1.0), "rescale"),
        ],
    )
    def test_basis_is_swappable(self, basis, norm):
        """The graph bases act on the Hodge Laplacian with no changes."""
        x_all, laplacian_all = _dummy_complex()
        m = PolynomialFilterTNN(
            4, 6, 2, K=4, basis=basis, laplacian_norm=norm
        )
        ys = m(x_all, laplacian_all)
        assert all(torch.isfinite(y).all() for y in ys)

    def test_rescale_maps_spectrum_to_0_2(self):
        """``rescale`` hands the basis an operator with spectrum ~[0, 2]."""
        laplacian = _sym_psd(6, 4)
        m = PolynomialFilterTNN(
            4, 4, 2, K=3, basis=Chebyshev(), laplacian_norm="rescale"
        )
        op = m._build_laplacian_apply(laplacian)(torch.eye(6))  # (2/bound) L
        eig = torch.linalg.eigvalsh(op)
        # The Gershgorin bound is a guaranteed upper bound on λmax, so the
        # rescaled spectrum always lands in [0, 2] (it may under-fill it).
        assert eig.min() >= -1e-4
        assert 0.0 < eig.max().item() <= 2.0 + 1e-4

    def test_spmm_matches_dense_for_sparse_operator(self):
        laplacian = _sym_psd(5, 1)
        h = torch.randn(5, 3)
        assert torch.allclose(
            spmm(laplacian, h), spmm(laplacian.to_sparse(), h), atol=1e-5
        )

    def test_spectral_upper_bound_bounds_lambda_max(self):
        """The Gershgorin bound is >= the true largest eigenvalue and is
        deterministic (identical across calls, no RNG)."""
        laplacian = _sym_psd(8, 5)
        true = torch.linalg.eigvalsh(laplacian).max()
        b1 = spectral_upper_bound(laplacian)
        b2 = spectral_upper_bound(laplacian)
        assert b1 >= true - 1e-6
        assert b1 == b2

    def test_forward_on_sparse_laplacians(self):
        """Forward + backward run on sparse COO Laplacians (the layout the
        real pipeline hands the backbone)."""
        x_all, laplacian_all = _dummy_complex()
        sparse_lap = tuple(L.to_sparse() for L in laplacian_all)
        m = PolynomialFilterTNN(
            4, 8, 3, K=4, basis=Chebyshev(), laplacian_norm="rescale"
        )
        ys = m(x_all, sparse_lap)
        assert [tuple(y.shape) for y in ys] == [(5, 3), (7, 3), (3, 3)]
        assert all(torch.isfinite(y).all() for y in ys)
        sum(y.sum() for y in ys).backward()
        assert all(t.grad is not None for t in m.theta)

    def test_empty_rank_no_triangles(self):
        """A complex with no triangles (rank-2 empty) filters cleanly."""
        x_all = (torch.randn(5, 4), torch.randn(7, 4), torch.zeros(0, 4))
        laplacian_all = (_sym_psd(5, 1), _sym_psd(7, 2), torch.zeros(0, 0))
        m = PolynomialFilterTNN(
            4, 8, 3, K=4, basis=Chebyshev(), laplacian_norm="rescale"
        )
        ys = m(x_all, laplacian_all)
        assert [tuple(y.shape) for y in ys] == [(5, 3), (7, 3), (0, 3)]
        assert all(torch.isfinite(y).all() for y in ys)

    def test_invalid_laplacian_norm_raises(self):
        with pytest.raises(ValueError, match="laplacian_norm"):
            PolynomialFilterTNN(
                4, 4, 2, K=3, basis=Monomial(), laplacian_norm="bogus"
            )
