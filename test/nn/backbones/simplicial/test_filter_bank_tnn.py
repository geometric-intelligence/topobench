"""Unit tests for FilterBankTNN (SCNN down/up filter bank on Hodge Laplacians)."""

import pytest
import torch

from topobench.nn.backbones.simplicial.filter_bank_tnn import FilterBankTNN


def _sym_psd(n: int, seed: int) -> torch.Tensor:
    """Small symmetric PSD matrix standing in for a Hodge (sub-)Laplacian."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g)
    return a @ a.T / n


def _dummy(f: int = 4):
    """Signals + down/up Laplacians; rank 0 has no lower simplices."""
    x_all = (torch.randn(5, f), torch.randn(7, f), torch.randn(3, f))
    down_all = (None, _sym_psd(7, 1), _sym_psd(3, 2))
    up_all = (_sym_psd(5, 3), _sym_psd(7, 4), _sym_psd(3, 5))
    return x_all, down_all, up_all


class TestFilterBankTNN:
    """Tests for the SCNN down/up filter bank."""

    def test_forward_shapes_per_rank(self):
        x_all, down_all, up_all = _dummy()
        m = FilterBankTNN(4, 8, 3, K=3)
        ys = m(x_all, down_all, up_all)
        assert [tuple(y.shape) for y in ys] == [(5, 3), (7, 3), (3, 3)]
        assert all(torch.isfinite(y).all() for y in ys)

    def test_two_channels_and_gamma_per_rank(self):
        m = FilterBankTNN(4, 8, 3, K=3, n_ranks=3)
        assert len(m.down) == 3 and len(m.up) == 3
        assert m.gamma.shape == (3, 2)

    def test_channels_have_learnable_theta(self):
        m = FilterBankTNN(4, 8, 3, K=4)
        for c in list(m.down) + list(m.up):
            assert c.theta.requires_grad and c.theta.shape == (5,)

    def test_rank0_down_channel_is_unused(self):
        """Rank 0 has no down operator, so its down channel never runs."""
        x_all, down_all, up_all = _dummy()  # down_all[0] is None
        m = FilterBankTNN(4, 8, 3, K=3)
        loss = sum(y.sum() for y in m(x_all, down_all, up_all))
        loss.backward()
        assert m.down[0].theta.grad is None  # skipped
        assert m.up[0].theta.grad is not None  # rank-0 up channel runs
        assert m.gamma.grad is not None and m.gamma.grad.abs().sum() > 0

    def test_rescale_runs(self):
        """The ``rescale`` path builds a [0,2] operator per down/up channel."""
        x_all, down_all, up_all = _dummy()
        m = FilterBankTNN(4, 8, 3, K=3, laplacian_norm="rescale")
        ys = m(x_all, down_all, up_all)
        assert all(torch.isfinite(y).all() for y in ys)

    def test_forward_on_sparse_laplacians(self):
        """Forward + backward on sparse COO down/up Laplacians (skipping the
        ``None`` rank-0 down)."""
        x_all, down_all, up_all = _dummy()
        to_sp = lambda L: None if L is None else L.to_sparse()  # noqa: E731
        down_s = tuple(to_sp(L) for L in down_all)
        up_s = tuple(to_sp(L) for L in up_all)
        m = FilterBankTNN(4, 8, 3, K=3, laplacian_norm="rescale")
        ys = m(x_all, down_s, up_s)
        assert [tuple(y.shape) for y in ys] == [(5, 3), (7, 3), (3, 3)]
        assert all(torch.isfinite(y).all() for y in ys)
        sum(y.sum() for y in ys).backward()
        assert m.up[0].theta.grad is not None

    def test_empty_rank_no_triangles(self):
        """A complex with no triangles (rank-2 empty) filters cleanly."""
        x_all = (torch.randn(5, 4), torch.randn(7, 4), torch.zeros(0, 4))
        down_all = (None, _sym_psd(7, 1), torch.zeros(0, 0))
        up_all = (_sym_psd(5, 3), _sym_psd(7, 4), torch.zeros(0, 0))
        m = FilterBankTNN(4, 8, 3, K=3)
        ys = m(x_all, down_all, up_all)
        assert [tuple(y.shape) for y in ys] == [(5, 3), (7, 3), (0, 3)]
        assert all(torch.isfinite(y).all() for y in ys)

    def test_invalid_laplacian_norm_raises(self):
        with pytest.raises(ValueError, match="laplacian_norm"):
            FilterBankTNN(4, 4, 2, K=3, laplacian_norm="bogus")
