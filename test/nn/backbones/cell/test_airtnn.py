"""Unit tests for the AirTNN cell-complex backbone."""

import pytest
import torch

from topobench.nn.backbones.cell.airtnn import AirTNN, AirTNNLayer


def _random_sparse_laplacian(n, density=0.3, seed=0):
    """Build a small symmetric coalesced sparse matrix to stand in for a Laplacian.

    Parameters
    ----------
    n : int
        Number of cells (matrix is ``[n, n]``).
    density : float, optional
        Approximate fraction of off-diagonal nonzeros (default: 0.3).
    seed : int, optional
        RNG seed for reproducibility (default: 0).

    Returns
    -------
    torch.Tensor
        Coalesced sparse ``[n, n]`` float32 tensor.
    """
    g = torch.Generator().manual_seed(seed)
    dense = (torch.rand(n, n, generator=g) < density).float()
    dense = torch.triu(dense, diagonal=1)
    dense = dense + dense.t()
    dense.fill_diagonal_(1.0)
    return dense.to_sparse_coo().coalesce()


@pytest.fixture
def toy_complex():
    """Return ``(x, Ld, Lu)`` for a small synthetic rank-1 signal.

    Returns
    -------
    tuple of torch.Tensor
        Signal ``x`` of shape ``[n, f]`` and two ``[n, n]`` sparse Laplacians.
    """
    n, f = 12, 8
    x = torch.randn(n, f)
    return x, _random_sparse_laplacian(n, seed=1), _random_sparse_laplacian(n, seed=2)


def test_layer_output_shape(toy_complex):
    """Layer maps ``[n, c_in]`` to ``[n, c_out]``.

    Parameters
    ----------
    toy_complex : tuple of torch.Tensor
        Signal and the up/down Laplacians from the fixture.
    """
    x, ld, lu = toy_complex
    layer = AirTNNLayer(c_in=8, c_out=16, k=1, snr_db=100)
    out = layer(x, ld, lu)
    assert out.shape == (12, 16)
    assert out.dtype == torch.float32


def test_backbone_output_shape(toy_complex):
    """Backbone preserves the channel dimension and cell count.

    Parameters
    ----------
    toy_complex : tuple of torch.Tensor
        Signal and the up/down Laplacians from the fixture.
    """
    x, ld, lu = toy_complex
    model = AirTNN(in_channels=8, n_layers=3, k=2, snr_db=100)
    out = model(x, ld, lu)
    assert out.shape == (12, 8)


def test_ideal_channel_is_deterministic(toy_complex):
    """With ``snr_db == 100`` the filter is noise-free and repeatable.

    Parameters
    ----------
    toy_complex : tuple of torch.Tensor
        Signal and the up/down Laplacians from the fixture.
    """
    x, ld, lu = toy_complex
    layer = AirTNNLayer(8, 8, k=2, snr_db=100).eval()
    torch.testing.assert_close(layer(x, ld, lu), layer(x, ld, lu))


def test_noisy_channel_is_stochastic(toy_complex):
    """With finite SNR, channel fading + noise make repeated passes differ.

    Parameters
    ----------
    toy_complex : tuple of torch.Tensor
        Signal and the up/down Laplacians from the fixture.
    """
    x, ld, lu = toy_complex
    layer = AirTNNLayer(8, 8, k=1, snr_db=10).eval()
    a, b = layer(x, ld, lu), layer(x, ld, lu)
    assert not torch.allclose(a, b)


def test_gradients_flow_to_linears(toy_complex):
    """Backprop populates gradients on every learnable map (ideal channel).

    Parameters
    ----------
    toy_complex : tuple of torch.Tensor
        Signal and the up/down Laplacians from the fixture.
    """
    x, ld, lu = toy_complex
    layer = AirTNNLayer(8, 8, k=1, snr_db=100)
    layer(x, ld, lu).pow(2).sum().backward()
    assert layer.h_lin.weight.grad is not None
    assert all(lin.weight.grad is not None for lin in layer.up_lins)
    assert all(lin.weight.grad is not None for lin in layer.low_lins)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_shift_order_parameterization(k):
    """A layer of order ``k`` holds ``k + 1`` maps per neighborhood.

    Parameters
    ----------
    k : int
        Shift order under test.
    """
    layer = AirTNNLayer(4, 4, k=k, snr_db=100)
    assert len(layer.up_lins) == k + 1
    assert len(layer.low_lins) == k + 1


def test_output_is_finite(toy_complex):
    """Noisy forward pass produces no NaNs/Infs (mirrors the upstream guard).

    Parameters
    ----------
    toy_complex : tuple of torch.Tensor
        Signal and the up/down Laplacians from the fixture.
    """
    x, ld, lu = toy_complex
    out = AirTNN(8, n_layers=2, k=2, snr_db=10)(x, ld, lu)
    assert torch.all(torch.isfinite(out))
