"""Unit tests for ``HOPSEWrapper``."""

import torch
from torch_geometric.data import Data

from topobench.nn.wrappers.combinatorial.hopse_wrapper import HOPSEWrapper


class _DummyHOPSEBackbone(torch.nn.Module):
    """Identity-like backbone matching the HOPSE I/O contract.

    ``HOPSE.forward`` returns a tuple-of-tuples shape
    ``(complex_dim, max_hop)`` of tensors. This mock simply echoes the input.
    """

    def forward(self, x_all):
        return x_all


def _make_batch(complex_dim=3, max_hop=3, n_per_dim=(4, 5, 3), feat_dim=6):
    """Build a HOPSE-ready ``Data`` object."""
    fields = {}
    for i in range(complex_dim + 1):
        # HOPSEWrapper uses dims [0..complex_dim] but only routes to backbone
        # for dims [0..complex_dim-1]; we populate batch_{i} for the readout
        # too.
        n = n_per_dim[i] if i < len(n_per_dim) else 2
        fields[f"batch_{i}"] = torch.zeros(n, dtype=torch.long)
        for j in range(max_hop):
            fields[f"x{i}_{j}"] = torch.randn(n, feat_dim)
    fields["y"] = torch.tensor([0])
    return Data(**fields)


def test_hopse_wrapper_routes_features_and_preserves_shapes():
    """Wrapper produces ``x{i}_{j}`` outputs and ``batch_{i}`` per dimension."""
    complex_dim, max_hop = 3, 3
    feat_dim = 6
    n_per_dim = (4, 5, 3, 2)
    batch = _make_batch(
        complex_dim=complex_dim,
        max_hop=max_hop,
        n_per_dim=n_per_dim,
        feat_dim=feat_dim,
    )
    wrapper = HOPSEWrapper(
        backbone=_DummyHOPSEBackbone(),
        out_channels=feat_dim,
        num_cell_dimensions=complex_dim + 1,
        complex_dim=complex_dim,
        max_hop=max_hop,
        residual_connections=False,
    )
    out = wrapper(batch)
    assert "labels" in out
    assert torch.equal(out["labels"], batch.y)
    for i in range(complex_dim + 1):
        assert f"batch_{i}" in out
        for j in range(max_hop):
            assert f"x{i}_{j}" in out
            assert out[f"x{i}_{j}"].shape == (n_per_dim[i], feat_dim)


def test_hopse_wrapper_call_equivalent_to_forward():
    """``__call__`` should just delegate to ``forward``."""
    complex_dim, max_hop = 2, 2
    feat_dim = 4
    batch = _make_batch(
        complex_dim=complex_dim,
        max_hop=max_hop,
        n_per_dim=(3, 4, 2),
        feat_dim=feat_dim,
    )
    wrapper = HOPSEWrapper(
        backbone=_DummyHOPSEBackbone(),
        out_channels=feat_dim,
        num_cell_dimensions=complex_dim + 1,
        complex_dim=complex_dim,
        max_hop=max_hop,
        residual_connections=False,
    )
    call_out = wrapper(batch)
    fwd_out = wrapper.forward(batch)
    for k, v in fwd_out.items():
        assert torch.equal(call_out[k], v) if isinstance(v, torch.Tensor) else (
            call_out[k] == v
        )
