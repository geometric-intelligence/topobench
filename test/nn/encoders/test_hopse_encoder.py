"""Unit tests for the HOPSE feature encoder and its OGB helpers."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.encoders.hopse_encoder import (
    HOPSEFeatureEncoder,
    SimpleAtomEncoder,
    SimpleBondEncoder,
)


def _make_data(n_nodes: int = 5, n_edges: int = 6, in_dim: int = 4, hops: int = 3, dims=(0, 1, 2)):
    """Build a tiny ``Data`` object with HOPSE-style features.

    ``HOPSEFeatureEncoder`` expects per-(dim, hop) tensors named
    ``x{dim}_{hop}`` plus a ``batch_{dim}`` assignment for each dim.
    """
    counts = {0: n_nodes, 1: n_edges, 2: max(1, n_edges // 2)}
    fields = {}
    for d in dims:
        n = counts[d]
        fields[f"batch_{d}"] = torch.zeros(n, dtype=torch.long)
        for h in range(hops):
            fields[f"x{d}_{h}"] = torch.randn(n, in_dim)
    return Data(**fields)


class TestSimpleAtomEncoder:
    """``SimpleAtomEncoder`` should wrap OGB's ``AtomEncoder``."""

    def test_forward_returns_correct_shape(self):
        enc = SimpleAtomEncoder(in_channels=16)
        # OGB's AtomEncoder expects 9 atom features per node.
        x = torch.zeros((4, 9), dtype=torch.long)
        out = enc(x, batch=torch.zeros(4, dtype=torch.long))
        assert out.shape == (4, 16)

    def test_forward_casts_to_long(self):
        """Float input should be cast to ``long`` internally."""
        enc = SimpleAtomEncoder(in_channels=8)
        x = torch.zeros((2, 9), dtype=torch.float)
        out = enc(x, batch=torch.zeros(2, dtype=torch.long))
        assert out.shape == (2, 8)


class TestSimpleBondEncoder:
    """``SimpleBondEncoder`` should wrap OGB's ``BondEncoder``."""

    def test_forward_returns_correct_shape(self):
        enc = SimpleBondEncoder(in_channels=12)
        # OGB's BondEncoder expects 3 bond features per edge.
        x = torch.zeros((5, 3), dtype=torch.long)
        out = enc(x, batch=torch.zeros(5, dtype=torch.long))
        assert out.shape == (5, 12)


class TestHOPSEFeatureEncoder:
    """``HOPSEFeatureEncoder`` should encode per-(dim, hop) features."""

    def test_repr(self):
        in_channels = [[4, 4, 4], [4, 4, 4], [4, 4, 4]]
        enc = HOPSEFeatureEncoder(
            in_channels=in_channels,
            out_channels=8,
            max_hop=3,
            selected_dimensions=[0, 1, 2],
        )
        text = repr(enc)
        assert "HOPSEFeatureEncoder" in text
        assert "dimensions" in text

    def test_default_selected_dimensions_uses_in_channels(self):
        """When ``selected_dimensions`` is ``None`` it defaults to range(len)."""
        in_channels = [[4, 4, 4], [4, 4, 4]]
        enc = HOPSEFeatureEncoder(
            in_channels=in_channels,
            out_channels=8,
            max_hop=3,
            selected_dimensions=None,
        )
        assert list(enc.dimensions) == [0, 1]

    def test_forward_no_atom_or_bond_encoder(self):
        """Forward updates ``x{i}_{j}`` features to ``out_channels`` dim."""
        in_dim, out_dim, hops = 4, 6, 3
        dims = [0, 1, 2]
        in_channels = [[in_dim] * hops for _ in dims]
        enc = HOPSEFeatureEncoder(
            in_channels=in_channels,
            out_channels=out_dim,
            max_hop=hops,
            selected_dimensions=dims,
        )
        data = _make_data(in_dim=in_dim, hops=hops, dims=dims)
        out = enc(data)
        for d in dims:
            for h in range(hops):
                assert out[f"x{d}_{h}"].shape[-1] == out_dim

    def test_forward_with_fuse_pse2cell(self):
        """``fuse_pse2cell=True`` should also populate ``x_{i}`` keys."""
        in_dim, out_dim, hops = 4, 6, 2
        dims = [0, 1]
        in_channels = [[in_dim] * hops for _ in dims]
        enc = HOPSEFeatureEncoder(
            in_channels=in_channels,
            out_channels=out_dim,
            max_hop=hops,
            selected_dimensions=dims,
            fuse_pse2cell=True,
        )
        data = _make_data(in_dim=in_dim, hops=hops, dims=dims)
        out = enc(data)
        for d in dims:
            assert out[f"x_{d}"].shape == (out[f"x{d}_0"].shape[0], out_dim)

    def test_use_atom_encoder_swaps_dim0_hop0(self):
        """``use_atom_encoder=True`` installs ``SimpleAtomEncoder`` at (0, 0)."""
        in_dim, out_dim, hops = 4, 6, 2
        dims = [0]
        in_channels = [[in_dim] * hops]
        enc = HOPSEFeatureEncoder(
            in_channels=in_channels,
            out_channels=out_dim,
            max_hop=hops,
            selected_dimensions=dims,
            use_atom_encoder=True,
        )
        assert isinstance(enc.encoder_0_0, SimpleAtomEncoder)

    def test_use_bond_encoder_swaps_dim1_hop0(self):
        """``use_bond_encoder=True`` installs ``SimpleBondEncoder`` at (1, 0)."""
        in_dim, out_dim, hops = 4, 6, 2
        dims = [0, 1]
        in_channels = [[in_dim] * hops for _ in dims]
        enc = HOPSEFeatureEncoder(
            in_channels=in_channels,
            out_channels=out_dim,
            max_hop=hops,
            selected_dimensions=dims,
            use_bond_encoder=True,
        )
        assert isinstance(enc.encoder_1_0, SimpleBondEncoder)
