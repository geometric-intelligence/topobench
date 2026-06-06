"""Tests for ``HOPSE_PE_Information`` and its module-level helpers."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.hopse_ps_information import (
    HOPSE_PE_Information,
    dotdict,
    interrank_boundary_index,
)
from topobench.data.utils.utils import load_manual_simplicial_complex


def _base_kwargs(**overrides):
    """Minimum kwargs needed to instantiate ``HOPSE_PE_Information``."""
    kwargs = {
        "max_rank": 2,
        "copy_initial": True,
        "neighborhoods": ["up_laplacian_0"],
        "encodings": [],  # empty -> no PSE/FE transforms to construct
        "parameters": {"RWSE": {"max_pe_dim": 4, "concat_to_x": False}},
        "in_channels": {0: [4], 1: [4], 2: [4]},
        "device": "cpu",
        "cuda": [0],
        "dim_target_node": 8,
    }
    kwargs.update(overrides)
    return kwargs


class TestDotDict:
    """``dotdict`` exposes dict items as attributes."""

    def test_getattr_setattr(self):
        d = dotdict()
        d.foo = 1
        assert d["foo"] == 1
        assert d.foo == 1

    def test_missing_returns_none(self):
        d = dotdict()
        assert d.missing is None


class TestInterrankBoundaryIndex:
    """The module-level ``interrank_boundary_index`` helper."""

    def test_returns_shifted_edge_index_and_attr(self):
        x_src = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        )
        boundary_index = [
            [0, 1, 1, 2],  # node ids (dst)
            [0, 0, 1, 1],  # edge ids (src)
        ]
        edge_index, edge_attr = interrank_boundary_index(
            x_src=x_src,
            boundary_index=boundary_index,
            n_dst_nodes=4,
        )
        # The src ids get shifted by ``n_dst_nodes``.
        assert edge_index.shape == (2, 4)
        assert torch.equal(edge_index[0], torch.tensor([0, 1, 1, 2]))
        assert torch.equal(edge_index[1], torch.tensor([4, 4, 5, 5]))
        # Edge attr is the source feature at the corresponding edge id.
        expected_attr = x_src[
            torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        ].squeeze()
        assert torch.equal(edge_attr, expected_attr)

    def test_accepts_tensor_boundary_index(self):
        x_src = torch.arange(6, dtype=torch.float).reshape(3, 2)
        bidx = [
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 2]),
        ]
        edge_index, _ = interrank_boundary_index(
            x_src=x_src, boundary_index=bidx, n_dst_nodes=3
        )
        assert edge_index.shape == (2, 3)


class TestHOPSEPEInformationInit:
    """Basic constructor + helper coverage."""

    def test_init_stores_kwargs(self):
        t = HOPSE_PE_Information(**_base_kwargs())
        assert t.max_rank == 2
        assert t.device == "cpu"
        assert t.num_pe_considered == 0
        assert t.hidden_dim == 8

    def test_repr_contains_class_name(self):
        t = HOPSE_PE_Information(**_base_kwargs())
        text = repr(t)
        assert "HOPSE_PE_Information" in text

    def test_data_to_device_returns_data(self):
        t = HOPSE_PE_Information(**_base_kwargs())
        d = Data(x=torch.randn(2, 3), some_str="foo")
        moved = t._data_to_device(d)
        assert isinstance(moved, Data)
        assert torch.equal(moved.x, d.x)
        # Non-tensor attributes are forwarded as-is.
        assert moved.some_str == "foo"

    def test_make_zero_encoding_data_shapes(self):
        t = HOPSE_PE_Information(**_base_kwargs())
        d = t._make_zero_encoding_data(
            n_cells=5, encodings=["LapPE", "RWSE"], dims=[3, 4]
        )
        assert d.LapPE.shape == (5, 3)
        assert d.RWSE.shape == (5, 4)
        assert torch.all(d.LapPE == 0.0)

    def test_make_zero_data_for_all_encodings(self):
        t = HOPSE_PE_Information(**_base_kwargs(encodings=["LapPE", "RWSE"]))
        d = t._make_zero_data_for_all_encodings(n_cells=2, dims=[3, 5])
        assert d.LapPE.shape == (2, 3)
        assert d.RWSE.shape == (2, 5)

    def test_aggregate_inter_nbhd_concatenates_encodings(self):
        t = HOPSE_PE_Information(**_base_kwargs(encodings=["LapPE"]))
        t.routes = [(0, 0), (1, 0), (0, 0)]
        # Three GNN outputs targeting the same destination rank 0; the
        # encoding tensor should be concatenated along dim=1 each time.
        x_out_per_route = [
            Data(LapPE=torch.ones(3, 2)),
            Data(LapPE=torch.full((3, 4), 2.0)),
            Data(LapPE=torch.full((3, 1), 3.0)),
        ]
        out = t.aggregate_inter_nbhd(x_out_per_route)
        assert 0 in out
        assert out[0].LapPE.shape == (3, 2 + 4 + 1)


class TestHOPSEPEInformationForward:
    """Functional tests for the forward method."""

    def test_forward_basic(self):
        data = load_manual_simplicial_complex()
        data["adjacency-0"] = data.adjacency_0
        data["down_incidence-1"] = data.incidence_1

        kwargs = _base_kwargs(
            neighborhoods=["adjacency-0", "down_incidence-1"],
            encodings=["RWSE"],
            parameters={"RWSE": {"max_pe_dim": 4, "concat_to_x": False}},
            in_channels={0: [2, 8], 1: [2, 4], 2: [2, 4]},
            dim_all_encodings=[4],
        )
        transform = HOPSE_PE_Information(**kwargs)
        out_data = transform(data)

        assert hasattr(out_data, "x0_0")
        assert hasattr(out_data, "x0_1")
        assert out_data.x0_0.shape == (5, 2)
        assert out_data.x0_1.shape == (5, 8)

    def test_forward_no_copy_initial(self):
        data = load_manual_simplicial_complex()
        data["adjacency-0"] = data.adjacency_0

        kwargs = _base_kwargs(
            copy_initial=False,
            neighborhoods=["adjacency-0"],
            encodings=["RWSE"],
            parameters={"RWSE": {"max_pe_dim": 4, "concat_to_x": False}},
            in_channels={0: [4], 1: [4], 2: [4]},
            dim_all_encodings=[4],
        )
        transform = HOPSE_PE_Information(**kwargs)
        out_data = transform(data)

        assert hasattr(out_data, "x0_0")
        assert out_data.x0_0.shape == (5, 4)

    def test_forward_empty_rank(self):
        data = Data(
            x_0=torch.randn(5, 2), x_1=torch.randn(0, 2), x_2=torch.randn(0, 2)
        )
        data["adjacency-0"] = torch.zeros((5, 5)).to_sparse()

        kwargs = _base_kwargs(
            neighborhoods=["adjacency-0"],
            encodings=["RWSE"],
            in_channels={0: [2, 4], 1: [2, 4], 2: [2, 4]},
            dim_all_encodings=[4],
        )
        transform = HOPSE_PE_Information(**kwargs)
        out_data = transform(data)

        assert out_data.x1_1.shape == (0, 4)
        assert out_data.x2_1.shape == (0, 4)

    def test_forward_single_node(self):
        data = Data(
            x_0=torch.randn(1, 2), x_1=torch.randn(0, 2), x_2=torch.randn(0, 2)
        )
        data["adjacency-0"] = torch.zeros((1, 1)).to_sparse()

        kwargs = _base_kwargs(
            neighborhoods=["adjacency-0"],
            encodings=["RWSE"],
            in_channels={0: [2, 4], 1: [2, 4], 2: [2, 4]},
            dim_all_encodings=[4],
        )
        transform = HOPSE_PE_Information(**kwargs)
        out_data = transform(data)

        assert out_data.x0_1.shape == (1, 4)
        assert torch.all(out_data.x0_1 == 0.0)

    def test_interrank_padding(self):
        data = Data(x_0=torch.randn(5, 2), x_1=torch.randn(3, 4))
        indices = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]])
        data["down_incidence-1"] = torch.sparse_coo_tensor(
            indices, torch.ones(6), (5, 3)
        )

        kwargs = _base_kwargs(
            max_rank=1,
            neighborhoods=["down_incidence-1"],
            encodings=["RWSE"],
            in_channels={0: [2, 4], 1: [4, 4]},
            dim_all_encodings=[4],
        )
        transform = HOPSE_PE_Information(**kwargs)
        out_data = transform(data)

        assert hasattr(out_data, "x0_1")
        assert out_data.x0_1.shape == (5, 4)


def test_init_requires_encodings_supported_when_nonempty():
    """Unsupported encodings raise via ``CombinedEncodings``."""
    with pytest.raises(ValueError, match="Unsupported encoding"):
        HOPSE_PE_Information(**_base_kwargs(encodings=["NotAnEncoding"]))
