"""Unit tests for COSIMO."""

import pytest
import torch

from topobench.nn.backbones.simplicial.cosimo import COSIMO, COSIMOLayer
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def _simplicial_inputs(graph):
    lifting = SimplicialCliqueLifting(complex_dim=3, signed=True)
    data = lifting(graph)
    x_all = (data.x, data.x_1, data.x_2)
    laplacian_all = (
        data.hodge_laplacian_0,
        data.down_laplacian_1,
        data.up_laplacian_1,
        data.down_laplacian_2,
        data.up_laplacian_2,
    )
    incidence_all = (data.incidence_1, data.incidence_2)
    return x_all, laplacian_all, incidence_all


def test_cosimo_forward_shapes(simple_graph_1):
    """Test COSIMO output shapes on a lifted clique complex."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(4, 4, 4),
        n_layers=2,
        t_init=0.5,
        num_branches=3,
        diffusion_method="taylor",
        taylor_order=4,
    )

    out = model(x_all, laplacian_all, incidence_all)

    assert len(out) == 3
    assert out[0].shape == (x_all[0].shape[0], 4)
    assert out[1].shape == (x_all[1].shape[0], 4)
    assert out[2].shape == (x_all[2].shape[0], 4)
    assert all(torch.isfinite(x).all() for x in out)


def test_cosimo_exact_diffusion_is_deterministic(simple_graph_1):
    """Test exact heat diffusion branch on a small complex."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    torch.manual_seed(0)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(3, 3, 3),
        n_layers=1,
        diffusion_method="exact",
        update_func=None,
    )

    out_1 = model(x_all, laplacian_all, incidence_all)
    out_2 = model(
        tuple(x.clone() for x in x_all), laplacian_all, incidence_all
    )

    for tensor_1, tensor_2 in zip(out_1, out_2, strict=True):
        assert torch.equal(tensor_1, tensor_2)


def test_cosimo_supports_rank_specific_hidden_channels(simple_graph_1):
    """Test COSIMO when hidden widths differ across simplex ranks."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(2, 3, 4),
        n_layers=2,
        num_branches=2,
    )

    out = model(x_all, laplacian_all, incidence_all)

    assert out[0].shape == (x_all[0].shape[0], 2)
    assert out[1].shape == (x_all[1].shape[0], 3)
    assert out[2].shape == (x_all[2].shape[0], 4)


def test_cosimo_accepts_four_laplacian_tuple(simple_graph_1):
    """Test fallback when only a lower 2-Laplacian is provided."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(2, 2, 2),
        n_layers=1,
    )

    out = model(x_all, laplacian_all[:4], incidence_all)

    assert out[2].shape == (x_all[2].shape[0], 2)


def test_cosimo_layer_uses_independent_branch_times():
    """Test that each COSIMO branch has its own learnable receptive fields."""
    layer = COSIMOLayer(
        in_channels=(2, 2, 2),
        out_channels=(2, 2, 2),
        num_branches=3,
    )

    assert layer.num_branches == 3
    assert len(layer.raw_times) == 3 * len(layer.branch_names())
    assert "0_x1_lower" in layer.raw_times
    assert "2_x1_lower" in layer.raw_times


def test_cosimo_taylor_diffusion_stabilizes_nonfinite_values():
    """Test finite guard for numerically unstable Taylor responses."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
        t_init=1.0,
        taylor_order=2,
        normalize_laplacian=False,
    )
    indices = torch.tensor([[0], [0]])
    values = torch.tensor([1.0e20])
    laplacian = torch.sparse_coo_tensor(indices, values, (1, 1))
    x = torch.tensor([[1.0e20]])

    out = layer.taylor_diffusion(laplacian, x, torch.tensor(1.0))

    assert torch.isfinite(out).all()
    assert out.abs().max() <= layer.max_abs_value


def test_cosimo_taylor_diffusion_can_disable_stabilization():
    """Test opt-out path for raw diffusion diagnostics."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
        t_init=1.0,
        taylor_order=2,
        stabilize=False,
        normalize_laplacian=False,
    )
    indices = torch.tensor([[0], [0]])
    values = torch.tensor([1.0e20])
    laplacian = torch.sparse_coo_tensor(indices, values, (1, 1))
    x = torch.tensor([[1.0e20]])

    out = layer.taylor_diffusion(laplacian, x, torch.tensor(1.0))

    assert not torch.isfinite(out).all()


def test_cosimo_diffusion_time_is_bounded():
    """Test learned diffusion times cannot exceed the configured cap."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
        t_init=1.0,
        max_diffusion_time=0.25,
    )
    layer.raw_times["0_x0_self"].data.fill_(100.0)

    time = layer.diffusion_time(0, "x0_self", torch.ones(1, 1))

    assert torch.isclose(time, torch.tensor(0.25))


def test_cosimo_normalizes_sparse_taylor_operator():
    """Test sparse diffusion operator row-sum normalization."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
        t_init=1.0,
        taylor_order=1,
    )
    indices = torch.tensor([[0, 0], [0, 1]])
    values = torch.tensor([2.0, -6.0])
    operator = torch.sparse_coo_tensor(indices, values, (2, 2))

    normalized = layer.normalize_operator(operator).coalesce()

    assert torch.isclose(normalized.values().abs().sum(), torch.tensor(1.0))


def test_cosimo_normalizes_dense_taylor_operator():
    """Test dense diffusion operator row-sum normalization."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
        t_init=1.0,
        taylor_order=1,
    )
    operator = torch.tensor([[2.0, -6.0], [1.0, 1.0]])

    normalized = layer.normalize_operator(operator)

    assert torch.isclose(normalized[0].abs().sum(), torch.tensor(1.0))


def test_cosimo_normalizes_empty_sparse_operator():
    """Test sparse normalization supports empty simplicial ranks."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
    )
    operator = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0,)),
        (0, 0),
    )

    normalized = layer.normalize_operator(operator)

    assert normalized.shape == operator.shape
    assert normalized._nnz() == 0


def test_cosimo_normalizes_empty_dense_operator():
    """Test dense normalization supports empty simplicial ranks."""
    layer = COSIMOLayer(
        in_channels=(1, 1, 1),
        out_channels=(1, 1, 1),
    )
    operator = torch.empty((0, 0))

    normalized = layer.normalize_operator(operator)

    assert normalized.shape == operator.shape


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"diffusion_method": "bad"}, "diffusion_method"),
        ({"update_func": "bad"}, "update_func"),
        ({"t_init": 0.0}, "t_init"),
        ({"num_branches": 0}, "num_branches"),
        ({"taylor_order": 0}, "taylor_order"),
        ({"max_diffusion_time": 0.0}, "max_diffusion_time"),
        ({"max_abs_value": 0.0}, "max_abs_value"),
    ],
)
def test_cosimo_layer_validation(kwargs, match):
    """Test validation of COSIMO layer options."""
    with pytest.raises(ValueError, match=match):
        COSIMOLayer(
            in_channels=(2, 2, 2),
            out_channels=(2, 2, 2),
            **kwargs,
        )
