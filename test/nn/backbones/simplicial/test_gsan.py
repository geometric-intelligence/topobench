"""Unit tests for the GSAN backbone."""

import pytest
import torch

from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.simplicial import GSAN
from topobench.nn.backbones.simplicial.gsan import GSANLayer, _to_dense
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def _operators(data):
    """Assemble the (laplacian_all, incidence_all) tuples from lifted data.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Clique-lifted simplicial complex.

    Returns
    -------
    tuple
        ``(laplacian_all, incidence_all)`` as expected by the backbone.
    """
    laplacian_all = (
        data.hodge_laplacian_0,
        data.down_laplacian_1,
        data.up_laplacian_1,
        data.down_laplacian_2,
        data.up_laplacian_2,
    )
    incidence_all = (data.incidence_1, data.incidence_2)
    return laplacian_all, incidence_all


def test_GSAN(simple_graph_1):
    """Test the GSAN backbone end to end on a lifted graph with triangles.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    data = SimplicialCliqueLifting(complex_dim=2, signed=True)(simple_graph_1)
    out_dim = 4
    laplacian_all, incidence_all = _operators(data)
    x_all = (data.x_0, data.x_1, data.x_2)
    in_dims = (data.x_0.shape[1], data.x_1.shape[1], data.x_2.shape[1])
    expected = [
        (data.x_0.shape[0], out_dim),
        (data.x_1.shape[0], out_dim),
        (data.x_2.shape[0], out_dim),
    ]
    NNModuleAutoTest(
        [
            {
                "module": GSAN,
                "init": (in_dims, (out_dim, out_dim, out_dim), 2),
                "forward": (x_all, laplacian_all, incidence_all),
                "assert_shape": expected,
            }
        ]
    ).run()


def test_forward_shapes_with_triangles(simple_graph_1):
    """Test GSAN output shapes on a complex that contains triangles.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    data = SimplicialCliqueLifting(complex_dim=2, signed=True)(simple_graph_1)
    laplacian_all, incidence_all = _operators(data)
    model = GSAN(
        in_channels_all=(
            data.x_0.shape[1],
            data.x_1.shape[1],
            data.x_2.shape[1],
        ),
        hidden_channels_all=(8, 8, 8),
        K=2,
        n_layers=2,
    )
    x0, x1, x2 = model(
        (data.x_0, data.x_1, data.x_2), laplacian_all, incidence_all
    )
    assert x0.shape == (data.x_0.shape[0], 8)
    assert x1.shape == (data.x_1.shape[0], 8)
    assert x2.shape == (data.x_2.shape[0], 8)
    assert data.x_2.shape[0] > 0  # the tetrahedron yields triangles
    for out in (x0, x1, x2):
        assert torch.isfinite(out).all()


def test_no_triangle_graph_runs(simple_graph_0):
    """Test GSAN on a graph whose clique complex has no triangles.

    Parameters
    ----------
    simple_graph_0 : torch_geometric.data.Data
        A simple graph fixture without 3-cliques.
    """
    data = SimplicialCliqueLifting(complex_dim=2, signed=True)(simple_graph_0)
    laplacian_all, incidence_all = _operators(data)
    model = GSAN(
        in_channels_all=(
            data.x_0.shape[1],
            data.x_1.shape[1],
            data.x_2.shape[1],
        ),
        hidden_channels_all=(8, 8, 8),
        K=2,
    )
    x0, x1, x2 = model(
        (data.x_0, data.x_1, data.x_2), laplacian_all, incidence_all
    )
    assert x0.shape == (data.x_0.shape[0], 8)
    # No triangles -> empty rank-2 signal, and finite node/edge embeddings.
    assert x2.shape[0] == 0
    assert torch.isfinite(x0).all()
    assert torch.isfinite(x1).all()


def test_single_tap(simple_graph_1):
    """Test GSAN with a single polynomial tap (K=1).

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    data = SimplicialCliqueLifting(complex_dim=2, signed=True)(simple_graph_1)
    laplacian_all, incidence_all = _operators(data)
    model = GSAN(
        in_channels_all=(
            data.x_0.shape[1],
            data.x_1.shape[1],
            data.x_2.shape[1],
        ),
        hidden_channels_all=(8, 8, 8),
        K=1,
    )
    x0, _, _ = model(
        (data.x_0, data.x_1, data.x_2), laplacian_all, incidence_all
    )
    assert x0.shape == (data.x_0.shape[0], 8)


@pytest.mark.parametrize("update_func", ["sigmoid", "relu", None])
def test_update_functions(simple_graph_1, update_func):
    """Test the supported per-layer nonlinearities.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    update_func : str or None
        Nonlinearity variant to test.
    """
    data = SimplicialCliqueLifting(complex_dim=2, signed=True)(simple_graph_1)
    laplacian_all, incidence_all = _operators(data)
    model = GSAN(
        in_channels_all=(
            data.x_0.shape[1],
            data.x_1.shape[1],
            data.x_2.shape[1],
        ),
        hidden_channels_all=(8, 8, 8),
        update_func=update_func,
    )
    x0, _, _ = model(
        (data.x_0, data.x_1, data.x_2), laplacian_all, incidence_all
    )
    assert torch.isfinite(x0).all()


def test_invalid_K():
    """Test that a non-positive number of taps raises an assertion error."""
    with pytest.raises(AssertionError):
        GSAN((3, 3, 3), (8, 8, 8), K=0)


def test_invalid_n_layers():
    """Test that a non-positive number of layers raises an assertion error."""
    with pytest.raises(AssertionError):
        GSAN((3, 3, 3), (8, 8, 8), n_layers=0)


def test_mismatched_hidden_widths():
    """Test that unequal per-rank hidden widths raise an assertion error."""
    with pytest.raises(AssertionError):
        GSAN((3, 3, 3), (8, 16, 8))


def test_reset_parameters():
    """Test that reset_parameters runs and keeps parameters finite."""
    model = GSAN((3, 3, 3), (8, 8, 8))
    model.reset_parameters()
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_branch_empty_guard():
    """Test that a layer branch returns zeros for an all-zero operator."""
    layer = GSANLayer(
        channels=8, K=2, dropout=0.0, alpha_leaky_relu=0.2, update_func=None
    )
    x = torch.randn(5, 8)
    zero_lap = torch.zeros(5, 5)
    out = layer._branch(x, x, layer.W, layer.att["a00"], zero_lap)
    assert torch.equal(out, torch.zeros_like(x))


def test_to_dense_helper():
    """Test that _to_dense converts sparse tensors and passes dense through."""
    dense = torch.eye(3)
    assert torch.equal(_to_dense(dense), dense)
    assert torch.equal(_to_dense(dense.to_sparse_coo()), dense)
