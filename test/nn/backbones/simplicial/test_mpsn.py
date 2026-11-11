"""Unit tests for MPSN."""

import pytest
import torch

from topobench.nn.backbones.simplicial import MPSN
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)

from ...._utils.nn_module_auto_test import NNModuleAutoTest


def test_MPSN(simple_graph_1):
    """Test the MPSN backbone end-to-end on a lifted graph.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    lifting_signed = SimplicialCliqueLifting(complex_dim=3, signed=True)
    data = lifting_signed(simple_graph_1)
    out_dim = 4
    n_layers = 2

    x_all = (data.x, data.x_1, data.x_2)
    incidence_all = (data.incidence_1, data.incidence_2)
    in_channels_all = (
        data.x.shape[1],
        data.x_1.shape[1],
        data.x_2.shape[1],
    )
    expected_shapes = [
        (data.x.shape[0], out_dim),
        (data.x_1.shape[0], out_dim),
        (data.x_2.shape[0], out_dim),
    ]

    auto_test = NNModuleAutoTest(
        [
            {
                "module": MPSN,
                "init": (in_channels_all, out_dim, n_layers),
                "forward": (x_all, incidence_all),
                "assert_shape": expected_shapes,
            },
        ]
    )
    auto_test.run()


@pytest.fixture
def two_triangles():
    """Two triangles {0,1,2} and {1,2,3} sharing the edge (1,2).

    Returns
    -------
    dict
        Feature banks and incidence matrices of the complex.
    """
    # edges: e0=(0,1) e1=(0,2) e2=(1,2) e3=(1,3) e4=(2,3)
    inc_1 = torch.tensor(
        [
            [-1.0, -1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, -1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    ).to_sparse_coo()
    # faces: t0={e0,e1,e2} t1={e2,e3,e4}
    inc_2 = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    ).to_sparse_coo()
    return {
        "x_0": torch.randn(4, 6),
        "x_1": torch.randn(5, 6),
        "x_2": torch.randn(2, 6),
        "incidence_all": (inc_1, inc_2),
    }


def test_mpsn_initialization():
    """Test basic initialization of MPSN."""
    model = MPSN(
        in_channels_all=(3, 4, 5),
        hidden_channels=6,
        n_layers=2,
    )
    assert model is not None
    assert len(model.layers) == 2
    assert hasattr(model, "in_linear_0")
    assert hasattr(model, "in_linear_1")
    assert hasattr(model, "in_linear_2")

    with pytest.raises(AssertionError):
        MPSN(in_channels_all=(3, 4, 5), hidden_channels=6, n_layers=0)


def test_mpsn_forward_shapes(two_triangles):
    """Test output shapes on a concrete complex.

    Parameters
    ----------
    two_triangles : dict
        Sample complex for testing.
    """
    data = two_triangles
    hidden = 8
    model = MPSN(
        in_channels_all=(6, 6, 6), hidden_channels=hidden, n_layers=2
    )
    out = model(
        (data["x_0"], data["x_1"], data["x_2"]), data["incidence_all"]
    )
    assert out[0].shape == (4, hidden)
    assert out[1].shape == (5, hidden)
    assert out[2].shape == (2, hidden)
    assert all(torch.isfinite(o).all() for o in out)


@pytest.mark.parametrize(
    "flags",
    [
        {"use_coboundary": False},
        {"use_lower": False},
        {"use_upper": False},
        {"use_boundary": False},
        {
            "use_boundary": True,
            "use_coboundary": False,
            "use_lower": False,
            "use_upper": True,
        },
    ],
)
def test_mpsn_adjacency_ablations(two_triangles, flags):
    """Test that each MPSN adjacency can be independently ablated.

    Parameters
    ----------
    two_triangles : dict
        Sample complex for testing.
    flags : dict
        Adjacency toggles passed to the model.
    """
    data = two_triangles
    model = MPSN(
        in_channels_all=(6, 6, 6), hidden_channels=8, n_layers=1, **flags
    )
    out = model(
        (data["x_0"], data["x_1"], data["x_2"]), data["incidence_all"]
    )
    assert all(torch.isfinite(o).all() for o in out)


def test_mpsn_no_faces():
    """Test that a complex without faces (empty rank 2) is handled."""
    num_nodes, num_edges = 5, 4
    x_0 = torch.randn(num_nodes, 6)
    x_1 = torch.randn(num_edges, 6)
    x_2 = torch.zeros(0, 6)
    incidence_1 = torch.zeros(num_nodes, num_edges).to_sparse_coo()
    incidence_2 = torch.zeros(num_edges, 0).to_sparse_coo()

    model = MPSN(in_channels_all=(6, 6, 6), hidden_channels=8, n_layers=1)
    out = model((x_0, x_1, x_2), (incidence_1, incidence_2))
    assert out[0].shape == (num_nodes, 8)
    assert out[2].shape == (0, 8)


def test_mpsn_gradient_flow(two_triangles):
    """Test that gradients propagate back to the inputs.

    Parameters
    ----------
    two_triangles : dict
        Sample complex for testing.
    """
    data = two_triangles
    x_0 = data["x_0"].clone().requires_grad_(True)
    model = MPSN(in_channels_all=(6, 6, 6), hidden_channels=8, n_layers=2)
    out = model((x_0, data["x_1"], data["x_2"]), data["incidence_all"])
    loss = sum(o.pow(2).sum() for o in out)
    loss.backward()
    assert x_0.grad is not None
    assert torch.isfinite(x_0.grad).all()

def test_mpsn_wrapper_forward():
    """Test the MPSNWrapper end-to-end on a tiny simplicial batch."""
    import torch
    import torch_geometric.data as tg_data

    from topobench.nn.wrappers.simplicial.mpsn_wrapper import MPSNWrapper

    torch.manual_seed(0)
    # triangle (0,1,2) plus edge (2,3): 4 nodes, 4 edges, 1 triangle
    i1 = torch.tensor([[0, 1, 0, 2, 1, 2, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]])
    v1 = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    inc1 = torch.sparse_coo_tensor(i1, v1, (4, 4)).coalesce()
    i2 = torch.tensor([[0, 1, 2], [0, 0, 0]])
    v2 = torch.tensor([1.0, -1.0, 1.0])
    inc2 = torch.sparse_coo_tensor(i2, v2, (4, 1)).coalesce()
    h = 8
    backbone = MPSN((h, h, h), h, n_layers=1)
    wrapper = MPSNWrapper(backbone, out_channels=h, num_cell_dimensions=3)
    batch = tg_data.Data(
        x_0=torch.randn(4, h), x_1=torch.randn(4, h), x_2=torch.randn(1, h),
        incidence_1=inc1, incidence_2=inc2,
        y=torch.zeros(4, dtype=torch.long),
        batch_0=torch.zeros(4, dtype=torch.long),
    )
    out = wrapper(batch)
    assert out["x_0"].shape == (4, h)
    assert out["x_1"].shape == (4, h)
    assert out["x_2"].shape == (1, h)
    assert torch.isfinite(out["x_0"]).all()
