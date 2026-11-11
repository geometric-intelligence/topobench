"""Tests for the AirTNN backbone and wrapper."""

import pytest
import torch
import torch_geometric.data as tg_data

from topobench.nn.backbones.cell.airtnn import AirTF, AirTNN, _pattern
from topobench.nn.wrappers.cell.airtnn_wrapper import AirTNNWrapper


@pytest.fixture
def toy_complex():
    """Build a tiny 2-complex: triangle (0,1,2) plus edge (2,3).

    Returns
    -------
    dict
        Edge signal, sparse rank-1 down/up Laplacian surrogates and
        incidence_1, for 4 edges: e0=(0,1), e1=(0,2), e2=(1,2), e3=(2,3).
    """
    torch.manual_seed(0)
    n1 = 4
    # lower neighbours (share a vertex) — symmetric off-diagonal pattern
    low = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
           (1, 3), (3, 1), (2, 3), (3, 2)]
    li = torch.tensor(low).t()
    Ld = torch.sparse_coo_tensor(li, torch.ones(li.shape[1]), (n1, n1))
    # upper neighbours (share the triangle): e0, e1, e2 pairwise
    up = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    ui = torch.tensor(up).t()
    Lu = torch.sparse_coo_tensor(ui, torch.ones(ui.shape[1]), (n1, n1))
    i1 = torch.tensor([[0, 1, 0, 2, 1, 2, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]])
    v1 = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    inc1 = torch.sparse_coo_tensor(i1, v1, (4, 4)).coalesce()
    return {"x1": torch.randn(n1, 6), "Ld": Ld.coalesce(),
            "Lu": Lu.coalesce(), "inc1": inc1}


def test_initialization():
    """Test constructor: per-order weight banks for both neighbourhoods."""
    m = AirTNN(in_channels=6, n_layers=3, filter_order=2)
    assert len(m.layers) == 3
    assert m.layers[0].w_down.shape == (3, 6, 6)
    assert m.layers[0].w_up.shape == (3, 6, 6)


def test_forward_shape(toy_complex):
    """Test output shape on the toy complex.

    Parameters
    ----------
    toy_complex : dict
        A fixture providing the toy 2-complex.
    """
    m = AirTNN(in_channels=6, n_layers=2, filter_order=2, air_enabled=True)
    y = m(toy_complex["x1"], toy_complex["Ld"], toy_complex["Lu"])
    assert y.shape == toy_complex["x1"].shape
    assert torch.isfinite(y).all()


def test_ideal_reduction_matches_eq2(toy_complex):
    """Faithfulness: with ideal channels the AirTF equals Eq. (2) exactly.

    The single-layer, air-disabled filter output is compared against a
    manual dense computation of
    ``sum_p S_d^p x W_d[p] + sum_p S_u^p x W_u[p]``.

    Parameters
    ----------
    toy_complex : dict
        A fixture providing the toy 2-complex.
    """
    torch.manual_seed(1)
    f = AirTF(6, 6, filter_order=2, air_enabled=False)
    x = toy_complex["x1"]
    pd = _pattern(toy_complex["Ld"])
    pu = _pattern(toy_complex["Lu"])
    y = f(x, pd, pu)

    n = x.shape[0]
    Sd = torch.zeros(n, n)
    Sd[pd[0], pd[1]] = 1.0
    Su = torch.zeros(n, n)
    Su[pu[0], pu[1]] = 1.0
    ref = x @ f.w_down[0] + x @ f.w_up[0]
    xd, xu = x, x
    for p in range(1, 3):
        xd = Sd @ xd
        xu = Su @ xu
        ref = ref + xd @ f.w_down[p] + xu @ f.w_up[p]
    assert torch.allclose(y, ref, atol=1e-5)


def test_air_is_stochastic_and_seed_reproducible(toy_complex):
    """Channel sampling: two forwards differ; same seed reproduces.

    Parameters
    ----------
    toy_complex : dict
        A fixture providing the toy 2-complex.
    """
    m = AirTNN(in_channels=6, n_layers=1, filter_order=1,
               air_enabled=True, delta=1.0, snr_db=20.0)
    x, Ld, Lu = toy_complex["x1"], toy_complex["Ld"], toy_complex["Lu"]
    torch.manual_seed(7)
    y1 = m(x, Ld, Lu)
    y2 = m(x, Ld, Lu)
    assert not torch.allclose(y1, y2)
    torch.manual_seed(7)
    y3 = m(x, Ld, Lu)
    assert torch.allclose(y1, y3)


def test_filter_order_zero_is_pointwise(toy_complex):
    """P=0: no shifts occur, so the map is pointwise even with air on.

    Parameters
    ----------
    toy_complex : dict
        A fixture providing the toy 2-complex.
    """
    torch.manual_seed(2)
    f = AirTF(6, 6, filter_order=0, air_enabled=True)
    x = toy_complex["x1"]
    y = f(x, _pattern(toy_complex["Ld"]), _pattern(toy_complex["Lu"]))
    assert torch.allclose(y, x @ f.w_down[0] + x @ f.w_up[0], atol=1e-6)


def test_pattern_accepts_edge_index():
    """The pattern extractor accepts a dense [2, E] edge-index too."""
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    r, c = _pattern(ei)
    assert torch.equal(r, ei[0]) and torch.equal(c, ei[1])


def test_gradient_flow(toy_complex):
    """Gradients reach every filter weight through the air shifts.

    Parameters
    ----------
    toy_complex : dict
        A fixture providing the toy 2-complex.
    """
    torch.manual_seed(3)
    m = AirTNN(in_channels=6, n_layers=2, filter_order=2, air_enabled=True)
    y = m(toy_complex["x1"], toy_complex["Ld"], toy_complex["Lu"])
    y.pow(2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_wrapper_forward(toy_complex):
    """Wrapper: returns x_1 and pushes x_0 down via incidence_1.

    Parameters
    ----------
    toy_complex : dict
        A fixture providing the toy 2-complex.
    """
    torch.manual_seed(4)
    backbone = AirTNN(in_channels=6, n_layers=1, filter_order=1,
                      air_enabled=False)
    wrapper = AirTNNWrapper(backbone, out_channels=6, num_cell_dimensions=2)
    batch = tg_data.Data(
        x_1=toy_complex["x1"],
        down_laplacian_1=toy_complex["Ld"],
        up_laplacian_1=toy_complex["Lu"],
        incidence_1=toy_complex["inc1"],
        y=torch.zeros(4, dtype=torch.long),
        batch_0=torch.zeros(4, dtype=torch.long),
    )
    out = wrapper(batch)
    assert out["x_1"].shape == (4, 6)
    assert out["x_0"].shape == (4, 6)
