"""Unit tests for the MPSN model wrapper.

The wrapper unpacks the lifted batch (``x_0/x_1/x_2``, ``incidence_1/2``,
``batch_0``, ``y``), calls the MPSN backbone using *only* the unsigned incidence
matrices for routing, and returns the per-rank embeddings consumed by the
readout. These tests run on a tiny clique-lifted graph (NO GraphUniverse).
"""

import pytest

from topobench.nn.backbones.simplicial.mpsn import MPSN
from topobench.nn.wrappers.simplicial.mpsn_wrapper import MPSNWrapper
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def test_wrapper_is_auto_discovered():
    """MPSNWrapper is exported via the wrappers auto-discovery (no manual init)."""
    import topobench.nn.wrappers as wrappers

    assert hasattr(wrappers, "MPSNWrapper")
    assert "MPSNWrapper" in wrappers.WRAPPER_CLASSES


@pytest.fixture
def sg1_clique_lifted_unsigned(simple_graph_1):
    """Clique-lift simple_graph_1 exactly as production does (unsigned, dim 2).

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        A simple graph data object (from the shared conftest).

    Returns
    -------
    torch_geometric.data.Data
        Lifted complex with unsigned 0/1 incidences and ``complex_dim = 2``.
    """
    lifting = SimplicialCliqueLifting(complex_dim=2, signed=False)
    data = lifting(simple_graph_1)
    data.batch_0 = "null"
    return data


def _build_wrapper(data, out_dim):
    """Construct an MPSNWrapper around an MPSN backbone for the given complex.

    Parameters
    ----------
    data : torch_geometric.data.Data
        A clique-lifted complex carrying x_0/x_1/x_2.
    out_dim : int
        Hidden width / output channels.

    Returns
    -------
    MPSNWrapper
        The wrapper instance.
    """
    backbone = MPSN(
        in_channels_all=(
            data.x_0.shape[1],
            data.x_1.shape[1],
            data.x_2.shape[1],
        ),
        hidden_dim=out_dim,
        n_layers=3,
    )
    return MPSNWrapper(backbone, out_channels=out_dim, num_cell_dimensions=3)


def test_wrapper_returns_expected_keys(sg1_clique_lifted_unsigned):
    """The wrapper output exposes labels, batch_0 and per-rank embeddings.

    Parameters
    ----------
    sg1_clique_lifted_unsigned : torch_geometric.data.Data
        Unsigned clique-lifted complex.
    """
    data = sg1_clique_lifted_unsigned
    wrapper = _build_wrapper(data, out_dim=8)

    out = wrapper(data)

    for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
        assert key in out


def test_wrapper_per_rank_shapes(sg1_clique_lifted_unsigned):
    """Per-rank embeddings have the right cell counts and hidden width.

    Parameters
    ----------
    sg1_clique_lifted_unsigned : torch_geometric.data.Data
        Unsigned clique-lifted complex.
    """
    data = sg1_clique_lifted_unsigned
    out_dim = 8
    wrapper = _build_wrapper(data, out_dim=out_dim)

    out = wrapper(data)

    assert out["x_0"].shape == (data.x_0.shape[0], out_dim)
    assert out["x_1"].shape == (data.x_1.shape[0], out_dim)
    assert out["x_2"].shape == (data.x_2.shape[0], out_dim)
    assert out["labels"] is data.y


def test_wrapper_also_handles_signed_fixture(sg1_clique_lifted):
    """The wrapper is robust to signed incidences (it uses the support only).

    The shared ``sg1_clique_lifted`` fixture lifts with ``signed=True``; the
    backbone extracts the nonzero support, so the wrapper must still run
    end-to-end.

    Parameters
    ----------
    sg1_clique_lifted : torch_geometric.data.Data
        Signed clique-lifted complex (from the shared conftest).
    """
    data = sg1_clique_lifted
    wrapper = _build_wrapper(data, out_dim=4)

    out = wrapper(data)

    for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
        assert key in out
