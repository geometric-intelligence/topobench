"""Unit tests for DirSNN backbone."""

import pytest
import torch

from topobench.nn.backbones.simplicial.dirsnn import DirSNN, DirSNNLayer
from topobench.nn.wrappers.simplicial.dirsnn_wrapper import DIRSNN_ADJ_KEYS
from topobench.transforms.liftings.graph2simplicial import (
    DirectedSimplicialLifting,
    SimplicialCliqueLifting,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_edge_data():
    """Return a small synthetic edge-level test bench.

    Returns
    -------
    dict
        Dictionary with edge features and two dense adjacency
        matrices, sized so that downstream tests stay cheap.
    """
    torch.manual_seed(0)
    n_edges = 6
    edge_channels = 4
    x_1 = torch.randn(n_edges, edge_channels)
    adj_a = (torch.rand(n_edges, n_edges) > 0.6).float()
    adj_b = (torch.rand(n_edges, n_edges) > 0.6).float()
    return {
        "x_1": x_1,
        "n_edges": n_edges,
        "edge_channels": edge_channels,
        "adjs": (adj_a, adj_b),
    }


# ---------------------------------------------------------------------------
# DirSNN (the outer module)
# ---------------------------------------------------------------------------


def test_dirsnn_construction(random_edge_data):
    """Build a vanilla DirSNN and inspect its structure.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    model = DirSNN(
        edge_channels=d["edge_channels"],
        n_layers=2,
        n_hid=8,
        conv_order=1,
        n_adjs=len(d["adjs"]),
    )
    assert isinstance(model, DirSNN)
    assert len(model.layers) == 2
    assert hasattr(model, "in_linear_1")
    assert all(isinstance(layer, DirSNNLayer) for layer in model.layers)


def test_dirsnn_forward_shape(random_edge_data):
    """The forward pass preserves the number of edges.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    n_hid = 10
    model = DirSNN(
        edge_channels=d["edge_channels"],
        n_layers=2,
        n_hid=n_hid,
        conv_order=2,
        n_adjs=len(d["adjs"]),
        update_func="relu",
    )
    out = model(d["x_1"], d["adjs"])
    assert out.shape == (d["n_edges"], n_hid)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "update_func", ["relu", "sigmoid", "leaky_relu", None]
)
def test_dirsnn_update_funcs(random_edge_data, update_func):
    """All advertised update activations run end-to-end.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    update_func : str or None
        Activation tag passed through to the layer.
    """
    d = random_edge_data
    model = DirSNN(
        edge_channels=d["edge_channels"],
        n_layers=1,
        n_hid=6,
        conv_order=1,
        n_adjs=len(d["adjs"]),
        update_func=update_func,
    )
    out = model(d["x_1"], d["adjs"])
    assert out.shape == (d["n_edges"], 6)


def test_dirsnn_aggr_norm(random_edge_data):
    """Aggregation normalisation does not crash and produces finite output.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    model = DirSNN(
        edge_channels=d["edge_channels"],
        n_layers=1,
        n_hid=5,
        conv_order=2,
        n_adjs=len(d["adjs"]),
        aggr_norm=True,
        update_func="relu",
    )
    out = model(d["x_1"], d["adjs"])
    assert torch.isfinite(out).all()


def test_dirsnn_backprop(random_edge_data):
    """Check that gradients flow through the network parameters.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    model = DirSNN(
        edge_channels=d["edge_channels"],
        n_layers=1,
        n_hid=4,
        conv_order=1,
        n_adjs=len(d["adjs"]),
        update_func="relu",
    )
    out = model(d["x_1"], d["adjs"])
    loss = out.pow(2).mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_dirsnn_backprop_sparse(random_edge_data):
    """Check that gradients flow through ``torch.sparse.mm`` adjacencies.

    Mirrors :func:`test_dirsnn_backprop` but converts every adjacency
    to a sparse COO tensor before the forward pass so the sparse
    propagation path is exercised end-to-end. This catches silent
    dtype / device / autograd issues that the dense path would not
    surface (e.g. ``torch.sparse.mm`` requiring a coalesced operand
    or a specific dtype combination).

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    sparse_adjs = tuple(a.to_sparse().coalesce() for a in d["adjs"])
    assert all(a.is_sparse for a in sparse_adjs)
    model = DirSNN(
        edge_channels=d["edge_channels"],
        n_layers=1,
        n_hid=4,
        conv_order=2,  # >1 exercises repeated sparse.mm propagation
        n_adjs=len(sparse_adjs),
        update_func="relu",
    )
    out = model(d["x_1"], sparse_adjs)
    loss = out.pow(2).mean()
    loss.backward()
    # Every learnable parameter must have a finite gradient, and at
    # least one of them must be non-zero (otherwise the sparse path
    # silently zeros out the propagation).
    any_nonzero = False
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        if p.grad.abs().sum().item() > 0:
            any_nonzero = True
    assert any_nonzero, (
        "All gradients are zero through the sparse path -- something "
        "in torch.sparse.mm propagation is silently dropping signal."
    )


# ---------------------------------------------------------------------------
# DirSNNLayer (the inner layer)
# ---------------------------------------------------------------------------


def test_layer_invalid_init_raises():
    """Construction rejects unknown initialisation tags."""
    with pytest.raises(AssertionError):
        DirSNNLayer(
            in_channels_1=3,
            out_channels_1=3,
            conv_order=1,
            n_adjs=1,
            initialization="not_a_thing",
        )


def test_layer_invalid_conv_order_raises():
    """Convolution order must be strictly positive."""
    with pytest.raises(AssertionError):
        DirSNNLayer(
            in_channels_1=3,
            out_channels_1=3,
            conv_order=0,
            n_adjs=1,
        )


def test_layer_xavier_uniform_init():
    """Selecting Xavier-uniform initialisation works and changes weights.

    The default initialisation is Xavier-normal; explicitly switching to
    Xavier-uniform must yield finite weights of the expected shape.
    """
    layer = DirSNNLayer(
        in_channels_1=3,
        out_channels_1=4,
        conv_order=2,
        n_adjs=2,
        initialization="xavier_uniform",
    )
    # conv_order * n_adjs + 1
    assert layer.weight_1.shape == (3, 4, 5)
    assert torch.isfinite(layer.weight_1).all()


def test_layer_reset_parameters_bad_init_raises():
    """``reset_parameters`` rejects unknown initialisations at call time.

    The init-time assertion guards normal construction; here we mutate
    the attribute so the runtime branch in ``reset_parameters`` is
    exercised directly.
    """
    layer = DirSNNLayer(
        in_channels_1=2,
        out_channels_1=2,
        conv_order=1,
        n_adjs=1,
    )
    layer.initialization = "unknown"
    with pytest.raises(RuntimeError):
        layer.reset_parameters()


def test_layer_wrong_n_adjs_raises(random_edge_data):
    """Mismatched ``n_adjs`` between init and forward must raise.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    layer = DirSNNLayer(
        in_channels_1=d["edge_channels"],
        out_channels_1=d["edge_channels"],
        conv_order=1,
        n_adjs=1,
    )
    with pytest.raises(AssertionError):
        layer(d["x_1"], d["adjs"])  # 2 adjacencies but layer expects 1


def test_layer_update_unknown_raises():
    """``update`` raises :class:`ValueError` for an unknown activation tag.

    The reference Dir-SNN code silently returned ``None`` here, which
    propagated to ``forward`` and produced opaque downstream crashes
    (e.g. ``NoneType has no attribute 'shape'`` deeper in the call
    stack). Surfacing the misconfiguration early is much friendlier.
    """
    layer = DirSNNLayer(
        in_channels_1=2,
        out_channels_1=2,
        conv_order=1,
        n_adjs=1,
        update_func="not_an_activation",
    )
    with pytest.raises(ValueError, match="Unknown update_func"):
        layer.update(torch.zeros(3, 2))


def test_layer_sparse_adjacency(random_edge_data):
    """The layer handles sparse adjacencies via ``torch.sparse.mm``.

    Parameters
    ----------
    random_edge_data : dict
        Synthetic edge-level data fixture.
    """
    d = random_edge_data
    sparse_adjs = tuple(a.to_sparse() for a in d["adjs"])
    layer = DirSNNLayer(
        in_channels_1=d["edge_channels"],
        out_channels_1=5,
        conv_order=2,
        n_adjs=2,
        aggr_norm=True,
        update_func="relu",
    )
    out = layer(d["x_1"], sparse_adjs)
    assert out.shape == (d["n_edges"], 5)
    assert torch.isfinite(out).all()


def test_layer_dense_aggr_norm_handles_isolated_rows():
    """``aggr_norm_func`` zeroes out empty rows safely."""
    layer = DirSNNLayer(
        in_channels_1=2,
        out_channels_1=2,
        conv_order=1,
        n_adjs=1,
        aggr_norm=True,
    )
    # row 1 has no neighbours -> inverse degree is +inf and must be
    # mapped to 0.
    adj = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = layer.aggr_norm_func(adj, x)
    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros(2))


# ---------------------------------------------------------------------------
# Integration with a TopoBench-lifted simplicial complex
# ---------------------------------------------------------------------------


def test_dirsnn_on_lifted_graph(simple_graph_1):
    """End-to-end test on a clique-lifted simple graph.

    Sanity-checks that the backbone can still run on plain Hodge
    Laplacians produced by the standard :class:`SimplicialCliqueLifting`
    -- this exercises the ``n_adjs`` configurability of the backbone
    independently of the new directed lifting.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph fixture from ``test/conftest.py``.
    """
    lifting = SimplicialCliqueLifting(complex_dim=3, signed=True)
    data = lifting(simple_graph_1)

    edge_channels = data.x_1.shape[1]
    out_dim = 8
    model = DirSNN(
        edge_channels=edge_channels,
        n_layers=2,
        n_hid=out_dim,
        conv_order=1,
        n_adjs=2,
        update_func="relu",
    )
    adjs = (data.down_laplacian_1, data.up_laplacian_1)
    out = model(data.x_1, adjs)
    assert out.shape == (data.x_1.shape[0], out_dim)
    assert torch.isfinite(out).all()


def test_dirsnn_on_directed_lifted_graph(simple_graph_1):
    """Run Dir-SNN end-to-end on a :class:`DirectedSimplicialLifting`.

    This is the production configuration: ten directed edge adjacencies
    feed the backbone via :data:`DIRSNN_ADJ_KEYS`.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph fixture from ``test/conftest.py``.
    """
    lifting = DirectedSimplicialLifting(signed=True)
    data = lifting(simple_graph_1)

    edge_channels = data.x_1.shape[1]
    out_dim = 6
    model = DirSNN(
        edge_channels=edge_channels,
        n_layers=2,
        n_hid=out_dim,
        conv_order=1,
        n_adjs=10,
        update_func="relu",
    )
    adjs = tuple(getattr(data, key) for key in DIRSNN_ADJ_KEYS)
    assert len(adjs) == 10
    out = model(data.x_1, adjs)
    assert out.shape == (data.x_1.shape[0], out_dim)
    assert torch.isfinite(out).all()
