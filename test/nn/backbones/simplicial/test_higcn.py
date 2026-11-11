"""Unit tests for the HiGCN backbone."""

import pytest
import torch
from torch_sparse import SparseTensor

from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.simplicial import HiGCN
from topobench.nn.backbones.simplicial.higcn import (
    HiGCNProp,
    build_flower_petals_laplacians,
    _binarize,
    _to_sparse_tensor,
)
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def _lift(graph):
    """Lift a graph to a simplicial complex with triangles.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        Input graph.

    Returns
    -------
    torch_geometric.data.Data
        Lifted simplicial complex (clique complex up to dimension 2).
    """
    lifting = SimplicialCliqueLifting(complex_dim=2, signed=False)
    return lifting(graph)


def test_HiGCN(simple_graph_1):
    """Test the HiGCN backbone end to end on a lifted graph.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    data = _lift(simple_graph_1)
    out_dim = 4
    incidences = (data.incidence_1, data.incidence_2)
    expected_shapes = [(data.x_0.shape[0], out_dim)]

    auto_test = NNModuleAutoTest(
        [
            {
                "module": HiGCN,
                "init": (data.x_0.shape[1], out_dim),
                "forward": (data.x_0, incidences),
                "assert_shape": expected_shapes,
            },
        ]
    )
    auto_test.run()


@pytest.fixture
def create_sample_data():
    """Create sample node features and incidence matrices.

    Returns
    -------
    dict
        Sample data for testing.
    """
    num_nodes, num_edges, num_tri = 6, 9, 4
    x = torch.randn(num_nodes, 5)

    # Node-edge incidence: each edge connects two nodes.
    edges = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 4, 0], [1, 2, 2, 3, 3, 4, 4, 5, 5]]
    )
    rows = torch.cat([edges[0], edges[1]])
    cols = torch.cat([torch.arange(num_edges), torch.arange(num_edges)])
    incidence_1 = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.ones(2 * num_edges),
        size=(num_nodes, num_edges),
    ).coalesce()

    # Edge-triangle incidence: each triangle has three edges.
    e_rows = torch.tensor([0, 1, 2, 0, 3, 4, 2, 4, 6, 4, 7, 8])
    t_cols = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    incidence_2 = torch.sparse_coo_tensor(
        torch.stack([e_rows, t_cols]),
        torch.ones(12),
        size=(num_edges, num_tri),
    ).coalesce()

    return {
        "x": x,
        "incidences": (incidence_1, incidence_2),
        "num_nodes": num_nodes,
    }


def test_basic_initialization():
    """Test basic initialization and module structure of HiGCN."""
    model = HiGCN(in_channels=5, hidden_channels=8, order=2)
    assert model is not None
    assert len(model.lin_in) == 2
    assert len(model.prop) == 2
    assert model.lin_out.in_features == 8 * 2
    assert model.lin_out.out_features == 8


def test_different_orders(create_sample_data):
    """Test HiGCN with order 1 and order 2.

    Parameters
    ----------
    create_sample_data : dict
        Sample data for testing.
    """
    data = create_sample_data
    for order in (1, 2):
        model = HiGCN(in_channels=5, hidden_channels=8, order=order)
        assert len(model.lin_in) == order
        out = model(data["x"], data["incidences"])
        assert out.shape == (data["num_nodes"], 8)
        assert torch.isfinite(out).all()


def test_invalid_order():
    """Test that a non-positive order raises an assertion error."""
    with pytest.raises(AssertionError):
        HiGCN(in_channels=5, hidden_channels=8, order=0)


def test_invalid_K():
    """Test that a non-positive number of hops raises an assertion error."""
    with pytest.raises(AssertionError):
        HiGCN(in_channels=5, hidden_channels=8, K=0)


def test_order_exceeds_incidences(create_sample_data):
    """Test that requesting more orders than incidences raises an error.

    Parameters
    ----------
    create_sample_data : dict
        Sample data for testing.
    """
    data = create_sample_data
    model = HiGCN(in_channels=5, hidden_channels=8, order=3)
    with pytest.raises(AssertionError):
        model(data["x"], data["incidences"])


def test_dprate_branch(create_sample_data):
    """Test the dprate dropout branch is exercised.

    Parameters
    ----------
    create_sample_data : dict
        Sample data for testing.
    """
    data = create_sample_data
    model = HiGCN(in_channels=5, hidden_channels=8, order=2, dprate=0.3)
    out = model(data["x"], data["incidences"])
    assert out.shape == (data["num_nodes"], 8)


def test_reset_parameters(create_sample_data):
    """Test that reset_parameters reinitializes the filter weights.

    Parameters
    ----------
    create_sample_data : dict
        Sample data for testing.
    """
    model = HiGCN(in_channels=5, hidden_channels=8, order=2, K=10, alpha=0.1)
    # Perturb a filter weight then reset; it should return to the PPR profile.
    with torch.no_grad():
        model.prop[0].filter_weights[0] += 1.0
    model.reset_parameters()
    expected_w0 = 0.1  # alpha * (1 - alpha) ** 0
    assert torch.isclose(
        model.prop[0].filter_weights[0], torch.tensor(expected_w0)
    )


def test_higcn_prop_ppr_initialization():
    """Test that HiGCNProp initializes filter weights with the PPR profile."""
    K, alpha = 5, 0.2
    prop = HiGCNProp(K=K, alpha=alpha)
    assert "HiGCNProp" in repr(prop)
    assert prop.filter_weights.shape == (K + 1,)
    for k in range(K):
        assert torch.isclose(
            prop.filter_weights[k], torch.tensor(alpha * (1 - alpha) ** k)
        )
    assert torch.isclose(
        prop.filter_weights[-1], torch.tensor((1 - alpha) ** K)
    )


def test_to_sparse_tensor_takes_absolute_value():
    """Test that _to_sparse_tensor returns unsigned membership entries."""
    signed = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([-1.0, 1.0]),
        size=(2, 2),
    )
    st = _to_sparse_tensor(signed)
    assert isinstance(st, SparseTensor)
    assert (st.coo()[2] >= 0).all()


def test_binarize():
    """Test that _binarize maps all stored values to one."""
    st = SparseTensor(
        row=torch.tensor([0, 1]),
        col=torch.tensor([0, 1]),
        value=torch.tensor([2.0, 3.0]),
        sparse_sizes=(2, 2),
    )
    binarized = _binarize(st)
    assert torch.equal(
        binarized.coo()[2], torch.ones_like(binarized.coo()[2])
    )


def test_build_flower_petals_laplacians(simple_graph_1):
    """Test FP Laplacian construction and the triangle-membership property.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    data = _lift(simple_graph_1)
    incidences = (data.incidence_1, data.incidence_2)
    laplacians = build_flower_petals_laplacians(incidences, order=2)

    assert len(laplacians) == 2
    n_nodes = data.x_0.shape[0]
    for lap in laplacians:
        assert lap.sparse_sizes() == (n_nodes, n_nodes)

    # Order-2 (triangle) operator: nodes that are in no triangle must have an
    # all-zero row in the Flower-Petals Laplacian.
    l2 = laplacians[1].to_dense()
    h2_counts = torch.sparse.mm(
        torch.abs(data.incidence_1).coalesce(),
        torch.abs(data.incidence_2).coalesce().to_dense(),
    )
    node_in_triangle = h2_counts.sum(dim=1) > 0
    for node in range(n_nodes):
        if not node_in_triangle[node]:
            assert torch.allclose(l2[node], torch.zeros(n_nodes))


def test_no_triangles_runs():
    """Test that the order-2 branch is well defined when there are no triangles."""
    num_nodes, num_edges = 4, 3
    # A path graph 0-1-2-3 has no triangles.
    rows = torch.tensor([0, 1, 1, 2, 2, 3])
    cols = torch.tensor([0, 0, 1, 1, 2, 2])
    incidence_1 = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), torch.ones(6), size=(num_nodes, num_edges)
    ).coalesce()
    incidence_2 = torch.sparse_coo_tensor(
        size=(num_edges, 0)
    ).coalesce()

    model = HiGCN(in_channels=5, hidden_channels=8, order=2)
    out = model(torch.randn(num_nodes, 5), (incidence_1, incidence_2))
    assert out.shape == (num_nodes, 8)
    assert torch.isfinite(out).all()
