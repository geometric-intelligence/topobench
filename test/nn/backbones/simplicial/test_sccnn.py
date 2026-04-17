"""Unit tests for SCCNN."""

import pytest
import torch
from torch_geometric.utils import get_laplacian
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.simplicial import SCCNNCustom
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def test_SCCNNCustom(simple_graph_1):
    """Test SCCNNCustom.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph data.
    """
    lifting_signed = SimplicialCliqueLifting(
            complex_dim=3, signed=True
        )
    data = lifting_signed(simple_graph_1)
    out_dim = 4
    conv_order = 1
    sc_order = 3
    laplacian_all = (
            data.hodge_laplacian_0,
            data.down_laplacian_1,
            data.up_laplacian_1,
            data.down_laplacian_2,
            data.up_laplacian_2,
        )
    incidence_all = (data.incidence_1, data.incidence_2)
    expected_shapes = [(data.x.shape[0], out_dim), (data.x_1.shape[0], out_dim), (data.x_2.shape[0], out_dim)]

    auto_test = NNModuleAutoTest([
        {
            "module" : SCCNNCustom,
            "init": ((data.x.shape[1], data.x_1.shape[1], data.x_2.shape[1]), (out_dim, out_dim, out_dim), conv_order, sc_order),
            "forward":  ((data.x, data.x_1, data.x_2), laplacian_all, incidence_all),
            "assert_shape": expected_shapes
        },
    ])
    auto_test.run()


@pytest.fixture
def create_sample_data():
    """Create sample data for testing.

    Returns
    -------
    dict
        Sample data for testing.
    """
    num_nodes = 5
    x = torch.randn(num_nodes, 3)  # 3 node features
    x_1 = torch.randn(8, 4)  # 8 edges with 4 features
    x_2 = torch.randn(6, 5)  # 6 faces with 5 features

    # Create sample Laplacians and incidence matrices
    hodge_laplacian_0 = torch.sparse_coo_tensor(size=(num_nodes, num_nodes))
    down_laplacian_1 = torch.sparse_coo_tensor(size=(8, 8))
    up_laplacian_1 = torch.sparse_coo_tensor(size=(8, 8))
    down_laplacian_2 = torch.sparse_coo_tensor(size=(6, 6))
    up_laplacian_2 = torch.sparse_coo_tensor(size=(6, 6))

    incidence_1 = torch.sparse_coo_tensor(size=(num_nodes, 8))
    incidence_2 = torch.sparse_coo_tensor(size=(8, 6))

    return {
        'x': x,
        'x_1': x_1,
        'x_2': x_2,
        'laplacian_all': (hodge_laplacian_0, down_laplacian_1, up_laplacian_1, down_laplacian_2, up_laplacian_2),
        'incidence_all': (incidence_1, incidence_2)
    }

def test_sccnn_basic_initialization():
    """Test basic initialization of SCCNNCustom."""
    in_channels = (3, 4, 5)
    hidden_channels = (6, 6, 6)

    # Test basic initialization
    model = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=3
    )
    assert model is not None

    # Verify layer structure
    assert len(model.layers) == 2  # Default n_layers is 2
    assert hasattr(model, 'in_linear_0')
    assert hasattr(model, 'in_linear_1')
    assert hasattr(model, 'in_linear_2')

def test_update_functions():
    """Test different update functions in the SCCNN."""
    in_channels = (3, 4, 5)
    hidden_channels = (6, 6, 6)

    # Test sigmoid update function
    model = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=3,
        update_func="sigmoid"
    )
    assert model is not None

    # Test ReLU update function
    model = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=3,
        update_func="relu"
    )
    assert model is not None

def test_aggr_norm(create_sample_data):
    """Test aggregation normalization functionality.

    Parameters
    ----------
    create_sample_data : dict
        Sample data for testing.
    """
    data = create_sample_data

    model = SCCNNCustom(
        in_channels_all=(3, 4, 5),
        hidden_channels_all=(6, 6, 6),
        conv_order=2,
        sc_order=3,
        aggr_norm=True
    )

    # Forward pass with aggregation normalization
    output = model(
        (data['x'], data['x_1'], data['x_2']),
        data['laplacian_all'],
        data['incidence_all']
    )

    assert len(output) == 3
    assert all(torch.isfinite(out).all() for out in output)

def test_different_conv_orders():
    """Test SCCNN with different convolution orders."""
    in_channels = (3, 4, 5)
    hidden_channels = (6, 6, 6)

    # Test with conv_order = 1
    model1 = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=1,
        sc_order=3
    )
    assert model1 is not None

    # Test with conv_order = 3
    model2 = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=3,
        sc_order=3
    )
    assert model2 is not None

    # Test invalid conv_order
    with pytest.raises(AssertionError):
        model = SCCNNCustom(
            in_channels_all=in_channels,
            hidden_channels_all=hidden_channels,
            conv_order=0,
            sc_order=3
        )

def test_different_sc_orders():
    """Test SCCNN with different simplicial complex orders."""
    in_channels = (3, 4, 5)
    hidden_channels = (6, 6, 6)

    # Test with sc_order = 2
    model1 = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=2
    )
    assert model1 is not None

    # Test with sc_order > 2
    model2 = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=3
    )
    assert model2 is not None

def test_forward_shapes(create_sample_data):
    """Test output shapes for different input configurations.

    Parameters
    ----------
    create_sample_data : dict
        Sample data for testing.
    """
    data = create_sample_data

    model = SCCNNCustom(
        in_channels_all=(3, 4, 5),
        hidden_channels_all=(6, 6, 6),
        conv_order=2,
        sc_order=3
    )

    output = model(
        (data['x'], data['x_1'], data['x_2']),
        data['laplacian_all'],
        data['incidence_all']
    )

    assert output[0].shape == (data['x'].shape[0], 6)
    assert output[1].shape == (data['x_1'].shape[0], 6)
    assert output[2].shape == (data['x_2'].shape[0], 6)

def test_n_layers():
    """Test SCCNN with different numbers of layers."""
    in_channels = (3, 4, 5)
    hidden_channels = (6, 6, 6)

    # Test with 1 layer
    model1 = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=3,
        n_layers=1
    )
    assert len(model1.layers) == 1

    # Test with 3 layers
    model2 = SCCNNCustom(
        in_channels_all=in_channels,
        hidden_channels_all=hidden_channels,
        conv_order=2,
        sc_order=3,
        n_layers=3
    )
    assert len(model2.layers) == 3
