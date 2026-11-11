"""Unit tests for WHNN."""

import torch

from topobench.nn.backbones.hypergraph.whnn import (
    WHNN,
    SlicedWassersteinPooling,
)

from ...._utils.nn_module_auto_test import NNModuleAutoTest


def test_whnn_forward_with_sparse_incidence():
    """Test WHNN with a sparse node-hyperedge incidence matrix."""
    x = torch.randn(5, 8)
    incidence = torch.tensor(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=torch.float32,
    ).to_sparse()

    auto_test = NNModuleAutoTest(
        [
            {
                "module": WHNN,
                "init": {
                    "num_features": x.shape[1],
                    "num_layers": 2,
                    "num_projections": 4,
                    "num_reference_points": 3,
                    "dropout": 0.0,
                },
                "forward": (x, incidence),
                "assert_shape": [x.shape, (incidence.shape[1], x.shape[1])],
            },
        ]
    )
    auto_test.run()


def test_whnn_forward_with_edge_index_incidence(random_graph_input):
    """Test WHNN with TopoBench's 2-row incidence index format."""
    x, _x_1, _x_2, edges_1, _edges_2 = random_graph_input
    model = WHNN(
        num_features=x.shape[1],
        num_layers=1,
        num_projections=3,
        num_reference_points=2,
        dropout=0.0,
    )

    x_out, hyperedge_out = model(x, edges_1)

    assert x_out.shape == x.shape
    assert hyperedge_out.shape[1] == x.shape[1]


def test_sliced_wasserstein_pooling_handles_empty_groups():
    """Test that pooling returns finite outputs for groups without members."""
    x = torch.randn(4, 6)
    groups = torch.tensor([0, 0, 2, 2])
    pooling = SlicedWassersteinPooling(
        in_channels=x.shape[1],
        out_channels=5,
        num_projections=3,
        num_reference_points=4,
    )

    out = pooling(x, groups, num_groups=4)

    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_whnn_reset_parameters(random_graph_input):
    """Test WHNN parameter reset."""
    x, _x_1, _x_2, edges_1, _edges_2 = random_graph_input
    model = WHNN(x.shape[1])

    model.reset_parameters()
    x_out, hyperedge_out = model(x, edges_1)

    assert x_out.shape == x.shape
    assert hyperedge_out.shape[1] == x.shape[1]
