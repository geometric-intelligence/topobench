"""Unit tests for SANN."""

import torch

from topobench.nn.backbones.combinatorial import HOPSE
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)
from topobench.transforms.data_manipulations.precompute_khop_features import (
    PrecomputeKHopFeatures
)


def test_SANN(simple_graph_1):
    """Test SANN.

        Test the SANN backbone module.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            A fixture of simple graph 1.
    """
    max_hop = 2
    complex_dim = 3
    lifting_signed = SimplicialCliqueLifting(
            complex_dim=complex_dim, signed=True
        )
    precompute_k_hop = PrecomputeKHopFeatures(
        max_hop=max_hop,
        complex_dim=complex_dim,
        use_initial_features=True,
    )
    data = lifting_signed(simple_graph_1)
    data = precompute_k_hop(data)
    out_dim = 4

    # PrecomputeKHopFeatures stores hops x{k}_{t} for t = 0 .. max_hop-1 (see transform impl).
    for t in range(max_hop):
        for j in range(complex_dim):
            data[f"x{j}_{t}"] = data[f"x{j}_{t}"][:, 0:1]

    x_in = tuple(
        tuple(data[f"x{i}_{t}"] for t in range(max_hop)) for i in range(complex_dim)
    )
    expected_shapes = [
        (data.x.shape[0], out_dim),
        (data.x_1.shape[0], out_dim),
        (data.x_2.shape[0], out_dim),
    ]

    model = HOPSE(
        (1, 1, 1),
        out_dim,
        "leaky_relu",
        complex_dim,
        max_hop,
        2,
    )
    torch.manual_seed(0)
    out1 = model(x_in)
    torch.manual_seed(0)
    out2 = model(x_in)

    assert len(out1) == complex_dim
    for k in range(complex_dim):
        assert len(out1[k]) == max_hop
        for t in range(max_hop):
            assert out1[k][t].shape == expected_shapes[k]
            assert torch.equal(out1[k][t], out2[k][t])
