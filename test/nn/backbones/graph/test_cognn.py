"""Unit tests for the Co-GNN backbone."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.graph import CoGNN
from topobench.nn.backbones.graph.cognn import (
    CoGNNActionNet,
    CoGNNTempSoftPlus,
    WeightedGCNConv,
    WeightedGINConv,
    WeightedGNNConv,
    build_conv_layer,
)
from topobench.nn.wrappers.graph import GNNWrapper


def testCoGNN(random_graph_input):
    """Unit test for the CoGNN backbone forward pass via GNNWrapper.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    batch = torch_geometric.data.Data(
        x_0=x,
        y=x,
        x=x,
        edge_index=edges_1,
        batch_0=torch.zeros(x.shape[0], dtype=torch.long),
    )
    model = CoGNN(x.shape[1], x.shape[1])
    wrapper = GNNWrapper(
        model,
        **{"out_channels": x.shape[1], "num_cell_dimensions": 1},
    )

    _ = wrapper.__repr__()

    model_out = wrapper(batch)
    assert model_out["x_0"].shape == x.shape


@pytest.mark.parametrize(
    "conv_type", ["sum_gnn", "mean_gnn", "gcn", "gin"]
)
def testCoGNN_conv_types(random_graph_input, conv_type):
    """Unit test for all Co-GNN action/environment layer types.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    conv_type : str
        The convolution type to test.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    model = CoGNN(
        x.shape[1],
        16,
        num_layers=2,
        env_conv_type=conv_type,
        act_conv_type=conv_type,
    )
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 16)


def testCoGNN_options(random_graph_input):
    """Unit test for skip connections, layer norm and learned temperature.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    model = CoGNN(
        x.shape[1],
        16,
        num_layers=2,
        act_num_layers=2,
        skip=True,
        layer_norm=True,
        learn_temp=True,
        dropout=0.1,
        act="gelu",
    )
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 16)

    # Input edge weights are combined with the action-derived gates.
    edge_weight = torch.rand(edges_1.shape[1])
    out = model(x, edges_1, edge_weight=edge_weight)
    assert out.shape == (x.shape[0], 16)


def testCoGNN_gradient_flow(random_graph_input):
    """Test that gradients flow through the straight-through estimator.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    model = CoGNN(x.shape[1], 16, num_layers=2)
    out = model(x, edges_1)
    out.sum().backward()

    # The action networks receive gradients only through the
    # straight-through Gumbel-softmax estimator of the edge weights.
    act_grads = [
        p.grad
        for p in model.in_act_net.parameters()
        if p.grad is not None
    ]
    assert len(act_grads) > 0
    assert model.encoder.weight.grad is not None


def testCoGNN_create_edge_weight():
    """Test the action-to-edge-weight combination rule (Eq. (2))."""
    model = CoGNN(4, 4, num_layers=1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    keep_in_prob = torch.tensor([1.0, 0.0, 1.0])
    keep_out_prob = torch.tensor([1.0, 1.0, 0.0])
    edge_weight = model.create_edge_weight(
        edge_index, keep_in_prob, keep_out_prob
    )
    # Edge (u, v) is kept iff u broadcasts and v listens:
    # (0, 1): node 1 does not listen -> 0
    # (1, 2): node 1 broadcasts, node 2 listens -> 1
    # (2, 0): node 2 does not broadcast -> 0
    assert torch.equal(edge_weight, torch.tensor([0.0, 1.0, 0.0]))


def testCoGNNActionNet(random_graph_input):
    """Unit test for the Co-GNN action network.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    for num_layers in [1, 2]:
        net = CoGNNActionNet(x.shape[1], 8, num_layers)
        logits = net(x, edges_1)
        assert logits.shape == (x.shape[0], 2)


def testCoGNNTempSoftPlus(random_graph_input):
    """Unit test for the learnable Gumbel-softmax temperature module.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    temp_model = CoGNNTempSoftPlus(x.shape[1], tau0=0.5)
    temp = temp_model(x)
    assert temp.shape == (x.shape[0], 1)
    # Temperatures are strictly positive and bounded by 1 / tau0.
    assert (temp > 0).all()
    assert (temp <= 2.0).all()


def testWeightedConvs(random_graph_input):
    """Unit test for the weighted convolution layers.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of random input tensors for testing.
    """
    torch.manual_seed(0)
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    edge_weight = torch.rand(edges_1.shape[1])
    for conv in [
        WeightedGNNConv(x.shape[1], 8, aggr="sum"),
        WeightedGNNConv(x.shape[1], 8, aggr="mean"),
        WeightedGCNConv(x.shape[1], 8),
        WeightedGINConv(x.shape[1], 8),
    ]:
        out = conv(x, edges_1)
        assert out.shape == (x.shape[0], 8)
        out = conv(x, edges_1, edge_weight=edge_weight)
        assert out.shape == (x.shape[0], 8)

    # Zero edge weights must block all messages: for WeightedGNNConv the
    # aggregated neighbor part is zero, so outputs for isolated updates
    # coincide with using a zeroed neighborhood.
    conv = WeightedGNNConv(x.shape[1], 8, aggr="sum")
    zero_w = torch.zeros(edges_1.shape[1])
    out_blocked = conv(x, edges_1, edge_weight=zero_w)
    out_manual = conv.lin(
        torch.cat((x, torch.zeros_like(x)), dim=-1)
    )
    assert torch.allclose(out_blocked, out_manual, atol=1e-6)


def testWeightedAggregate_mean_gating():
    """Regression test for mean aggregation with 0/1 action gates.

    Mirrors the original Co-GNN implementation (``models/layers.py``):
    messages are scaled by the edge weight and then reduced with a
    standard mean over the full in-degree, so gated (weight-0) edges
    contribute zeros to the numerator but still count in the
    denominator.
    """
    from topobench.nn.backbones.graph.cognn import weighted_aggregate

    x = torch.tensor([[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]])
    # Two edges into node 2: one active (from node 0), one gated
    # (from node 1).
    edge_index = torch.tensor([[0, 1], [2, 2]])
    edge_weight = torch.tensor([1.0, 0.0])
    out = weighted_aggregate(
        x, edge_index, num_nodes=3, edge_weight=edge_weight, aggr="mean"
    )
    # Reference semantics: (1*x_0 + 0*x_1) / in_degree(2) = x_0 / 2.
    assert torch.allclose(out[2], x[0] / 2)
    # Nodes without incoming edges aggregate to zero.
    assert torch.equal(out[0], torch.zeros(2))
    assert torch.equal(out[1], torch.zeros(2))


def test_build_conv_layer_invalid():
    """Test that an unsupported convolution type raises a ValueError."""
    with pytest.raises(ValueError):
        build_conv_layer("unsupported", 4, 4)
