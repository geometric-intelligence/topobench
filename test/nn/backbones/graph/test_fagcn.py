"""Unit tests for the FAGCN graph backbone."""

import pytest
import torch

from topobench.nn.backbones.graph.fagcn import FAGCN


@pytest.fixture
def graph_inputs():
    """Create a small graph and node features."""
    x = torch.randn(6, 4)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def test_initialization():
    """The model exposes the requested output dimension and layer count."""
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        out_channels=5,
        num_layers=3,
    )
    assert model.out_channels == 5
    assert len(model.layers) == 3


def test_forward_shape(graph_inputs):
    """The forward pass returns one embedding per node."""
    x, edge_index = graph_inputs
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        out_channels=6,
        num_layers=2,
        dropout=0.0,
    )
    output = model(x, edge_index)
    assert output.shape == (6, 6)
    assert torch.isfinite(output).all()


def test_default_output_dimension(graph_inputs):
    """The hidden dimension is used when out_channels is omitted."""
    x, edge_index = graph_inputs
    model = FAGCN(
        in_channels=4,
        hidden_channels=7,
        num_layers=1,
        dropout=0.0,
    )
    assert model(x, edge_index).shape == (6, 7)


def test_batched_input_is_accepted(graph_inputs):
    """The wrapper-provided batch vector does not alter node embeddings."""
    x, edge_index = graph_inputs
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        dropout=0.0,
    )
    output = model(x, edge_index, batch=batch)
    assert output.shape == (6, 8)


def test_scalar_edge_attributes_are_accepted(graph_inputs):
    """Single-channel edge attributes are accepted by the interface."""
    x, edge_index = graph_inputs
    edge_attr = torch.ones(edge_index.shape[1], 1)
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        dropout=0.0,
    )
    output = model(x, edge_index, edge_attr=edge_attr)
    assert output.shape == (6, 8)


def test_explicit_edge_weights_are_accepted(graph_inputs):
    """Explicit edge weights are accepted by the interface."""
    x, edge_index = graph_inputs
    edge_weight = torch.ones(edge_index.shape[1])
    edge_attr = torch.zeros(edge_index.shape[1], 1)
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=1,
        dropout=0.0,
    )
    output = model(
        x,
        edge_index,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
    )
    assert torch.isfinite(output).all()


def test_edgeless_graph():
    """FAGCN handles graphs without explicit edges."""
    x = torch.randn(5, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        dropout=0.0,
    )
    output = model(x, edge_index)
    assert output.shape == (5, 8)


def test_gradient_flow(graph_inputs):
    """Gradients flow through projections and adaptive-frequency layers."""
    x, edge_index = graph_inputs
    x.requires_grad_()
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=2,
        dropout=0.0,
    )
    loss = model(x, edge_index).square().mean()
    loss.backward()
    assert x.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_reset_parameters_changes_weights():
    """Resetting parameters reinitializes the input projection."""
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
    )
    previous = model.input_linear.weight.detach().clone()
    model.reset_parameters()
    assert not torch.equal(previous, model.input_linear.weight)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"in_channels": 0, "hidden_channels": 8},
            "in_channels",
        ),
        (
            {"in_channels": 4, "hidden_channels": 0},
            "hidden_channels",
        ),
        (
            {
                "in_channels": 4,
                "hidden_channels": 8,
                "out_channels": 0,
            },
            "out_channels",
        ),
        (
            {
                "in_channels": 4,
                "hidden_channels": 8,
                "num_layers": 0,
            },
            "at least one",
        ),
        (
            {
                "in_channels": 4,
                "hidden_channels": 8,
                "dropout": 1.5,
            },
            "dropout",
        ),
    ],
)
def test_invalid_arguments(kwargs, message):
    """Invalid constructor arguments fail with clear errors."""
    with pytest.raises(ValueError, match=message):
        FAGCN(**kwargs)


def test_invalid_forward_inputs(graph_inputs):
    """Malformed node and edge tensors are rejected."""
    x, edge_index = graph_inputs
    model = FAGCN(
        in_channels=4,
        hidden_channels=8,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="rank-two"):
        model(x.unsqueeze(0), edge_index)

    with pytest.raises(ValueError, match="shape"):
        model(x, edge_index.T)

    with pytest.raises(ValueError, match="expected 4"):
        model(torch.randn(6, 3), edge_index)
