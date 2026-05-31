"""Unit tests for the TopNets graph route operator."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.topnets import (
    _TopNetsRouteOperator,
    _TopologicalFiltrationLayer,
)
from topobench.nn.wrappers.graph import GNNWrapper


def _batch_data() -> Batch:
    graph_0 = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.long,
        ),
        y=torch.tensor([0]),
    )
    graph_1 = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
            dtype=torch.long,
        ),
        y=torch.tensor([1]),
    )
    batch = Batch.from_data_list([graph_0, graph_1])
    batch.x_0 = batch.x
    batch.batch_0 = batch.batch
    return batch


def test_topnets_forward_shape_and_gradients():
    """TopNets returns finite node embeddings and supports backprop."""
    batch = _batch_data()
    model = _TopNetsRouteOperator(
        in_channels=5,
        hidden_channels=8,
        num_steps=2,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch.x, batch.edge_index, batch=batch.batch)

    assert out.shape == (batch.num_nodes, 8)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert model.input_projection.weight.grad is not None


def test_topnets_gin_variant():
    """TopNets supports the GIN branch used in the reference code."""
    batch = _batch_data()
    model = _TopNetsRouteOperator(
        in_channels=5,
        hidden_channels=8,
        num_steps=2,
        gnn_type="gin",
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch.x, batch.edge_index, batch=batch.batch)

    assert out.shape == (batch.num_nodes, 8)
    assert torch.isfinite(out).all()


def test_topnets_wrapper_output():
    """The graph wrapper exposes TopNets outputs in TopoBench format."""
    batch = _batch_data()
    model = _TopNetsRouteOperator(
        in_channels=5,
        hidden_channels=8,
        num_steps=2,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )
    wrapper = GNNWrapper(
        model,
        out_channels=8,
        num_cell_dimensions=1,
        residual_connections=False,
    )

    model_out = wrapper(batch)

    assert model_out["x_0"].shape == (batch.num_nodes, 8)
    assert torch.equal(model_out["labels"], batch.y)
    assert torch.equal(model_out["batch_0"], batch.batch_0)


def test_topnets_without_edges():
    """TopNets handles edgeless graph batches."""
    x = torch.randn(3, 5)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    model = _TopNetsRouteOperator(
        in_channels=5,
        hidden_channels=8,
        num_steps=2,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(x, edge_index, batch=batch)

    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_topnets_invalid_parameters():
    """Invalid TopNets construction parameters raise clear errors."""
    with pytest.raises(ValueError, match="num_steps must be positive"):
        _TopNetsRouteOperator(in_channels=5, hidden_channels=8, num_steps=0)

    with pytest.raises(ValueError, match="gnn_type"):
        _TopNetsRouteOperator(in_channels=5, hidden_channels=8, gnn_type="gat")


def test_proxy_persistence_pairs_for_single_edge():
    """Proxy persistence uses node births, incident edge deaths, and graph maxima."""
    layer = _TopologicalFiltrationLayer(
        channels=2,
        num_filtrations=1,
        filtration_hidden=2,
        coord_fun_count=1,
    )
    filtrations = torch.tensor([[0.2], [0.7], [0.5]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    edge_batch = torch.zeros(1, dtype=torch.long)

    persistence0, persistence1 = layer._compute_persistence(
        filtrations=filtrations,
        edge_index=edge_index,
        batch=batch,
        edge_batch=edge_batch,
        num_graphs=1,
    )

    expected0 = torch.tensor([[[0.2, 0.7], [0.7, 0.7], [0.5, 0.7]]])
    expected1 = torch.tensor([[[0.7, 0.7]]])
    torch.testing.assert_close(persistence0, expected0)
    torch.testing.assert_close(persistence1, expected1)
