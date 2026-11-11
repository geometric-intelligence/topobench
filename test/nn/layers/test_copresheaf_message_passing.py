"""Tests for generic copresheaf message passing."""

import pytest
import torch
from torch import nn

from topobench.nn.layers.copresheaf import (
    CopresheafMessagePassing,
    CopresheafRoute,
    HigherOrderCopresheafLayer,
)


def _sparse(indices, values, size):
    return torch.sparse_coo_tensor(
        torch.tensor(indices), torch.tensor(values, dtype=torch.float), size
    ).coalesce()


def test_identity_transport_sum_matches_manual_aggregation():
    """Identity rho reduces Definition 9 to ordinary sum aggregation."""
    layer = CopresheafMessagePassing(
        channels=2, map_type="identity", aggr="sum"
    )
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros(2, 2)
    edge_index = torch.tensor([[0, 1, 1], [0, 0, 1]])

    output = layer(source, target, edge_index)

    expected = torch.tensor([[4.0, 6.0], [3.0, 4.0]])
    torch.testing.assert_close(output, expected)


def test_message_aggregation_is_invariant_to_edge_order():
    """Reordering a neighborhood edge list does not change the aggregate."""
    torch.manual_seed(2)
    layer = CopresheafMessagePassing(
        channels=4, heads=2, map_type="full", aggr="mean"
    )
    x = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 1, 1, 2, 2]])
    permutation = torch.tensor([4, 2, 0, 3, 1])

    first = layer(x, x, edge_index)
    second = layer(x, x, edge_index[:, permutation])

    torch.testing.assert_close(first, second)


def test_structural_padding_outside_feature_range_is_ignored():
    """Sparse padding for absent cell ranks cannot create phantom messages."""
    layer = CopresheafMessagePassing(
        channels=2, map_type="identity", aggr="sum"
    )
    features = torch.tensor([[1.0, 2.0]])
    edge_index = torch.tensor([[0, 4], [0, 4]])

    output = layer(features, features, edge_index)

    torch.testing.assert_close(output, features)


def test_route_parsing_and_connectivity_orientation():
    """Selected TopoBench matrices are converted from target/source order."""
    route = CopresheafRoute.from_neighborhood("up_incidence-0")
    # Two rank-1 targets by three rank-0 sources.
    connectivity = _sparse([[0, 0, 1], [0, 2, 1]], [1, 1, 1], (2, 3))

    edge_index, weights = route.edge_index(connectivity)

    assert (route.source_rank, route.target_rank) == (0, 1)
    assert edge_index.tolist() == [[0, 2, 1], [0, 0, 1]]
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_higher_order_layer_supports_bidirectional_cross_rank_routes():
    """Definition 10 updates both ranks through independent CIM routes."""
    layer = HigherOrderCopresheafLayer(
        channels=4,
        neighborhoods=["up_incidence-0", "down_incidence-1"],
        map_type="identity",
        aggr="mean",
    )
    features = {0: torch.randn(3, 4), 1: torch.randn(2, 4)}
    incidence = _sparse([[0, 0, 1, 1], [0, 1, 1, 2]], [1, 1, 1, 1], (2, 3))
    connectivities = {
        "up_incidence-0": incidence,
        "down_incidence-1": incidence.t(),
    }

    output = layer(features, connectivities)

    assert output[0].shape == features[0].shape
    assert output[1].shape == features[1].shape
    sum(value.square().sum() for value in output.values()).backward()
    assert any(parameter.grad is not None for parameter in layer.parameters())


def test_gated_route_aggregation_prefers_same_rank_at_initialization():
    """Learned route mixing is normalized per target and starts conservatively."""
    layer = HigherOrderCopresheafLayer(
        channels=4,
        neighborhoods=[
            "up_adjacency-0",
            "down_incidence-1",
            "up_incidence-0",
            "up_adjacency-1",
        ],
        map_type="identity",
        neighborhood_aggr="gated",
        route_self_bias=2.0,
        message_gate_init=-3.0,
    )

    weights = layer.route_weights()

    assert weights["up_adjacency-0"] > weights["down_incidence-1"]
    assert weights["up_adjacency-1"] > weights["up_incidence-0"]
    assert weights["up_adjacency-0"] + weights[
        "down_incidence-1"
    ] == pytest.approx(1.0)
    assert float(torch.sigmoid(layer.updates["0"].message_gate)) < 0.05


class _AddUpdate(nn.Module):
    """Tiny deterministic update used to verify scheduled route order."""

    def forward(self, features, message):
        return features + message


def test_scheduled_layer_applies_requested_up_down_route_order():
    """A scheduled layer performs sequential, in-place rank updates."""
    layer = HigherOrderCopresheafLayer(
        channels=1,
        neighborhoods=[
            "up_incidence-0",
            "up_incidence-1",
            "down_incidence-2",
            "down_incidence-1",
        ],
        map_type="identity",
        aggr="sum",
        message_schedule=[
            "up_incidence-0",
            "up_incidence-1",
            "down_incidence-2",
            "down_incidence-1",
        ],
    )
    for rank in ("0", "1", "2"):
        layer.updates[rank] = _AddUpdate()

    features = {
        0: torch.tensor([[1.0], [10.0]]),
        1: torch.tensor([[100.0]]),
        2: torch.tensor([[1000.0]]),
    }
    connectivities = {
        "up_incidence-0": _sparse([[0, 0], [0, 1]], [1, 1], (1, 2)),
        "up_incidence-1": _sparse([[0], [0]], [1], (1, 1)),
        "down_incidence-2": _sparse([[0], [0]], [1], (1, 1)),
        "down_incidence-1": _sparse([[0, 1], [0, 0]], [1, 1], (2, 1)),
    }

    output = layer(features, connectivities)

    torch.testing.assert_close(output[0], torch.tensor([[1223.0], [1232.0]]))
    torch.testing.assert_close(output[1], torch.tensor([[1222.0]]))
    torch.testing.assert_close(output[2], torch.tensor([[1111.0]]))
    assert layer.route_weights() == {}


def test_scheduled_layer_rejects_unknown_neighborhood():
    """A schedule typo should fail during construction, not mid-training."""
    with pytest.raises(ValueError, match="unknown neighborhoods"):
        HigherOrderCopresheafLayer(
            channels=2,
            neighborhoods=["up_incidence-0"],
            map_type="identity",
            message_schedule=["up_incidence-1"],
        )
