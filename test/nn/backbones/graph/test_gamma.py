"""Equation-level and integration tests for GAMMA."""

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.graph.gamma import GAMMA
from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper


def _path_edge_index() -> torch.Tensor:
    """Return both directions of the path 0--1--2."""
    return torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )


def _identity_gamma(
    *,
    num_routing_iterations: int = 2,
    bias: bool = False,
    propagation: str = "raw",
) -> GAMMA:
    """Create a deterministic two-channel layer."""
    model = GAMMA(
        in_channels=2,
        out_channels=2,
        max_hops=2,
        num_routing_iterations=num_routing_iterations,
        propagation=propagation,
        bias=bias,
    ).double()
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.hop_scale.fill_(1.0)
        if model.bias is not None:
            model.bias.zero_()
    return model


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 0, "out_channels": 2}, "in_channels"),
        ({"in_channels": 2, "out_channels": 0}, "out_channels"),
        (
            {"in_channels": 2, "out_channels": 2, "max_hops": -1},
            "max_hops",
        ),
        (
            {
                "in_channels": 2,
                "out_channels": 2,
                "num_routing_iterations": 0,
            },
            "num_routing_iterations",
        ),
        (
            {
                "in_channels": 2,
                "out_channels": 2,
                "propagation": "random_walk",
            },
            "propagation",
        ),
        (
            {
                "in_channels": 2,
                "out_channels": 2,
                "normalization": "batch_norm",
            },
            "normalization",
        ),
        (
            {
                "in_channels": 2,
                "out_channels": 2,
                "squash_mode": "approximate",
            },
            "squash_mode",
        ),
        (
            {"in_channels": 2, "out_channels": 2, "eps": 0.0},
            "eps",
        ),
    ],
)
def test_constructor_rejects_invalid_arguments(kwargs, message):
    """Invalid architectural settings fail with a targeted message."""
    with pytest.raises(ValueError, match=message):
        GAMMA(**kwargs)


def test_squash_matches_hand_computed_fixture():
    """Capsule squash matches the paper formula, including a zero vector."""
    inputs = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [[0.0, 0.0], [15.0 / 26.0, 20.0 / 26.0]],
        dtype=torch.float64,
    )
    actual = GAMMA.squash(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)


def test_reference_squash_matches_tiny_norm_behavior():
    """Compatibility mode retains the authors' in-square-root epsilon."""
    inputs = torch.tensor(
        [[1e-4, 0.0], [1e-6, 0.0]],
        dtype=torch.float64,
    )
    norm_squared = inputs.square().sum(dim=-1, keepdim=True)
    expected = (
        norm_squared
        / (1.0 + norm_squared)
        * inputs
        / torch.sqrt(norm_squared + 1e-8)
    )
    model = GAMMA(2, 2, squash_mode="reference").double()

    actual = model._apply_squash(inputs)

    torch.testing.assert_close(actual, expected, atol=1e-20, rtol=1e-12)


def test_sparse_propagation_matches_adjacency_powers():
    """One and two recurrent propagations equal hand-computed A@H."""
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
    )
    first_expected = torch.tensor(
        [[0.0, 1.0], [3.0, 3.0], [0.0, 1.0]],
    )
    second_expected = torch.tensor(
        [[3.0, 3.0], [0.0, 2.0], [3.0, 3.0]],
    )

    first = GAMMA._propagate(features, _path_edge_index(), None)
    second = GAMMA._propagate(first, _path_edge_index(), None)

    torch.testing.assert_close(first, first_expected)
    torch.testing.assert_close(second, second_expected)


def test_weighted_propagation_honors_edge_orientation():
    """Each edge weight scales its source-to-target message exactly once."""
    features = torch.tensor([[2.0], [3.0], [5.0]])
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 2.0])
    expected = torch.tensor([[0.0], [11.0], [0.0]])

    actual = GAMMA._propagate(features, edge_index, edge_weight)

    torch.testing.assert_close(actual, expected)


def test_third_positional_argument_is_edge_weight():
    """The public forward signature remains compatible with official GAMMA."""
    torch.manual_seed(5)
    model = GAMMA(2, 3, propagation="raw")
    features = torch.randn(3, 2)
    edge_weight = torch.tensor([0.5, 1.0, 1.5, 2.0])

    positional = model(features, _path_edge_index(), edge_weight)
    keyword = model(
        features,
        _path_edge_index(),
        edge_weight=edge_weight,
    )

    torch.testing.assert_close(positional, keyword)


def test_paper_normalized_operator_matches_dense_adjacency():
    """Normalized mode applies symmetric normalization to A plus self-loops."""
    model = GAMMA(
        in_channels=2,
        out_channels=2,
        max_hops=1,
        propagation="normalized",
        bias=False,
    )
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )
    adjacency_with_loops = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    )
    degree = adjacency_with_loops.sum(dim=1)
    inverse_sqrt_degree = degree.rsqrt()
    normalized_adjacency = (
        inverse_sqrt_degree[:, None]
        * adjacency_with_loops
        * inverse_sqrt_degree[None, :]
    )
    expected = normalized_adjacency @ features
    prepared_edges, prepared_weights = model._prepare_propagation(
        features,
        _path_edge_index(),
        None,
    )

    actual = model._propagate(
        features,
        prepared_edges,
        prepared_weights,
    )

    torch.testing.assert_close(actual, expected)


def test_routing_matches_two_iteration_hand_fixture():
    """Routing returns the last coefficients that formed the output."""
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        dtype=torch.float64,
    )
    model = GAMMA(
        in_channels=2,
        out_channels=2,
        num_routing_iterations=2,
        bias=False,
    ).double()
    expected_weights = torch.tensor(
        [[0.322043464396, 0.355913071207, 0.322043464396]],
        dtype=torch.float64,
    )
    expected_output = torch.tensor(
        [[0.0, 0.112431902582]],
        dtype=torch.float64,
    )

    output, weights = model._route(candidates)

    torch.testing.assert_close(
        weights,
        expected_weights,
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output,
        expected_output,
        atol=1e-12,
        rtol=0.0,
    )


def test_one_routing_iteration_is_uniform_mixing():
    """R=1 cannot use its final, otherwise-unused agreement update."""
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        dtype=torch.float64,
    )
    model = GAMMA(
        in_channels=2,
        out_channels=2,
        num_routing_iterations=1,
        bias=False,
    ).double()
    output, weights = model._route(candidates)
    expected_mixture = candidates.mean(dim=1)

    torch.testing.assert_close(
        weights,
        torch.full_like(weights, 1.0 / 3.0),
    )
    torch.testing.assert_close(
        output,
        GAMMA.squash(expected_mixture),
    )


def test_paper_faithful_end_to_end_fixture():
    """The complete layer matches Algorithm 1 on a deterministic path."""
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [0.3920657049323666, 0.2422851122963414],
            [0.0, 0.5],
            [-0.3920657049323666, 0.2422851122963414],
        ],
        dtype=torch.float64,
    )
    model = _identity_gamma(propagation="normalized")

    actual = model(features, _path_edge_index())

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)


def test_authors_code_compatibility_fixture():
    """LayerNorm mode reproduces the pinned public reference implementation."""
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float64,
    )
    mathematical_weight = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    projection_bias = torch.tensor(
        [0.5, -0.5, 0.25],
        dtype=torch.float64,
    )
    hop_scale = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 0.5],
            [0.5, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    final_bias = torch.tensor(
        [0.1, -0.2, 0.3],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [0.0350188963, 0.3098698792, -0.1448887756],
            [0.0351177321, 0.3098640383, -0.1449817703],
            [0.0351186086, 0.3098638919, -0.1449825005],
        ],
        dtype=torch.float64,
    )
    model = GAMMA(
        in_channels=2,
        out_channels=3,
        max_hops=2,
        num_routing_iterations=2,
        propagation="raw",
        normalization="layer_norm",
        squash_mode="reference",
        projection_bias=True,
        bias=True,
    ).double()
    with torch.no_grad():
        model.projection.weight.copy_(mathematical_weight.T)
        model.projection.bias.copy_(projection_bias)
        model.hop_scale.copy_(hop_scale)
        model.bias.copy_(final_bias)

    actual = model(features, _path_edge_index())

    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=0.0)


def test_l2_normalizes_each_nonzero_node_hop_candidate():
    """Algorithm 1 normalization acts along channels, not nodes or hops."""
    model = _identity_gamma()
    features = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    hops = model._compute_hop_embeddings(
        features,
        torch.empty((2, 0), dtype=torch.long),
        None,
    )
    norms = torch.linalg.vector_norm(hops, dim=-1)

    assert torch.all((norms == 0.0) | torch.isclose(norms, norms.new_ones(())))


def test_forward_uses_one_projection_and_k_propagations():
    """The implementation realizes the paper's linear-time recurrence."""
    model = GAMMA(3, 4, max_hops=3)
    features = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )

    with (
        patch.object(
            model.projection,
            "forward",
            wraps=model.projection.forward,
        ) as projection,
        patch.object(
            model,
            "_propagate",
            wraps=model._propagate,
        ) as propagate,
    ):
        model(features, edge_index)

    assert projection.call_count == 1
    assert propagate.call_count == 3


def test_permutation_equivariance():
    """Permuting node ids only permutes GAMMA's node embeddings."""
    torch.manual_seed(7)
    model = GAMMA(3, 4)
    features = torch.randn(4, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())

    original = model(features, edge_index)
    permuted = model(
        features.index_select(0, permutation),
        inverse.index_select(0, edge_index.reshape(-1)).reshape(2, -1),
    )

    torch.testing.assert_close(
        permuted,
        original.index_select(0, permutation),
    )


def test_disjoint_batch_matches_individual_graphs():
    """Routing and propagation never mix two graphs in a PyG batch."""
    torch.manual_seed(11)
    model = GAMMA(3, 4)
    first_x = torch.randn(3, 3)
    second_x = torch.randn(2, 3)
    first_edges = _path_edge_index()
    second_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batched_edges = torch.cat(
        [first_edges, second_edges + first_x.size(0)],
        dim=1,
    )
    batched_x = torch.cat([first_x, second_x])
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    separately = torch.cat(
        [
            model(first_x, first_edges),
            model(second_x, second_edges),
        ]
    )
    together = model(batched_x, batched_edges, batch=batch)

    torch.testing.assert_close(together, separately)


def test_empty_edges_isolated_nodes_and_zero_features_are_finite():
    """Degenerate sparse inputs remain finite and preserve output shape."""
    model = GAMMA(2, 3)
    features = torch.zeros(4, 2)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    output = model(features, edge_index)
    weights = model.get_routing_weights(features, edge_index)

    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(4),
    )


def test_float16_zero_candidates_are_finite():
    """The numerical floor remains representable during mixed precision."""
    model = GAMMA(2, 3).half()
    features = torch.zeros(4, 2, dtype=torch.float16)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    output = model(features, edge_index)
    squashed = GAMMA.squash(torch.zeros(2, 3, dtype=torch.float16))

    assert torch.isfinite(output).all()
    assert torch.isfinite(squashed).all()


def test_cancelled_hops_have_finite_bounded_gradients():
    """Zero-safe L2 normalization avoids a 1/eps cancellation Jacobian."""
    model = _identity_gamma(propagation="normalized")
    features = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    prepared_edges, prepared_weights = model._prepare_propagation(
        features,
        edge_index,
        None,
    )
    hops = model._compute_hop_embeddings(
        features,
        prepared_edges,
        prepared_weights,
    )

    torch.testing.assert_close(hops[:, 1:], torch.zeros_like(hops[:, 1:]))
    hops.sum().backward()

    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().max() < 10.0


def test_gradients_reach_input_projection_gates_and_bias():
    """Every trainable component receives a finite gradient."""
    torch.manual_seed(13)
    model = GAMMA(3, 4)
    features = torch.randn(4, 3, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )

    loss = model(features, edge_index).square().sum()
    loss.backward()

    gradients = [
        features.grad,
        model.projection.weight.grad,
        model.hop_scale.grad,
        model.bias.grad,
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_reset_parameters_resets_all_deterministic_state():
    """Reset covers the projection bias, final bias, gates, and LayerNorm."""
    model = GAMMA(
        2,
        3,
        normalization="layer_norm",
        projection_bias=True,
    )
    with torch.no_grad():
        model.projection.bias.fill_(4.0)
        model.hop_scale.zero_()
        model.bias.fill_(4.0)
        model.hop_normalizer.weight.zero_()
        model.hop_normalizer.bias.fill_(4.0)

    model.reset_parameters()

    torch.testing.assert_close(
        model.bias,
        torch.zeros_like(model.bias),
    )
    torch.testing.assert_close(
        model.hop_normalizer.weight,
        torch.ones_like(model.hop_normalizer.weight),
    )
    torch.testing.assert_close(
        model.hop_normalizer.bias,
        torch.zeros_like(model.hop_normalizer.bias),
    )
    assert torch.count_nonzero(model.projection.bias) > 0
    assert torch.count_nonzero(model.hop_scale) > 0


def test_parameter_count_matches_paper_formula():
    """Default parameters are W, K+1 channel gates, and final bias."""
    model = GAMMA(
        in_channels=2,
        out_channels=3,
        max_hops=2,
    )
    expected = (2 * 3) + (3 * 3) + 3
    actual = sum(parameter.numel() for parameter in model.parameters())

    assert actual == expected


@pytest.mark.parametrize(
    ("x", "edge_index", "batch", "edge_weight", "error"),
    [
        (
            torch.ones(2),
            torch.empty((2, 0), dtype=torch.long),
            None,
            None,
            "x must have shape",
        ),
        (
            torch.ones(2, 3),
            torch.empty((2, 0), dtype=torch.long),
            None,
            None,
            "expected 2",
        ),
        (
            torch.ones(2, 2, dtype=torch.long),
            torch.empty((2, 0), dtype=torch.long),
            None,
            None,
            "floating-point",
        ),
        (
            torch.ones(2, 2),
            torch.empty((3, 0), dtype=torch.long),
            None,
            None,
            "edge_index",
        ),
        (
            torch.ones(2, 2),
            torch.empty((2, 0), dtype=torch.int32),
            None,
            None,
            "torch.long",
        ),
        (
            torch.ones(2, 2),
            torch.tensor([[0], [1]], dtype=torch.long),
            None,
            torch.ones(2),
            "one value per edge",
        ),
        (
            torch.ones(2, 2),
            torch.empty((2, 0), dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            None,
            "one graph index per node",
        ),
    ],
)
def test_forward_validates_structural_inputs(
    x,
    edge_index,
    batch,
    edge_weight,
    error,
):
    """Malformed tensors fail before sparse propagation."""
    model = GAMMA(2, 2)
    with pytest.raises((TypeError, ValueError), match=error):
        model(x, edge_index, batch=batch, edge_weight=edge_weight)


@pytest.mark.parametrize(
    ("edge_index", "batch", "edge_weight", "error"),
    [
        (
            torch.empty((2, 0), dtype=torch.long, device="meta"),
            None,
            None,
            "edge_index and x",
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            None,
            torch.empty(0, device="meta"),
            "edge_weight and x",
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(2, dtype=torch.long, device="meta"),
            None,
            "batch and x",
        ),
    ],
)
def test_forward_rejects_mixed_devices(
    edge_index,
    batch,
    edge_weight,
    error,
):
    """Device mismatches fail before any graph operation runs."""
    model = GAMMA(2, 2)
    with pytest.raises(ValueError, match=error):
        model(
            torch.ones(2, 2),
            edge_index,
            batch=batch,
            edge_weight=edge_weight,
        )


def test_gnn_wrapper_contract_and_repr():
    """The standard TopoBench graph wrapper accepts GAMMA without adaptation."""
    model = GAMMA(2, 3)
    wrapper = GNNWrapper(
        model,
        out_channels=3,
        num_cell_dimensions=1,
        residual_connections=False,
    )
    batch = Data(
        x_0=torch.randn(3, 2),
        edge_index=_path_edge_index(),
        edge_weight=torch.ones(4),
        batch_0=torch.zeros(3, dtype=torch.long),
        y=torch.tensor([1]),
    )

    output = wrapper(batch)

    assert output["x_0"].shape == (3, 3)
    assert output["labels"] is batch.y
    assert "GAMMA" in repr(wrapper)
