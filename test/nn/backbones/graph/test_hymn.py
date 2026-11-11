"""Tests for the paper-faithful HyMN graph backbone."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.hymn import (
    _GLOBAL_STATISTICS_CACHE,
    HyMN,
)


def _graph(edge_pairs, num_nodes, feature_dim=5):
    """Create a simple undirected PyG graph."""
    directed_edges = []
    for source, target in edge_pairs:
        directed_edges.extend([(source, target), (target, source)])
    edge_index = torch.tensor(directed_edges, dtype=torch.long)
    if directed_edges:
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=torch.randn(num_nodes, feature_dim), edge_index=edge_index)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 0, "hidden_channels": 8}, "positive"),
        ({"in_channels": 5, "hidden_channels": 0}, "positive"),
        (
            {"in_channels": 5, "hidden_channels": 8, "num_layers": 0},
            "num_layers",
        ),
        (
            {"in_channels": 5, "hidden_channels": 8, "num_samples": 0},
            "num_samples",
        ),
        (
            {"in_channels": 5, "hidden_channels": 8, "cse_steps": 0},
            "cse_steps",
        ),
        ({"in_channels": 5, "hidden_channels": 8, "dropout": 1.1}, "dropout"),
        (
            {
                "in_channels": 5,
                "hidden_channels": 8,
                "sample_aggregation": "max",
            },
            "sample_aggregation",
        ),
        (
            {"in_channels": 5, "hidden_channels": 8, "cache_size": -1},
            "cache_size",
        ),
        (
            {
                "in_channels": 5,
                "hidden_channels": 8,
                "global_cache_size": -1,
            },
            "global_cache_size",
        ),
        (
            {"in_channels": 5, "hidden_channels": 8, "cse_channels": 8},
            "cse_channels",
        ),
    ],
)
def test_hymn_rejects_invalid_configuration(kwargs, message):
    """Every constructor constraint raises a useful error."""
    with pytest.raises(ValueError, match=message):
        HyMN(**kwargs)


def test_hymn_graph_key_is_edge_order_independent():
    """The cache key preserves graph structure but ignores COO ordering."""
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    reordered = edge_index[:, torch.tensor([2, 0, 1])]

    assert HyMN._graph_key(3, edge_index) == HyMN._graph_key(3, reordered)
    assert HyMN._graph_key(4, edge_index) != HyMN._graph_key(3, edge_index)
    assert HyMN._graph_key(0, torch.empty((2, 0), dtype=torch.long)) == (
        0,
        b"",
    )


def test_hymn_cse_and_selection_match_equation_four():
    """CSE columns equal diag(A^k)/k! and rank the top-SC node first."""
    graph = _graph(
        [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
        5,
    )
    roots, cse = HyMN._compute_graph_statistics(
        graph.num_nodes,
        graph.edge_index,
        num_marked_views=2,
        cse_steps=3,
    )

    assert roots[0] == 0
    torch.testing.assert_close(cse[:, 0], torch.zeros(5))
    torch.testing.assert_close(
        cse[:, 1], torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
    )
    assert cse[0, 2] == pytest.approx(2 / 3)
    assert cse[1, 2] == pytest.approx(1 / 3)


def test_hymn_statistics_handle_empty_and_oversampled_graphs():
    """Algorithm 1 statistics cover empty graphs and the reference T>|V| rule."""
    empty_roots, empty_cse = HyMN._compute_graph_statistics(
        0,
        torch.empty((2, 0), dtype=torch.long),
        num_marked_views=2,
        cse_steps=3,
    )
    roots, cse = HyMN._compute_graph_statistics(
        1,
        torch.empty((2, 0), dtype=torch.long),
        num_marked_views=3,
        cse_steps=2,
    )

    assert empty_roots == ()
    assert empty_cse.shape == (0, 3)
    assert roots == (0, 0, 0)
    torch.testing.assert_close(cse, torch.zeros((1, 2)))
    with pytest.raises(ValueError, match="num_nodes"):
        HyMN._compute_graph_statistics(-1, torch.empty((2, 0)), 1)
    with pytest.raises(ValueError, match="num_marked_views"):
        HyMN._compute_graph_statistics(1, torch.empty((2, 0)), -1)


def test_hymn_batch_statistics_mark_roots_and_use_lru_cache():
    """Batched preprocessing is graph-local and reuses cached CSEs."""
    first = _graph([(0, 1), (1, 2), (2, 0)], 3)
    second = _graph([(0, 1)], 2)
    batched = Batch.from_data_list([first, second])
    model = HyMN(
        5, 8, num_samples=2, cse_steps=3, cse_channels=2, cache_size=2
    )

    markers, cse = model._statistics_for_batch(
        batched.edge_index,
        batched.batch,
        batched.num_nodes,
    )
    cached_markers, cached_cse = model._statistics_for_batch(
        batched.edge_index,
        batched.batch,
        batched.num_nodes,
    )

    assert markers.shape == (5, 2)
    assert markers[:, 0].sum() == 0
    assert markers[:3, 1].sum() == 1
    assert markers[3:, 1].sum() == 1
    assert cse.shape == (5, 3)
    assert len(model._statistics_cache) == 2
    torch.testing.assert_close(markers, cached_markers)
    torch.testing.assert_close(cse, cached_cse)


def test_hymn_global_cache_reuses_reference_preprocessing(monkeypatch):
    """Independent models reuse the authors' deterministic preprocessing."""
    graph = _graph([(0, 1), (1, 2), (2, 0)], 3)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    _GLOBAL_STATISTICS_CACHE.clear()
    try:
        first = HyMN(
            5,
            8,
            num_samples=2,
            cse_steps=3,
            cse_channels=2,
            cache_size=0,
            global_cache_size=2,
        )
        expected = first._statistics_for_batch(
            graph.edge_index,
            batch,
            graph.num_nodes,
        )
        assert len(_GLOBAL_STATISTICS_CACHE) == 1

        def fail_if_recomputed(*args, **kwargs):
            raise AssertionError("global cache entry was recomputed")

        monkeypatch.setattr(
            HyMN,
            "_compute_graph_statistics",
            staticmethod(fail_if_recomputed),
        )
        second = HyMN(
            5,
            8,
            num_samples=2,
            cse_steps=3,
            cse_channels=2,
            cache_size=0,
            global_cache_size=2,
        )
        actual = second._statistics_for_batch(
            graph.edge_index,
            batch,
            graph.num_nodes,
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    finally:
        _GLOBAL_STATISTICS_CACHE.clear()


def test_hymn_batch_statistics_validate_assignments_and_skip_empty_ids():
    """Batched CSE validates assignments and tolerates empty graph IDs."""
    model = HyMN(5, 8, cse_steps=2, cse_channels=2, cache_size=0)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch_with_gap = torch.tensor([0, 0, 2, 2])
    markers, cse = model._statistics_for_batch(edge_index, batch_with_gap, 4)

    assert markers.shape == (4, 3)
    assert cse.shape == (4, 2)
    assert not model._statistics_cache
    with pytest.raises(ValueError, match="assign every node"):
        model._statistics_for_batch(edge_index, torch.tensor([[0, 0]]), 2)
    with pytest.raises(ValueError, match="negative"):
        model._statistics_for_batch(edge_index, torch.tensor([-1, -1]), 2)
    with pytest.raises(ValueError, match="grouped"):
        model._statistics_for_batch(
            edge_index,
            torch.tensor([0, 1, 0, 1]),
            4,
        )


def test_hymn_expand_views_offsets_edges_and_markers():
    """The augmented bag is materialized as independent disjoint views."""
    model = HyMN(2, 4, num_samples=2, cse_steps=2, cse_channels=2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    markers = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    expanded_x, expanded_edges, expanded_markers = model._expand_views(
        x,
        edge_index,
        markers,
    )

    torch.testing.assert_close(expanded_x, x.repeat(2, 1))
    assert expanded_edges.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert expanded_markers.squeeze(-1).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_hymn_disjoint_views_match_authors_wide_tensor_update():
    """Disjoint views exactly reproduce the authors' shared wide-tensor GIN."""
    graph = _graph([(0, 1), (1, 2), (2, 0)], 3, feature_dim=4)
    model = HyMN(
        4,
        4,
        num_layers=2,
        num_samples=3,
        cse_steps=2,
        use_centrality_encoding=False,
        dropout=0.0,
        residual=True,
    ).eval()
    markers = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    disjoint, expanded_edges, expanded_markers = model._expand_views(
        graph.x,
        graph.edge_index,
        markers,
    )
    reference = graph.x[:, None, :].expand(-1, model.num_samples, -1)
    source, target = graph.edge_index

    for conv in model.convs:
        disjoint_input = torch.cat((disjoint, expanded_markers), dim=-1)
        disjoint_update = torch.relu(conv(disjoint_input, expanded_edges))
        disjoint = disjoint + disjoint_update

        # The reference implementation stores all views in one wide tensor,
        # aggregates each view independently, then applies one shared MLP.
        reference_input = torch.cat((reference, markers.unsqueeze(-1)), dim=-1)
        wide_input = reference_input.reshape(graph.num_nodes, -1)
        wide_messages = torch.zeros_like(wide_input)
        wide_messages.index_add_(0, target, wide_input[source])
        wide_update = (1 + conv.eps) * wide_input + wide_messages
        reference_update = torch.relu(
            conv.nn(wide_update.reshape(-1, model.hidden_channels + 1))
        ).reshape(graph.num_nodes, model.num_samples, model.hidden_channels)
        reference = reference + reference_update

    disjoint = disjoint.reshape(
        model.num_samples,
        graph.num_nodes,
        model.hidden_channels,
    ).permute(1, 0, 2)
    torch.testing.assert_close(disjoint, reference)
    torch.testing.assert_close(disjoint.mean(dim=1), reference.mean(dim=1))


def test_hymn_forward_and_gradients():
    """HyMN returns node-aligned, finite, differentiable embeddings."""
    graph = _graph([(0, 1), (1, 2), (2, 0), (2, 3)], 4)
    model = HyMN(
        5,
        8,
        num_layers=2,
        num_samples=3,
        cse_steps=3,
        cse_channels=2,
    )
    output = model(graph.x, graph.edge_index)

    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_hymn_supports_no_cse_no_norm_no_residual_and_sum():
    """Paper-supported ablations and sum view pooling execute together."""
    graph = _graph([(0, 1), (1, 2)], 3)
    model = HyMN(
        5,
        8,
        num_layers=1,
        num_samples=2,
        cse_steps=2,
        batch_norm=False,
        residual=False,
        use_centrality_encoding=False,
        sample_aggregation="sum",
    )

    output = model(graph.x, graph.edge_index)

    assert model.cse_norm is None
    assert model.cse_encoder is None
    assert output.shape == (3, 8)


def test_hymn_mean_and_sum_aggregation_are_consistent():
    """Sum aggregation equals mean aggregation times the bag size."""
    graph = _graph([(0, 1), (1, 2), (2, 0)], 3)
    model = HyMN(5, 8, num_samples=3, cse_steps=3, cse_channels=2).eval()

    mean_output = model(graph.x, graph.edge_index)
    model.sample_aggregation = "sum"
    sum_output = model(graph.x, graph.edge_index)

    torch.testing.assert_close(sum_output, mean_output * model.num_samples)


def test_hymn_is_deterministic_and_batch_local_in_eval_mode():
    """Selection is deterministic and another graph cannot change outputs."""
    first = _graph([(0, 1), (1, 2), (2, 0), (2, 3)], 4)
    second = _graph([(0, 1), (1, 2), (2, 3)], 4)
    model = HyMN(
        5,
        8,
        num_layers=2,
        num_samples=3,
        cse_steps=3,
        cse_channels=2,
        dropout=0.2,
    ).eval()

    isolated = model(first.x, first.edge_index)
    repeated = model(first.x, first.edge_index)
    batched = Batch.from_data_list([first, second])
    combined = model(batched.x, batched.edge_index, batch=batched.batch)

    torch.testing.assert_close(isolated, repeated)
    torch.testing.assert_close(isolated, combined[: first.num_nodes])


def test_hymn_permutation_equivariance_with_unique_root():
    """Relabelling an asymmetric graph relabels node embeddings."""
    graph = _graph(
        [(0, 1), (1, 2), (2, 0), (0, 3), (0, 4), (4, 5)],
        6,
    )
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted = Data(
        x=graph.x[permutation],
        edge_index=inverse[graph.edge_index],
    )
    model = HyMN(
        5,
        8,
        num_layers=2,
        num_samples=2,
        cse_steps=4,
        cse_channels=2,
    ).eval()

    original_output = model(graph.x, graph.edge_index)
    permuted_output = model(permuted.x, permuted.edge_index)
    torch.testing.assert_close(
        original_output[permutation],
        permuted_output,
        rtol=1e-5,
        atol=1e-6,
    )


def test_hymn_forward_validates_tensor_shapes():
    """Forward rejects malformed node and edge tensors."""
    model = HyMN(5, 8, cse_steps=2, cse_channels=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        model(torch.randn(5), torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="edge_index"):
        model(torch.randn(2, 5), torch.empty((3, 0), dtype=torch.long))
