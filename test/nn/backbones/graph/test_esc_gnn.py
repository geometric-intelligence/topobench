import torch
from torch import nn
from torch_geometric.data import Data

from topobench.nn.backbones.graph.esc_gnn import ESCGNN
from topobench.transforms.data_manipulations.esc_structural_encoding import (
    ESCStructuralEncoding,
)


def _features(num_nodes: int, offset: float = 0.0) -> torch.Tensor:
    values = torch.arange(num_nodes * 64, dtype=torch.float32)
    return values.reshape(num_nodes, 64).div(100).add(offset)


def _small_inputs() -> dict[str, torch.Tensor]:
    return {
        "x": _features(5),
        "edge_index": torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
            dtype=torch.long,
        ),
        "esc_code_id": torch.tensor(
            [0, 12, 4, 386, 11, 23, 299], dtype=torch.long
        ),
        "esc_code_count": torch.tensor(
            [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0],
            dtype=torch.float32,
        ),
        "esc_nnz_per_edge": torch.tensor([2, 1, 0, 2, 1, 1], dtype=torch.long),
    }


def _second_graph_inputs() -> dict[str, torch.Tensor]:
    return {
        "x": _features(4, offset=1.0),
        "edge_index": torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
        ),
        "esc_code_id": torch.tensor([8, 17, 22, 30], dtype=torch.long),
        "esc_code_count": torch.tensor(
            [1.0, 2.0, 1.0, 4.0], dtype=torch.float32
        ),
        "esc_nnz_per_edge": torch.tensor([0, 1, 2, 1], dtype=torch.long),
    }


def _assert_gradients(module: nn.Module) -> None:
    parameters = [parameter for parameter in module.parameters()]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in parameters
    )
    assert any(
        torch.count_nonzero(parameter.grad) > 0 for parameter in parameters
    )


def test_sparse_weighted_embedding_and_zero_code_routing_match_dense():
    model = ESCGNN().eval()
    inputs = _small_inputs()
    num_edges = inputs["edge_index"].size(1)
    edge_id = torch.repeat_interleave(
        torch.arange(num_edges), inputs["esc_nnz_per_edge"]
    )
    dense = torch.zeros(num_edges, model.num_structural_codes)
    dense.index_put_(
        (edge_id, inputs["esc_code_id"]),
        inputs["esc_code_count"],
        accumulate=True,
    )

    with torch.no_grad():
        actual = model._encode_structure(
            num_edges,
            inputs["esc_code_id"],
            inputs["esc_code_count"],
            inputs["esc_nnz_per_edge"],
        )
        expected = model.structural_mlp(
            dense @ model.structural_embedding.weight
        )

    assert actual.shape == (num_edges, 64)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_gine_update_matches_documented_pyg_instantiation():
    conv = ESCGNN().convs[0]
    conv.nn = nn.Identity()
    conv.lin = nn.Identity()
    with torch.no_grad():
        conv.eps.fill_(0.25)

    x = torch.zeros(3, 64)
    x[:, 0] = torch.tensor([1.0, 2.0, 4.0])
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]])
    edge_structure = torch.zeros(3, 64)
    edge_structure[:, 0] = torch.tensor([-3.0, 1.0, -10.0])

    actual = conv(x, edge_index, edge_attr=edge_structure)
    messages = torch.relu(x[edge_index[0]] + edge_structure)
    expected = (1.0 + conv.eps) * x
    expected = expected.index_add(0, edge_index[1], messages)

    assert torch.equal(actual, expected)


def test_forward_shapes_and_gradients_reach_every_component():
    model = ESCGNN()
    inputs = _small_inputs()

    output = model(**inputs)
    output.square().mean().backward()

    assert output.shape == (5, 64)
    _assert_gradients(model.structural_embedding)
    _assert_gradients(model.structural_mlp)
    for conv in model.convs:
        _assert_gradients(conv)
    _assert_gradients(model.jk_projection)


def test_reset_parameters_resets_all_learned_components():
    model = ESCGNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(7.0)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.fill_(7.0)
                module.running_var.fill_(7.0)
                module.num_batches_tracked.fill_(7)

    model.reset_parameters()

    for name, parameter in model.named_parameters():
        assert not torch.all(parameter == 7), (
            f"parameter was not reset: {name}"
        )
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
            assert torch.equal(
                module.running_mean, torch.zeros_like(module.running_mean)
            )
            assert torch.equal(
                module.running_var, torch.ones_like(module.running_var)
            )
            assert module.num_batches_tracked.item() == 0
    for conv in model.convs:
        assert torch.equal(conv.eps, torch.zeros_like(conv.eps))


def test_empty_edges_remain_finite_and_skip_edge_batch_norm():
    model = ESCGNN().eval()
    edge_batch_norm_calls = []
    handle = model.structural_mlp[0].register_forward_pre_hook(
        lambda *_: edge_batch_norm_calls.append(None)
    )
    inputs = {
        "x": _features(4),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "esc_code_id": torch.empty(0, dtype=torch.long),
        "esc_code_count": torch.empty(0, dtype=torch.float32),
        "esc_nnz_per_edge": torch.empty(0, dtype=torch.long),
    }

    try:
        with torch.no_grad():
            edge_structure = model._encode_structure(
                0,
                inputs["esc_code_id"],
                inputs["esc_code_count"],
                inputs["esc_nnz_per_edge"],
            )
            output = model(**inputs)
    finally:
        handle.remove()

    assert edge_structure.shape == (0, 64)
    assert output.shape == (4, 64)
    assert torch.isfinite(output).all()
    assert edge_batch_norm_calls == []


def test_batched_output_matches_single_graph_outputs_in_eval_mode():
    model = ESCGNN().eval()
    first = _small_inputs()
    second = _second_graph_inputs()
    batched = {
        "x": torch.cat((first["x"], second["x"])),
        "edge_index": torch.cat(
            (
                first["edge_index"],
                second["edge_index"] + first["x"].size(0),
            ),
            dim=1,
        ),
        "esc_code_id": torch.cat(
            (first["esc_code_id"], second["esc_code_id"])
        ),
        "esc_code_count": torch.cat(
            (first["esc_code_count"], second["esc_code_count"])
        ),
        "esc_nnz_per_edge": torch.cat(
            (first["esc_nnz_per_edge"], second["esc_nnz_per_edge"])
        ),
    }

    with torch.no_grad():
        expected_first = model(**first)
        expected_second = model(**second)
        actual = model(**batched)
        repeated = model(**batched)

    split = first["x"].size(0)
    assert torch.equal(actual, repeated)
    assert torch.allclose(actual[:split], expected_first, atol=1e-6)
    assert torch.allclose(actual[split:], expected_second, atol=1e-6)


def test_eval_output_is_equivariant_to_relabeling_and_reversed_edge_order():
    model = ESCGNN().eval()
    x = _features(5)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
    )
    original = ESCStructuralEncoding()(
        Data(edge_index=edge_index, num_nodes=5)
    )
    inputs = {
        "x": x,
        "edge_index": original.edge_index,
        "esc_code_id": original.esc_code_id,
        "esc_code_count": original.esc_code_count,
        "esc_nnz_per_edge": original.esc_nnz_per_edge,
    }
    node_permutation = torch.tensor([2, 4, 1, 0, 3], dtype=torch.long)
    edge_column_order = torch.tensor([4, 0, 3, 5, 1, 2], dtype=torch.long)
    relabeled_edge_index = node_permutation[
        edge_index.flip(0)[:, edge_column_order]
    ]
    relabeled = ESCStructuralEncoding()(
        Data(edge_index=relabeled_edge_index, num_nodes=5)
    )
    permuted_x = torch.empty_like(x)
    permuted_x[node_permutation] = x
    transformed = {
        "x": permuted_x,
        "edge_index": relabeled.edge_index,
        "esc_code_id": relabeled.esc_code_id,
        "esc_code_count": relabeled.esc_code_count,
        "esc_nnz_per_edge": relabeled.esc_nnz_per_edge,
    }

    with torch.no_grad():
        expected = model(**inputs)
        actual = model(**transformed)

    assert torch.allclose(actual[node_permutation], expected, atol=1e-6)
