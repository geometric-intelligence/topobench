"""Tests for the controlled NSAPH-to-TopoBench QM9 model comparison."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from topobench.nn.backbones.combinatorial.etnn_coordinate_policy import (
    _ETNNMessagePassing,
)
from topobench.nn.backbones.combinatorial.etnn_qm9_adapter import (
    _QM9ParityBatch,
)
from topobench.nn.backbones.combinatorial.etnn_qm9_parity import (
    ETNNQM9Parity,
    load_nsaph_experiment_1_state_dict,
)


@pytest.fixture
def parity_batch() -> _QM9ParityBatch:
    """Build a finite two-molecule batch satisfying the adapter contract."""
    generator = torch.Generator().manual_seed(42)
    edge_index = {
        "0_0_2": torch.tensor(
            [
                [0, 1, 1, 2, 3, 4],
                [1, 0, 2, 1, 4, 3],
            ],
            dtype=torch.long,
        ),
        "1_1_2": torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ],
            dtype=torch.long,
        ),
    }
    return _QM9ParityBatch(
        features={
            0: torch.randn(5, 15, generator=generator),
            1: torch.randn(4, 19, generator=generator),
        },
        edge_index=edge_index,
        raw_invariants={
            relation: torch.randn(
                relation_edges.shape[1],
                5,
                generator=generator,
            )
            for relation, relation_edges in edge_index.items()
        },
        cell_batch={
            0: torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
            1: torch.tensor([0, 0, 1, 1], dtype=torch.long),
        },
        positions=torch.randn(5, 3, generator=generator),
        targets=torch.tensor([1.2, 2.4]),
        num_graphs=2,
    )


def test_qm9_parity_matches_native_parameter_count() -> None:
    """Require the exact pinned experiment_1 trainable-parameter count."""
    model = ETNNQM9Parity()

    assert (
        sum(parameter.numel() for parameter in model.parameters()) == 1_497_871
    )


def test_qm9_parity_uses_native_message_and_update_shapes() -> None:
    """Distinguish the final message SiLU from the linear-ended update MLP."""
    model = ETNNQM9Parity()
    layer = model.layers[0]

    message_mlp = layer.message_passing["0_0_2"].message_mlp
    update_mlp = layer.update["0"]

    assert isinstance(message_mlp[-1], nn.SiLU)
    assert isinstance(message_mlp[-2], nn.Linear)
    assert isinstance(update_mlp[-1], nn.Linear)
    assert isinstance(message_mlp[0], nn.Linear)
    assert message_mlp[0].in_features == 2 * 128 + 5
    assert update_mlp[0].in_features == 2 * 128


def test_shared_invariant_normalizers_run_once_per_forward(
    parity_batch: _QM9ParityBatch,
) -> None:
    """Normalize each static relation once rather than once per ETNN layer."""
    model = ETNNQM9Parity()
    call_counts = {"0_0_2": 0, "1_1_2": 0}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def count_call(
        relation: str,
    ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def hook(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            _output: Tensor,
        ) -> None:
            call_counts[relation] += 1

        return hook

    for relation, normalizer in model.inv_normalizer.items():
        handles.append(normalizer.register_forward_hook(count_call(relation)))

    try:
        model(parity_batch)
    finally:
        for handle in handles:
            handle.remove()

    assert call_counts == {"0_0_2": 1, "1_1_2": 1}


def test_qm9_parity_forward_returns_one_value_per_graph(
    parity_batch: _QM9ParityBatch,
) -> None:
    """Return finite scalar molecular predictions in native graph order."""
    model = ETNNQM9Parity()

    output = model(parity_batch)

    assert output.shape == (parity_batch.num_graphs,)
    assert torch.isfinite(output).all()


def test_qm9_parity_backward_reaches_every_trainable_parameter(
    parity_batch: _QM9ParityBatch,
) -> None:
    """Keep the complete feature-message-readout gradient path connected."""
    model = ETNNQM9Parity()

    loss = torch.nn.functional.l1_loss(
        model(parity_batch),
        parity_batch.targets,
    )
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_qm9_parity_reuses_static_invariants_when_positions_change(
    parity_batch: _QM9ParityBatch,
) -> None:
    """Ignore positions after canonical invariants have been supplied."""
    model = ETNNQM9Parity().eval()
    moved_batch = parity_batch._replace(
        positions=parity_batch.positions + 100.0,
    )

    with torch.no_grad():
        original = model(parity_batch)
        moved = model(moved_batch)

    torch.testing.assert_close(original, moved, rtol=0.0, atol=0.0)


def test_qm9_parity_state_translation_is_complete() -> None:
    """Translate every native parameter and BatchNorm buffer strictly."""
    source = ETNNQM9Parity()
    destination = ETNNQM9Parity()
    native_state = {
        _to_native_state_key(key): value.detach().clone()
        for key, value in source.state_dict().items()
    }

    load_nsaph_experiment_1_state_dict(destination, native_state)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            destination.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_qm9_parity_state_translation_rejects_missing_parameter() -> None:
    """Fail rather than silently train after incomplete native state loading."""
    model = ETNNQM9Parity()
    native_state = {
        _to_native_state_key(key): value.detach().clone()
        for key, value in model.state_dict().items()
    }
    native_state.pop("post_pool.2.bias")

    with pytest.raises(RuntimeError, match="Missing key"):
        load_nsaph_experiment_1_state_dict(model, native_state)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("hidden_channels", 64),
        ("num_layers", 6),
        ("dropout", 0.1),
        ("normalize_invariants", False),
    ],
)
def test_qm9_parity_rejects_protocol_drift(
    override: str,
    value: int | float | bool,
) -> None:
    """Reject constructor values outside the controlled native architecture."""
    kwargs = {override: value}

    with pytest.raises(ValueError, match="pinned experiment_1"):
        ETNNQM9Parity(**kwargs)


def test_qm9_parity_rejects_missing_relation(
    parity_batch: _QM9ParityBatch,
) -> None:
    """Require both same-rank native relations at the model boundary."""
    model = ETNNQM9Parity()
    invalid = parity_batch._replace(
        edge_index={"0_0_2": parity_batch.edge_index["0_0_2"]},
    )

    with pytest.raises(ValueError, match="requires exactly"):
        model(invalid)


def test_default_coordinate_policy_message_shape_is_unchanged() -> None:
    """Keep the standard coordinate-policy message MLP linear-ended."""
    default_message_passing = _ETNNMessagePassing(
        hidden_channels=8,
        edge_channels=1,
        dropout=0.0,
        activation="silu",
        use_batch_norm=False,
    )
    parity = _ETNNMessagePassing(
        hidden_channels=8,
        edge_channels=5,
        dropout=0.0,
        activation="silu",
        use_batch_norm=False,
        final_activation=True,
    )

    assert isinstance(default_message_passing.message_mlp[-1], nn.Linear)
    assert isinstance(parity.message_mlp[-1], nn.SiLU)


def _to_native_state_key(target_key: str) -> str:
    """Invert the deterministic state-key mapping for a synthetic audit."""
    native_key = target_key.replace(".edge_gate.", ".edge_inf_mlp.")
    if ".message_mlp.3." in native_key:
        native_key = native_key.replace(
            ".message_mlp.3.",
            ".message_mlp.2.",
        )
    key_parts = native_key.split(".")
    if "update" in key_parts and key_parts[-2] == "3":
        key_parts[-2] = "2"
    return ".".join(key_parts)
