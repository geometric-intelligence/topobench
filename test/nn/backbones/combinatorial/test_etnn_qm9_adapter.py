"""Tests for the controlled NSAPH-QM9-to-TopoBench ETNN adapter.

The controlled QM9 study uses the official processed molecular batches on
both sides of the comparison.  These tests protect the adapter boundary:
native feature rows, relation direction and order, physical invariant rows,
coordinates, targets, and graph assignments must reach the TopoBench ETNN core
without reinterpretation or cross-molecule leakage.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.combinatorial.etnn_qm9_adapter import (
    _QM9ParityBatch,
    adapt_nsaph_qm9_batch,
)

_BatchType = SimpleNamespace | Batch


class _NativeComplexData(Data):
    """Minimal PyG data class with the native relation batching semantics."""

    def __inc__(
        self,
        key: str,
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor | int:
        """Offset native relation rows by their sender/receiver cell counts."""
        if key.startswith("adj_"):
            src_rank, dst_rank = (int(rank) for rank in key.split("_")[1:3])
            return torch.tensor(
                [
                    [getattr(self, f"x_{src_rank}").shape[0]],
                    [getattr(self, f"x_{dst_rank}").shape[0]],
                ],
                dtype=torch.long,
            )
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(
        self,
        key: str,
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> int | tuple[int, int]:
        """Concatenate native relation tensors along their edge dimension."""
        if key.startswith("adj_"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _create_native_qm9_batch() -> SimpleNamespace:
    """Create a two-molecule batch with the native experiment_1 schema."""
    x_0 = torch.arange(5 * 15, dtype=torch.float32).reshape(5, 15)
    x_1 = torch.arange(3 * 19, dtype=torch.float32).reshape(3, 19) + 100.0
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )

    # Deliberately non-sorted edge order makes accidental relation reordering
    # visible. The first graph owns rank-0 rows 0:2 and rank-1 row 0; the
    # second graph owns the remaining rows.
    adjacency_0 = torch.tensor(
        [[1, 0, 4, 2, 3], [0, 1, 2, 4, 2]], dtype=torch.long
    )
    adjacency_1 = torch.tensor([[0, 2, 1], [0, 1, 2]], dtype=torch.long)
    return SimpleNamespace(
        x_0=x_0,
        x_1=x_1,
        pos=positions,
        y=torch.tensor([[1.25], [2.50]], dtype=torch.float32),
        adj_0_0_2=adjacency_0,
        adj_1_1_2=adjacency_1,
        num_graphs=2,
        _slice_dict={
            "slices_0": torch.tensor([0, 2, 5], dtype=torch.long),
            "slices_1": torch.tensor([0, 1, 3], dtype=torch.long),
        },
    )


def _create_raw_invariants(batch: _BatchType) -> dict[str, Tensor]:
    """Create distinct five-channel rows aligned with each native edge."""
    return {
        "0_0_2": torch.arange(
            batch.adj_0_0_2.shape[1] * 5, dtype=torch.float32
        ).reshape(-1, 5),
        "1_1_2": (
            torch.arange(
                batch.adj_1_1_2.shape[1] * 5, dtype=torch.float32
            ).reshape(-1, 5)
            + 1000.0
        ),
    }


def _create_invariant_edges(batch: _BatchType) -> dict[str, Tensor]:
    """Capture the exact relations supplied to the invariant provider."""
    return {
        "0_0_2": batch.adj_0_0_2.clone(),
        "1_1_2": batch.adj_1_1_2.clone(),
    }


def _adapt(
    batch: _BatchType,
    invariants: dict[str, Tensor] | None = None,
    invariant_edges: dict[str, Tensor] | None = None,
    *,
    target_name: str = "mu",
    target_unit: str = "D",
) -> _QM9ParityBatch:
    """Run the adapter under the fixed experiment_1 target contract."""
    canonical_invariants = (
        _create_raw_invariants(batch) if invariants is None else invariants
    )
    canonical_edges = (
        _create_invariant_edges(batch)
        if invariant_edges is None
        else invariant_edges
    )
    return adapt_nsaph_qm9_batch(
        batch,
        canonical_invariants,
        canonical_edges,
        target_name=target_name,
        target_unit=target_unit,
    )


def test_adapter_preserves_native_tensor_order_and_values() -> None:
    """The adapter should preserve values while isolating mutable inputs."""
    batch = _create_native_qm9_batch()
    invariants = _create_raw_invariants(batch)
    invariant_edges = _create_invariant_edges(batch)

    adapted = _adapt(batch, invariants, invariant_edges)

    assert adapted.num_graphs == 2
    assert torch.equal(adapted.features[0], batch.x_0)
    assert torch.equal(adapted.features[1], batch.x_1)
    assert torch.equal(adapted.positions, batch.pos)
    assert torch.equal(adapted.targets, batch.y)
    assert torch.equal(adapted.edge_index["0_0_2"], batch.adj_0_0_2)
    assert torch.equal(adapted.edge_index["1_1_2"], batch.adj_1_1_2)
    assert torch.equal(adapted.raw_invariants["0_0_2"], invariants["0_0_2"])
    assert torch.equal(adapted.raw_invariants["1_1_2"], invariants["1_1_2"])

    # Independent tensors prevent one implementation from contaminating the
    # canonical batch used by the other implementation.
    assert adapted.features[0] is not batch.x_0
    assert adapted.positions is not batch.pos
    assert adapted.targets is not batch.y
    assert adapted.edge_index["0_0_2"] is not batch.adj_0_0_2
    assert adapted.raw_invariants["0_0_2"] is not invariants["0_0_2"]

    adapted.features[0][0, 0] = -1.0
    adapted.positions[0, 0] = -1.0
    adapted.targets[0, 0] = -1.0
    adapted.edge_index["0_0_2"][0, 0] = 0
    adapted.raw_invariants["0_0_2"][0, 0] = -1.0
    assert batch.x_0[0, 0].item() == 0.0
    assert batch.pos[0, 0].item() == 0.0
    assert batch.y[0, 0].item() == 1.25
    assert batch.adj_0_0_2[0, 0].item() == 1
    assert invariants["0_0_2"][0, 0].item() == 0.0

    assert torch.equal(
        adapted.cell_batch[0],
        torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
    )
    assert torch.equal(
        adapted.cell_batch[1],
        torch.tensor([0, 1, 1], dtype=torch.long),
    )
    assert set(adapted.raw_invariants) == {"0_0_2", "1_1_2"}


def test_adapter_accepts_vector_targets_and_empty_relations() -> None:
    """A valid scalar-target batch may contain an empty same-rank relation."""
    batch = _create_native_qm9_batch()
    batch.y = batch.y.squeeze(-1)
    batch.adj_1_1_2 = torch.empty((2, 0), dtype=torch.long)

    adapted = _adapt(batch)

    assert adapted.targets.shape == (2,)
    assert adapted.edge_index["1_1_2"].shape == (2, 0)
    assert adapted.raw_invariants["1_1_2"].shape == (0, 5)


@pytest.mark.parametrize(
    ("relation", "shape"),
    [
        ("0_0_2", (4, 5)),
        ("0_0_2", (5, 3)),
        ("1_1_2", (3, 4)),
    ],
)
def test_adapter_rejects_misaligned_invariant_shapes(
    relation: str,
    shape: tuple[int, int],
) -> None:
    """Invariant rows and channels must match native relation edges exactly."""
    batch = _create_native_qm9_batch()
    invariants = _create_raw_invariants(batch)
    invariants[relation] = torch.zeros(shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="must have shape"):
        _adapt(batch, invariants)


def test_adapter_rejects_reordered_invariant_relation() -> None:
    """Invariant provenance must use the native relation edge order."""
    batch = _create_native_qm9_batch()
    invariant_edges = _create_invariant_edges(batch)
    invariant_edges["0_0_2"] = invariant_edges["0_0_2"].flip(1)

    with pytest.raises(ValueError, match="does not match.*edge order"):
        _adapt(batch, invariant_edges=invariant_edges)


def test_adapter_rejects_unexpected_invariant_relation() -> None:
    """The experiment_1 comparison must not silently add another relation."""
    batch = _create_native_qm9_batch()
    invariants = _create_raw_invariants(batch)
    invariants["0_1_1"] = torch.zeros((1, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="unsupported invariant relations"):
        _adapt(batch, invariants)


def test_adapter_rejects_out_of_bounds_native_edges() -> None:
    """Invalid sender or receiver rows must fail before message passing."""
    batch = _create_native_qm9_batch()
    batch.adj_1_1_2 = batch.adj_1_1_2.clone()
    batch.adj_1_1_2[1, 0] = batch.x_1.shape[0]

    with pytest.raises(ValueError, match="receiver index outside"):
        _adapt(batch)


def test_adapter_rejects_cross_molecule_edges() -> None:
    """Shape-valid edges must not connect cells from different molecules."""
    batch = _create_native_qm9_batch()
    batch.adj_0_0_2 = batch.adj_0_0_2.clone()
    batch.adj_0_0_2[:, 0] = torch.tensor([0, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="crosses molecule boundaries"):
        _adapt(batch)


def test_adapter_rejects_inconsistent_native_cell_boundaries() -> None:
    """PyG collation boundaries must account for every native feature row."""
    batch = _create_native_qm9_batch()
    batch._slice_dict["slices_1"] = torch.tensor([0, 1, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="x_1 has 3"):
        _adapt(batch)


def test_adapter_rejects_noninteger_native_cell_boundaries() -> None:
    """Float boundaries must not be silently truncated to graph assignments."""
    batch = _create_native_qm9_batch()
    batch._slice_dict["slices_0"] = torch.tensor([0.0, 2.9, 5.0])

    with pytest.raises(TypeError, match="must use torch.long boundaries"):
        _adapt(batch)


def test_adapter_rejects_target_count_mismatch() -> None:
    """The native batch must supply exactly one dipole target per molecule."""
    batch = _create_native_qm9_batch()
    batch.y = torch.tensor([[1.25]], dtype=torch.float32)

    with pytest.raises(ValueError, match="one row per graph"):
        _adapt(batch)


@pytest.mark.parametrize(
    ("target_name", "target_unit", "message"),
    [
        ("alpha", "D", "requires target `mu`"),
        ("mu", "eV", "requires dipole targets in Debye"),
    ],
)
def test_adapter_rejects_wrong_target_protocol(
    target_name: str,
    target_unit: str,
    message: str,
) -> None:
    """The comparison must bind the resolved target identity and physical unit."""
    batch = _create_native_qm9_batch()

    with pytest.raises(ValueError, match=message):
        _adapt(batch, target_name=target_name, target_unit=target_unit)


@pytest.mark.parametrize(
    ("attribute", "shape"),
    [
        ("x_0", (5, 11)),
        ("x_1", (3, 15)),
    ],
)
def test_adapter_rejects_non_native_feature_widths(
    attribute: str,
    shape: tuple[int, int],
) -> None:
    """Generic PyG feature matrices must not pass as native QM9CC features."""
    batch = _create_native_qm9_batch()
    setattr(batch, attribute, torch.zeros(shape, dtype=torch.float32))

    with pytest.raises(ValueError, match="features must have"):
        _adapt(batch)


@pytest.mark.parametrize("attribute", ["x_0", "x_1", "pos", "y"])
def test_adapter_rejects_noncanonical_float_dtype(attribute: str) -> None:
    """Every floating input must use the native float32 representation."""
    batch = _create_native_qm9_batch()
    setattr(batch, attribute, getattr(batch, attribute).to(torch.float64))

    with pytest.raises(TypeError, match="must use torch.float32"):
        _adapt(batch)


def test_adapter_rejects_noncanonical_invariant_dtype() -> None:
    """Physical invariants must share the native float32 representation."""
    batch = _create_native_qm9_batch()
    invariants = _create_raw_invariants(batch)
    invariants["0_0_2"] = invariants["0_0_2"].to(torch.float64)

    with pytest.raises(TypeError, match="must use torch.float32"):
        _adapt(batch, invariants)


@pytest.mark.parametrize("attribute", ["x_0", "pos", "y"])
def test_adapter_rejects_nonfinite_model_inputs(attribute: str) -> None:
    """NaN or infinite model inputs must fail before either implementation."""
    batch = _create_native_qm9_batch()
    tensor = getattr(batch, attribute).clone()
    tensor.reshape(-1)[0] = torch.nan
    setattr(batch, attribute, tensor)

    with pytest.raises(ValueError, match="contains non-finite"):
        _adapt(batch)


def test_adapter_rejects_nonfinite_physical_invariants() -> None:
    """NaN or infinite geometry must not enter the controlled comparison."""
    batch = _create_native_qm9_batch()
    invariants = _create_raw_invariants(batch)
    invariants["0_0_2"][0, 0] = torch.nan

    with pytest.raises(ValueError, match="non-finite"):
        _adapt(batch, invariants)


def test_adapter_accepts_real_pyg_batch_boundaries() -> None:
    """Exercise the adapter against actual PyG collation metadata and offsets."""
    graph_0 = _NativeComplexData(
        x_0=torch.arange(2 * 15, dtype=torch.float32).reshape(2, 15),
        x_1=torch.arange(19, dtype=torch.float32).reshape(1, 19),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        y=torch.tensor([1.25], dtype=torch.float32),
        adj_0_0_2=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        adj_1_1_2=torch.tensor([[0], [0]], dtype=torch.long),
        slices_0=torch.ones(2, dtype=torch.long),
        slices_1=torch.ones(1, dtype=torch.long),
        num_nodes=2,
    )
    graph_1 = _NativeComplexData(
        x_0=torch.arange(3 * 15, dtype=torch.float32).reshape(3, 15),
        x_1=torch.arange(2 * 19, dtype=torch.float32).reshape(2, 19),
        pos=torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.5, 1.5, 0.0]]),
        y=torch.tensor([2.50], dtype=torch.float32),
        adj_0_0_2=torch.tensor([[2, 0, 1], [0, 2, 1]], dtype=torch.long),
        adj_1_1_2=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        slices_0=torch.ones(3, dtype=torch.long),
        slices_1=torch.ones(2, dtype=torch.long),
        num_nodes=3,
    )
    batch = Batch.from_data_list([graph_0, graph_1])

    adapted = _adapt(batch)

    assert batch._slice_dict["slices_0"].tolist() == [0, 2, 5]
    assert batch._slice_dict["slices_1"].tolist() == [0, 1, 3]
    assert adapted.cell_batch[0].tolist() == [0, 0, 1, 1, 1]
    assert adapted.cell_batch[1].tolist() == [0, 1, 1]
    assert torch.equal(adapted.cell_batch[0], batch.batch)
