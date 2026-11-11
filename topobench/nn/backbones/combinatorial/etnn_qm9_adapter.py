"""Controlled adapter for the ETNN QM9 implementation-parity study.

The official NSAPH implementation and TopoBench represent combinatorial
complex batches differently.  The parity experiment must translate that
representation without changing the scientific inputs.  This module therefore
adapts an already collated native QM9CC batch into the minimal tensor contract
needed by the TopoBench ETNN message/update core.

The adapter does not lift molecules, recompute geometry, normalize invariants,
or add TopoBench sparse relation values.  Those operations would introduce
additional variables into the comparison.  It preserves the values and order
of the native atom and bond features, relations, five raw invariant channels,
physical coordinates, targets, and graph assignments.  Returned tensors are
cloned so that one implementation cannot mutate the other's canonical inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Protocol

import torch
from torch import Tensor

_EXPERIMENT_1_RELATIONS = ("0_0_2", "1_1_2")
_EXPERIMENT_1_VISIBLE_RANKS = (0, 1)
_EXPERIMENT_1_FEATURE_CHANNELS = {0: 15, 1: 19}
_EXPERIMENT_1_FLOAT_DTYPE = torch.float32
_EXPERIMENT_1_TARGET_NAME = "mu"
_EXPERIMENT_1_TARGET_UNIT = "D"
_PHYSICAL_INVARIANT_CHANNELS = 5


class _NativeQM9Batch(Protocol):
    """Structural type required from a collated native QM9CC batch."""

    x_0: Tensor
    x_1: Tensor
    pos: Tensor
    y: Tensor
    adj_0_0_2: Tensor
    adj_1_1_2: Tensor
    num_graphs: int
    _slice_dict: Mapping[str, Tensor]


class _QM9ParityBatch(NamedTuple):
    """Native QM9 tensors aligned for the TopoBench ETNN core.

    Attributes
    ----------
    features : dict[int, Tensor]
        Rank-indexed native atom and bond feature matrices.
    edge_index : dict[str, Tensor]
        Native sender/receiver indices, preserved in their original order.
    raw_invariants : dict[str, Tensor]
        Five unnormalized physical invariant channels aligned with each edge.
    cell_batch : dict[int, Tensor]
        Graph assignment for every visible cell row.
    positions : Tensor
        Native physical rank-0 coordinates.
    targets : Tensor
        Native graph-level dipole targets in Debye.
    num_graphs : int
        Number of molecules represented by the collated batch.
    """

    features: dict[int, Tensor]
    edge_index: dict[str, Tensor]
    raw_invariants: dict[str, Tensor]
    cell_batch: dict[int, Tensor]
    positions: Tensor
    targets: Tensor
    num_graphs: int


def adapt_nsaph_qm9_batch(
    batch: _NativeQM9Batch,
    raw_invariants: Mapping[str, Tensor],
    invariant_edge_index: Mapping[str, Tensor],
    *,
    target_name: str,
    target_unit: str,
) -> _QM9ParityBatch:
    """Translate one native ``experiment_1`` batch without changing values.

    Parameters
    ----------
    batch : _NativeQM9Batch
        Collated native QM9CC batch for the atom-bond-supercell construction.
    raw_invariants : Mapping[str, Tensor]
        Unnormalized five-channel invariant tensors returned by the pinned
        NSAPH invariant function.
    invariant_edge_index : Mapping[str, Tensor]
        Exact relation tensors supplied to the pinned NSAPH invariant function.
        Pairing these tensors with ``raw_invariants`` lets the adapter verify
        that no relation was reordered between invariant computation and
        adaptation.  The invariant values themselves remain opaque and must
        come directly from that canonical provider.
    target_name : str
        Resolved QM9 target name.  The experiment_1 parity protocol requires
        ``"mu"``.
    target_unit : str
        Physical unit recorded by the comparison harness.  Dipole moment must
        be supplied in Debye, represented by ``"D"``.

    Returns
    -------
    _QM9ParityBatch
        Validated tensor copies suitable for the controlled TopoBench ETNN
        message/update comparison.

    Raises
    ------
    AttributeError
        If a required native batch attribute is absent.
    TypeError
        If a required value is not a tensor or has an incompatible dtype.
    ValueError
        If tensor shapes, edge bounds, graph boundaries, target metadata, or
        relation/invariant provenance violate the pinned ``experiment_1``
        contract.
    """
    _validate_target_protocol(target_name=target_name, target_unit=target_unit)
    if not isinstance(raw_invariants, Mapping):
        raise TypeError(
            "Native physical invariants must be provided as a mapping."
        )
    if not isinstance(invariant_edge_index, Mapping):
        raise TypeError(
            "Canonical invariant relations must be provided as a mapping."
        )
    num_graphs = _validate_num_graphs(batch)
    features = {
        rank: _require_matrix(batch, f"x_{rank}")
        for rank in _EXPERIMENT_1_VISIBLE_RANKS
    }
    _validate_native_features(features)
    positions = _require_matrix(batch, "pos")
    targets = _require_tensor(batch, "y")

    _validate_floating_tensor(
        tensor=positions,
        name="Native QM9 physical coordinates",
    )
    if positions.shape[0] != features[0].shape[0]:
        raise ValueError(
            "Native QM9 positions must have one row per rank-0 atom: "
            f"{positions.shape[0]} != {features[0].shape[0]}."
        )
    if positions.shape[1] != 3:
        raise ValueError(
            "Native QM9 physical coordinates must have three columns; "
            f"received shape {tuple(positions.shape)}."
        )
    _validate_targets(targets=targets, num_graphs=num_graphs)
    canonical_device = features[0].device
    _validate_common_tensor_contract(
        features=features,
        positions=positions,
        targets=targets,
        device=canonical_device,
    )

    cell_batch = {
        rank: _cell_batch_from_native_slices(
            batch=batch,
            rank=rank,
            num_cells=features[rank].shape[0],
            num_graphs=num_graphs,
            device=features[rank].device,
        )
        for rank in _EXPERIMENT_1_VISIBLE_RANKS
    }

    edge_index: dict[str, Tensor] = {}
    aligned_invariants: dict[str, Tensor] = {}
    for relation in _EXPERIMENT_1_RELATIONS:
        relation_index = _require_edge_index(batch, relation)
        src_rank, dst_rank = _relation_ranks(relation)
        _validate_edge_bounds(
            relation=relation,
            edge_index=relation_index,
            num_src_cells=features[src_rank].shape[0],
            num_dst_cells=features[dst_rank].shape[0],
        )
        if relation_index.device != canonical_device:
            raise ValueError(
                f"Native relation `{relation}` must share the canonical batch "
                f"device: {canonical_device} != {relation_index.device}."
            )
        _validate_relation_graph_membership(
            relation=relation,
            edge_index=relation_index,
            src_cell_batch=cell_batch[src_rank],
            dst_cell_batch=cell_batch[dst_rank],
        )

        canonical_relation_index = invariant_edge_index.get(relation)
        if canonical_relation_index is None:
            raise ValueError(
                f"Missing canonical invariant relation for `{relation}`."
            )
        _validate_invariant_relation_provenance(
            relation=relation,
            native_edge_index=relation_index,
            invariant_edge_index=canonical_relation_index,
        )

        invariant_tensor = raw_invariants.get(relation)
        if invariant_tensor is None:
            raise ValueError(
                f"Missing native physical invariants for relation `{relation}`."
            )
        _validate_raw_invariants(
            relation=relation,
            invariants=invariant_tensor,
            num_edges=relation_index.shape[1],
            device=canonical_device,
        )
        edge_index[relation] = relation_index.clone()
        aligned_invariants[relation] = invariant_tensor.clone()

    unexpected_relations = (
        set(raw_invariants) | set(invariant_edge_index)
    ) - set(_EXPERIMENT_1_RELATIONS)
    if unexpected_relations:
        unexpected = ", ".join(sorted(unexpected_relations))
        raise ValueError(
            "The experiment_1 parity adapter received unsupported invariant "
            f"relations: {unexpected}."
        )

    return _QM9ParityBatch(
        features={rank: feature.clone() for rank, feature in features.items()},
        edge_index=edge_index,
        raw_invariants=aligned_invariants,
        cell_batch=cell_batch,
        positions=positions.clone(),
        targets=targets.clone(),
        num_graphs=num_graphs,
    )


def _validate_target_protocol(target_name: str, target_unit: str) -> None:
    """Require the fixed dipole target and unit used by experiment_1.

    Parameters
    ----------
    target_name : str
        Resolved QM9 target name.
    target_unit : str
        Physical unit recorded by the comparison harness.
    """
    if target_name != _EXPERIMENT_1_TARGET_NAME:
        raise ValueError(
            "The experiment_1 parity adapter requires target `mu`; "
            f"received `{target_name}`."
        )
    if target_unit != _EXPERIMENT_1_TARGET_UNIT:
        raise ValueError(
            "The experiment_1 parity adapter requires dipole targets in "
            f"Debye (`D`); received `{target_unit}`."
        )


def _validate_num_graphs(batch: _NativeQM9Batch) -> int:
    """Return a positive native batch size.

    Parameters
    ----------
    batch : _NativeQM9Batch
        Collated native batch exposing ``num_graphs``.

    Returns
    -------
    int
        Positive number of molecules in the batch.
    """
    num_graphs = getattr(batch, "num_graphs", None)
    if type(num_graphs) is not int:
        raise TypeError("Native QM9 `num_graphs` must be an integer.")
    if num_graphs <= 0:
        raise ValueError("Native QM9 batches must contain at least one graph.")
    return num_graphs


def _require_tensor(batch: _NativeQM9Batch, attribute: str) -> Tensor:
    """Read a required tensor attribute from a native batch.

    Parameters
    ----------
    batch : _NativeQM9Batch
        Collated native batch.
    attribute : str
        Name of the required tensor attribute.

    Returns
    -------
    Tensor
        Tensor stored under ``attribute``.
    """
    if not hasattr(batch, attribute):
        raise AttributeError(
            f"Native QM9 batch is missing required attribute `{attribute}`."
        )
    value = getattr(batch, attribute)
    if not torch.is_tensor(value):
        raise TypeError(
            f"Native QM9 attribute `{attribute}` must be a tensor."
        )
    return value


def _require_matrix(batch: _NativeQM9Batch, attribute: str) -> Tensor:
    """Read a required rank-two tensor attribute.

    Parameters
    ----------
    batch : _NativeQM9Batch
        Collated native batch.
    attribute : str
        Name of the required matrix attribute.

    Returns
    -------
    Tensor
        Rank-two tensor stored under ``attribute``.
    """
    value = _require_tensor(batch, attribute)
    if value.ndim != 2:
        raise ValueError(
            f"Native QM9 attribute `{attribute}` must be rank two; "
            f"received shape {tuple(value.shape)}."
        )
    return value


def _validate_targets(targets: Tensor, num_graphs: int) -> None:
    """Validate one scalar target per molecule without changing its shape.

    Parameters
    ----------
    targets : Tensor
        Native graph-level target tensor.
    num_graphs : int
        Number of molecules in the collated batch.
    """
    _validate_floating_tensor(
        tensor=targets,
        name="Native QM9 dipole targets",
    )
    if targets.ndim not in {1, 2}:
        raise ValueError(
            "Native QM9 targets must have shape [G] or [G, 1]; "
            f"received {tuple(targets.shape)}."
        )
    if targets.shape[0] != num_graphs:
        raise ValueError(
            "Native QM9 targets must have one row per graph: "
            f"{targets.shape[0]} != {num_graphs}."
        )
    if targets.ndim == 2 and targets.shape[1] != 1:
        raise ValueError(
            "The experiment_1 adapter expects one dipole target per graph; "
            f"received shape {tuple(targets.shape)}."
        )


def _validate_native_features(features: Mapping[int, Tensor]) -> None:
    """Validate native experiment_1 feature widths and dtypes.

    Parameters
    ----------
    features : Mapping[int, Tensor]
        Rank-indexed atom and bond feature matrices.
    """
    for rank, expected_channels in _EXPERIMENT_1_FEATURE_CHANNELS.items():
        feature = features[rank]
        _validate_floating_tensor(
            tensor=feature,
            name=f"Native QM9 rank-{rank} features",
        )
        if feature.shape[1] != expected_channels:
            raise ValueError(
                f"Native QM9 rank-{rank} features must have "
                f"{expected_channels} channels; received "
                f"{feature.shape[1]}."
            )


def _validate_floating_tensor(tensor: Tensor, name: str) -> None:
    """Require finite float32 values for one parity-protocol tensor.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose numerical representation is validated.
    name : str
        Human-readable tensor name used in validation messages.
    """
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if tensor.dtype != _EXPERIMENT_1_FLOAT_DTYPE:
        raise TypeError(
            f"{name} must use {_EXPERIMENT_1_FLOAT_DTYPE}; "
            f"received {tensor.dtype}."
        )
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values.")


def _validate_common_tensor_contract(
    features: Mapping[int, Tensor],
    positions: Tensor,
    targets: Tensor,
    device: torch.device,
) -> None:
    """Require every model input to share the canonical batch device.

    Parameters
    ----------
    features : Mapping[int, Tensor]
        Rank-indexed atom and bond feature matrices.
    positions : Tensor
        Rank-0 physical coordinates.
    targets : Tensor
        Graph-level dipole targets.
    device : torch.device
        Device established by the native rank-0 feature tensor.
    """
    tensors = {
        **{f"x_{rank}": feature for rank, feature in features.items()},
        "pos": positions,
        "y": targets,
    }
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"Native QM9 tensor `{name}` must share the canonical batch "
                f"device: {device} != {tensor.device}."
            )


def _cell_batch_from_native_slices(
    batch: _NativeQM9Batch,
    rank: int,
    num_cells: int,
    num_graphs: int,
    device: torch.device,
) -> Tensor:
    """Convert native PyG collation boundaries to cell graph assignments.

    Parameters
    ----------
    batch : _NativeQM9Batch
        Collated native batch with PyG ``_slice_dict`` metadata.
    rank : int
        Visible cell rank whose graph assignments are required.
    num_cells : int
        Number of feature rows at ``rank``.
    num_graphs : int
        Number of molecules in the batch.
    device : torch.device
        Device on which to construct the assignment tensor.

    Returns
    -------
    Tensor
        Integer graph index for every cell row at ``rank``.
    """
    slice_dict = getattr(batch, "_slice_dict", None)
    if not isinstance(slice_dict, Mapping):
        raise TypeError(
            "Native QM9 batches must expose PyG collation boundaries through "
            "`_slice_dict`."
        )

    key = f"slices_{rank}"
    boundaries = slice_dict.get(key)
    if boundaries is None:
        raise KeyError(f"Native QM9 batch is missing collation key `{key}`.")
    if not torch.is_tensor(boundaries):
        raise TypeError(f"Native QM9 collation key `{key}` must be a tensor.")
    if boundaries.ndim != 1 or boundaries.numel() != num_graphs + 1:
        raise ValueError(
            f"Native QM9 collation key `{key}` must contain G + 1 "
            f"boundaries; received shape {tuple(boundaries.shape)} for "
            f"G={num_graphs}."
        )
    if boundaries.dtype != torch.long:
        raise TypeError(
            f"Native QM9 collation key `{key}` must use torch.long boundaries."
        )

    boundaries = boundaries.to(device=device)
    if boundaries[0].item() != 0:
        raise ValueError(f"Native QM9 collation key `{key}` must start at 0.")
    counts = boundaries[1:] - boundaries[:-1]
    if torch.any(counts < 0):
        raise ValueError(
            f"Native QM9 collation key `{key}` must be nondecreasing."
        )
    if boundaries[-1].item() != num_cells:
        raise ValueError(
            f"Native QM9 collation key `{key}` describes "
            f"{boundaries[-1].item()} cells, but x_{rank} has {num_cells}."
        )

    graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
    return torch.repeat_interleave(graph_ids, counts)


def _require_edge_index(batch: _NativeQM9Batch, relation: str) -> Tensor:
    """Read and validate one native sender/receiver relation tensor.

    Parameters
    ----------
    batch : _NativeQM9Batch
        Collated native batch.
    relation : str
        Native relation identifier without the ``adj_`` prefix.

    Returns
    -------
    Tensor
        Native edge indices with shape ``[2, E]``.
    """
    edge_index = _require_tensor(batch, f"adj_{relation}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Native relation `{relation}` must have shape [2, E]; "
            f"received {tuple(edge_index.shape)}."
        )
    if edge_index.dtype != torch.long:
        raise TypeError(
            f"Native relation `{relation}` must use torch.long indices."
        )
    return edge_index


def _relation_ranks(relation: str) -> tuple[int, int]:
    """Return sender and receiver ranks encoded by a native relation name.

    Parameters
    ----------
    relation : str
        Native three-component relation identifier.

    Returns
    -------
    tuple[int, int]
        Sender rank followed by receiver rank.
    """
    parts = relation.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unsupported native relation name `{relation}`.")
    return int(parts[0]), int(parts[1])


def _validate_edge_bounds(
    relation: str,
    edge_index: Tensor,
    num_src_cells: int,
    num_dst_cells: int,
) -> None:
    """Ensure native relation indices address existing feature rows.

    Parameters
    ----------
    relation : str
        Native relation identifier used in validation messages.
    edge_index : Tensor
        Native sender/receiver indices with shape ``[2, E]``.
    num_src_cells : int
        Number of valid sender feature rows.
    num_dst_cells : int
        Number of valid receiver feature rows.
    """
    if edge_index.numel() == 0:
        return
    sender, receiver = edge_index
    if sender.min().item() < 0 or sender.max().item() >= num_src_cells:
        raise ValueError(
            f"Native relation `{relation}` contains a sender index outside "
            f"[0, {num_src_cells})."
        )
    if receiver.min().item() < 0 or receiver.max().item() >= num_dst_cells:
        raise ValueError(
            f"Native relation `{relation}` contains a receiver index outside "
            f"[0, {num_dst_cells})."
        )


def _validate_relation_graph_membership(
    relation: str,
    edge_index: Tensor,
    src_cell_batch: Tensor,
    dst_cell_batch: Tensor,
) -> None:
    """Reject relation edges that connect cells from different molecules.

    Parameters
    ----------
    relation : str
        Native relation identifier used in validation messages.
    edge_index : Tensor
        Native sender/receiver indices with shape ``[2, E]``.
    src_cell_batch : Tensor
        Graph assignment for each sender-rank cell.
    dst_cell_batch : Tensor
        Graph assignment for each receiver-rank cell.
    """
    if edge_index.numel() == 0:
        return
    sender, receiver = edge_index
    sender_graph = src_cell_batch[sender]
    receiver_graph = dst_cell_batch[receiver]
    mismatches = torch.nonzero(
        sender_graph != receiver_graph,
        as_tuple=False,
    ).flatten()
    if mismatches.numel() == 0:
        return

    edge_offset = mismatches[0].item()
    raise ValueError(
        f"Native relation `{relation}` edge {edge_offset} crosses molecule "
        f"boundaries: graph {sender_graph[edge_offset].item()} -> "
        f"graph {receiver_graph[edge_offset].item()}."
    )


def _validate_invariant_relation_provenance(
    relation: str,
    native_edge_index: Tensor,
    invariant_edge_index: Tensor,
) -> None:
    """Verify the edge order used by the canonical invariant provider.

    Parameters
    ----------
    relation : str
        Native relation identifier used in validation messages.
    native_edge_index : Tensor
        Relation tensor read from the collated native batch.
    invariant_edge_index : Tensor
        Relation tensor supplied to the pinned NSAPH invariant function.
    """
    if not torch.is_tensor(invariant_edge_index):
        raise TypeError(
            f"Canonical invariant relation `{relation}` must be a tensor."
        )
    if invariant_edge_index.ndim != 2 or invariant_edge_index.shape[0] != 2:
        raise ValueError(
            f"Canonical invariant relation `{relation}` must have shape "
            f"[2, E]; received {tuple(invariant_edge_index.shape)}."
        )
    if invariant_edge_index.dtype != torch.long:
        raise TypeError(
            f"Canonical invariant relation `{relation}` must use torch.long "
            "indices."
        )
    if invariant_edge_index.device != native_edge_index.device:
        raise ValueError(
            f"Canonical invariant relation `{relation}` and its native "
            "relation must share a device."
        )
    if not torch.equal(invariant_edge_index, native_edge_index):
        raise ValueError(
            f"Canonical invariant relation `{relation}` does not match the "
            "native relation edge order."
        )


def _validate_raw_invariants(
    relation: str,
    invariants: Tensor,
    num_edges: int,
    device: torch.device,
) -> None:
    """Validate native physical invariant width, alignment, and placement.

    Parameters
    ----------
    relation : str
        Native relation identifier used in validation messages.
    invariants : Tensor
        Raw physical invariant channels aligned with relation edges.
    num_edges : int
        Number of edges in the corresponding native relation.
    device : torch.device
        Device holding the corresponding edge-index tensor.
    """
    if not torch.is_tensor(invariants):
        raise TypeError(
            f"Native invariants for relation `{relation}` must be a tensor."
        )
    expected_shape = (num_edges, _PHYSICAL_INVARIANT_CHANNELS)
    if tuple(invariants.shape) != expected_shape:
        raise ValueError(
            f"Native invariants for relation `{relation}` must have shape "
            f"{expected_shape}; received {tuple(invariants.shape)}."
        )
    _validate_floating_tensor(
        tensor=invariants,
        name=f"Native invariants for relation `{relation}`",
    )
    if invariants.device != device:
        raise ValueError(
            f"Native relation `{relation}` and its invariants must share a "
            f"device: {device} != {invariants.device}."
        )
