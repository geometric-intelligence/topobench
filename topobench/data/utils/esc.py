"""ESC cache checks shared by preprocessing and model code."""

import torch
from torch_geometric.data import Data

ESC_HOP_RADIUS = 1
ESC_DEGREE_BINS = 300
ESC_DISTANCE_BASE = ESC_HOP_RADIUS + 2
ESC_DISTANCE_OFFSET = ESC_DEGREE_BINS
ESC_EDGE_OFFSET = ESC_DEGREE_BINS + 2 * ESC_DISTANCE_BASE
ESC_NUM_STRUCTURAL_CODES = ESC_EDGE_OFFSET + ESC_DISTANCE_BASE**4
ESC_CACHE_FIELDS = (
    "esc_code_id",
    "esc_code_count",
    "esc_nnz_per_edge",
)


def require_esc_tensors(
    data: Data, *, context: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch cached ESC tensors or fail with caller context.

    Field lookup stays separate from tensor checks. Callers can choose either
    step.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data with ``edge_index`` and cached ESC tensors.
    context : str
        Component name added to validation errors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Edge index, code IDs, counts, and nonzero counts per edge.

    Raises
    ------
    ValueError
        When any required field is missing.

    See Also
    --------
    validate_esc_tensors : Validate the returned sparse representation.
    """
    missing = [field for field in ESC_CACHE_FIELDS if data.get(field) is None]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            f"{context}: missing cached ESC field(s): {fields}. Select "
            "transforms=model_defaults/esc_gnn so ESC preprocessing runs "
            "before batching."
        )
    if data.get("edge_index") is None:
        raise ValueError(f"{context}: missing edge_index")
    return (
        data.edge_index,
        data.esc_code_id,
        data.esc_code_count,
        data.esc_nnz_per_edge,
    )


def validate_esc_tensors(
    edge_index: torch.Tensor,
    esc_code_id: torch.Tensor,
    esc_code_count: torch.Tensor,
    esc_nnz_per_edge: torch.Tensor,
    num_structural_codes: int = ESC_NUM_STRUCTURAL_CODES,
    *,
    context: str = "ESC structural encoding",
) -> None:
    """Reject malformed ESC cache tensors before model use.

    Checks shape, dtype, device, routing, range, and counts.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edges, shape ``[2, E]``.
    esc_code_id : torch.Tensor
        Flat structural code IDs, shape ``[K]``.
    esc_code_count : torch.Tensor
        Positive integer counts, shape ``[K]``.
    esc_nnz_per_edge : torch.Tensor
        Stored-code counts per edge, shape ``[E]``.
    num_structural_codes : int, optional
        Exclusive upper bound for code IDs.
    context : str, optional
        Component name added to validation errors.

    Raises
    ------
    ValueError
        When any tensor breaks ESC cache format.

    Examples
    --------
    >>> edge_index = torch.empty((2, 0), dtype=torch.long)
    >>> code_id = torch.empty(0, dtype=torch.long)
    >>> count = torch.empty(0, dtype=torch.float32)
    >>> nnz = torch.empty(0, dtype=torch.long)
    >>> validate_esc_tensors(edge_index, code_id, count, nnz)
    """
    tensors = {
        "edge_index": edge_index,
        "esc_code_id": esc_code_id,
        "esc_code_count": esc_code_count,
        "esc_nnz_per_edge": esc_nnz_per_edge,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{context}: {name} must be a torch.Tensor")

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{context}: edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise ValueError(f"{context}: edge_index must have dtype torch.long")
    sparse_tensors = (
        esc_code_id,
        esc_code_count,
        esc_nnz_per_edge,
    )
    if any(tensor.ndim != 1 for tensor in sparse_tensors):
        raise ValueError(
            f"{context}: cached ESC tensors must be one-dimensional"
        )
    if esc_code_id.dtype != torch.long:
        raise ValueError(f"{context}: esc_code_id must have dtype torch.long")
    if esc_code_count.dtype != torch.float32:
        raise ValueError(
            f"{context}: esc_code_count must have dtype torch.float32"
        )
    if esc_nnz_per_edge.dtype != torch.long:
        raise ValueError(
            f"{context}: esc_nnz_per_edge must have dtype torch.long"
        )
    if any(tensor.device != edge_index.device for tensor in sparse_tensors):
        raise ValueError(
            f"{context}: cached ESC tensors and edge_index must share a device"
        )

    num_edges = edge_index.size(1)
    num_codes = esc_code_id.numel()
    if esc_nnz_per_edge.numel() != num_edges:
        raise ValueError(
            f"{context}: len(esc_nnz_per_edge) must equal the number of edges"
        )
    if esc_code_count.numel() != num_codes:
        raise ValueError(
            f"{context}: esc_code_id and esc_code_count must have equal length"
        )

    value_invariants = torch.stack(
        (
            torch.all(esc_nnz_per_edge >= 0),
            esc_nnz_per_edge.sum() == num_codes,
            torch.all(
                (esc_code_id >= 0) & (esc_code_id < num_structural_codes)
            ),
            torch.all(torch.isfinite(esc_code_count)),
            torch.all(
                (esc_code_count > 0)
                & (esc_code_count == esc_code_count.floor())
            ),
        )
    )
    if bool(value_invariants.all().item()):
        return

    # keep useful diagnostics without another device sync
    if torch.any(esc_nnz_per_edge < 0):
        raise ValueError(
            f"{context}: esc_nnz_per_edge must contain nonnegative integers"
        )
    if int(esc_nnz_per_edge.sum().item()) != num_codes:
        raise ValueError(
            f"{context}: sum(esc_nnz_per_edge) must equal the number of codes"
        )

    if torch.any(esc_code_id < 0) or torch.any(
        esc_code_id >= num_structural_codes
    ):
        raise ValueError(
            f"{context}: esc_code_id values must lie in "
            f"[0, {num_structural_codes})"
        )
    if not torch.all(torch.isfinite(esc_code_count)):
        raise ValueError(
            f"{context}: esc_code_count must contain finite values"
        )
    if torch.any(esc_code_count <= 0) or not torch.equal(
        esc_code_count, esc_code_count.floor()
    ):
        raise ValueError(
            f"{context}: esc_code_count must contain positive integer-valued counts"
        )


__all__ = [
    "ESC_CACHE_FIELDS",
    "ESC_DEGREE_BINS",
    "ESC_DISTANCE_BASE",
    "ESC_DISTANCE_OFFSET",
    "ESC_EDGE_OFFSET",
    "ESC_HOP_RADIUS",
    "ESC_NUM_STRUCTURAL_CODES",
    "require_esc_tensors",
    "validate_esc_tensors",
]
