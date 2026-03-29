"""Budget utilities for structure-centric sampling.

This module provides utilities for managing node budgets when sampling
structures, ensuring batches stay within memory constraints.
"""

from typing import Any


def estimate_nodes_for_structures(
    structures: list[tuple[int, list[int]]],
) -> int:
    """Estimate unique node count across structures.

    Parameters
    ----------
    structures : list of (int, list of int)
        List of (structure_id, nodes) tuples.

    Returns
    -------
    int
        Number of unique nodes across all structures.
    """
    node_set = set()
    for _, nodes in structures:
        node_set.update(nodes)
    return len(node_set)


def select_structures_within_budget(
    structure_ids: list[int],
    query_engine: Any,
    node_budget: int,
    structures_per_batch: int | None = None,
) -> tuple[list[int], int]:
    """Select structures that fit within node budget.

    This function greedily selects structures from the provided list
    until adding another structure would exceed the node budget.

    Parameters
    ----------
    structure_ids : list of int
        Candidate structure IDs to select from.
    query_engine : StructureQueryEngine
        Query engine for retrieving structure nodes.
    node_budget : int
        Maximum number of unique nodes allowed in the batch.
    structures_per_batch : int, optional
        Maximum number of structures per batch (default: None for unlimited).

    Returns
    -------
    selected_ids : list of int
        Selected structure IDs that fit within budget.
    node_count : int
        Total unique node count for selected structures.

    Notes
    -----
    This uses a greedy approach which may not be optimal but is fast.
    The selection order matters - structures appearing earlier are
    prioritized.
    """
    selected = []
    node_set = set()

    for sid in structure_ids:
        # Check structure count limit
        if structures_per_batch and len(selected) >= structures_per_batch:
            break

        # Query structure nodes
        structures = query_engine.query_cliques_by_id([sid])
        if not structures:
            continue  # Skip if structure not found

        _, nodes = structures[0]

        # Check if adding this structure would exceed budget
        new_node_set = node_set | set(nodes)
        if len(new_node_set) <= node_budget:
            selected.append(sid)
            node_set = new_node_set
        else:
            # Budget exceeded, stop adding structures
            break

    return selected, len(node_set)


def estimate_batch_memory(
    num_nodes: int,
    num_edges: int,
    num_features: int = 128,
    feature_dtype_bytes: int = 4,
) -> float:
    """Estimate memory usage for a batch in MB.

    This provides a rough estimate of memory needed for a batch
    based on node count, edge count, and feature dimensions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the batch.
    num_edges : int
        Number of edges in the batch.
    num_features : int, optional
        Feature dimension per node (default: 128).
    feature_dtype_bytes : int, optional
        Bytes per feature value (default: 4 for float32).

    Returns
    -------
    float
        Estimated memory usage in MB.
    """
    # Node features
    node_features_bytes = num_nodes * num_features * feature_dtype_bytes

    # Edge indices (2 * num_edges for COO format, int64)
    edge_indices_bytes = num_edges * 2 * 8

    # Edge features (assume same dimension as node features)
    edge_features_bytes = num_edges * num_features * feature_dtype_bytes

    # Total in MB
    total_bytes = (
        node_features_bytes + edge_indices_bytes + edge_features_bytes
    )
    total_mb = total_bytes / (1024 * 1024)

    return total_mb


def calculate_optimal_batch_size(
    avg_structure_size: float,
    node_budget: int = 2000,
    target_structures: int = 500,
) -> dict[str, int]:
    """Calculate optimal batch configuration.

    Parameters
    ----------
    avg_structure_size : float
        Average number of nodes per structure.
    node_budget : int, optional
        Target node budget per batch (default: 2000).
    target_structures : int, optional
        Target number of structures per batch (default: 500).

    Returns
    -------
    dict
        Configuration with keys:
        - structures_per_batch: Recommended structures per batch
        - node_budget: Node budget to use
        - expected_nodes: Expected unique nodes per batch

    Notes
    -----
    This assumes structures overlap, so unique nodes < structures * avg_size.
    """
    # Estimate unique nodes (assuming 2x overlap factor)
    overlap_factor = 2.0
    expected_nodes = int(
        target_structures * avg_structure_size / overlap_factor
    )

    # Adjust if exceeds budget
    if expected_nodes > node_budget:
        # Reduce structures to fit budget
        adjusted_structures = int(
            node_budget * overlap_factor / avg_structure_size
        )
        structures_per_batch = max(1, adjusted_structures)
        expected_nodes = node_budget
    else:
        structures_per_batch = target_structures

    return {
        "structures_per_batch": structures_per_batch,
        "node_budget": node_budget,
        "expected_nodes": expected_nodes,
    }
