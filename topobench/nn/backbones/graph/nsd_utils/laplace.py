# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for computing Laplacian indices and operations.

This module provides efficient functions for computing indices required
for sheaf Laplacian construction and sparse matrix operations.
"""

import torch


def compute_left_right_map_index(edge_index, full_matrix=False):
    """
    Compute indices for mapping edges to their reverse edges.

    For each edge in the lower triangular part (source < target) or all edges,
    find the index of its corresponding reverse edge in the original edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges] representing directed edges.
    full_matrix : bool, optional
        If True, use all edges. If False, only use edges where source < target.
        Default is False.

    Returns
    -------
    left_right_index : torch.Tensor
        Indices of shape [2, num_selected_edges] where first row contains indices
        of selected edges and second row contains indices of their reverse edges.
    new_edge_index : torch.Tensor
        Edge indices of selected edges, shape [2, num_selected_edges].
    """
    # Extract source and target nodes
    source, target = edge_index[0], edge_index[1]

    if full_matrix:
        # For full matrix, use all edges
        mask = torch.ones(
            edge_index.size(1), dtype=torch.bool, device=edge_index.device
        )
    else:
        # For lower triangular, only use edges where source < target
        mask = source < target

    # Get the indices where mask is True
    selected_indices = torch.where(mask)[0]

    if len(selected_indices) == 0:
        # Handle edge case where no edges match criteria
        empty_tensor = torch.empty(
            (2, 0), dtype=torch.long, device=edge_index.device
        )
        return empty_tensor, empty_tensor

    # Get the selected edges
    selected_source = source[selected_indices]
    selected_target = target[selected_indices]

    # Create the new edge index
    new_edge_index = torch.stack([selected_source, selected_target])

    # Create a mapping from edge pairs to their indices
    # This is the key optimization - we create a hash-like mapping using tensor operations
    edge_pairs = torch.stack([source, target], dim=1)

    # For each selected edge, we need to find its reverse edge
    # Create reverse pairs for selected edges
    reverse_pairs = torch.stack([selected_target, selected_source], dim=1)

    # Use broadcasting to find all matches at once
    # This creates a matrix where entry [i,j] is True if reverse_pairs[i] == edge_pairs[j]
    # Shape: (num_selected_edges, num_total_edges)
    matches = torch.all(
        reverse_pairs.unsqueeze(1) == edge_pairs.unsqueeze(0), dim=2
    )

    # Convert boolean to float for argmax to work on CUDA
    matches_float = matches.float()

    # For each selected edge, find the index of its reverse edge
    # argmax will find the first True value in each row
    right_index = torch.argmax(matches_float, dim=1)

    # Verify that we actually found matches (all rows should have at least one True)
    assert torch.all(torch.any(matches, dim=1)), (
        "Some reverse edges not found in original edge list"
    )

    left_index = selected_indices
    left_right_index = torch.stack([left_index, right_index])

    # Verify the expected sizes
    if full_matrix:
        assert len(selected_indices) == edge_index.size(1)
    else:
        assert len(selected_indices) == edge_index.size(1) // 2

    return left_right_index, new_edge_index


def compute_learnable_laplacian_indices(size, edge_index, learned_d, total_d):
    """
    Compute sparse indices for a learnable Laplacian with full matrices.

    Generates indices for both diagonal and non-diagonal blocks of a block-structured
    Laplacian matrix where each block is learned_d x learned_d.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges] where edge_index[0] < edge_index[1].
    learned_d : int
        Dimension of each learned block matrix.
    total_d : int
        Total dimension per node (typically equal to learned_d).

    Returns
    -------
    diag_indices : torch.Tensor
        Sparse indices for diagonal blocks, shape [2, size * learned_d * learned_d].
    non_diag_indices : torch.Tensor
        Sparse indices for non-diagonal blocks, shape [2, num_edges * learned_d * learned_d].
    """
    assert torch.all(edge_index[0] < edge_index[1])

    row, col = edge_index
    device = edge_index.device
    row_template = (
        torch.arange(0, learned_d, device=device)
        .view(1, -1, 1)
        .tile(1, 1, learned_d)
    )
    col_template = torch.transpose(row_template, dim0=1, dim1=2)

    non_diag_row_indices = (
        row_template + total_d * row.reshape(-1, 1, 1)
    ).reshape(1, -1)
    non_diag_col_indices = (
        col_template + total_d * col.reshape(-1, 1, 1)
    ).reshape(1, -1)
    non_diag_indices = torch.cat(
        (non_diag_row_indices, non_diag_col_indices), dim=0
    )

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (
        row_template + total_d * diag.reshape(-1, 1, 1)
    ).reshape(1, -1)
    diag_col_indices = (
        col_template + total_d * diag.reshape(-1, 1, 1)
    ).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def compute_learnable_diag_laplacian_indices(
    size, edge_index, learned_d, total_d
):
    """
    Compute sparse indices for a learnable Laplacian with diagonal matrices.

    Generates indices for both diagonal and non-diagonal blocks of a block-structured
    Laplacian matrix where each block is diagonal (learned_d x learned_d).

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges] where edge_index[0] < edge_index[1].
    learned_d : int
        Dimension of each learned diagonal block.
    total_d : int
        Total dimension per node (typically equal to learned_d).

    Returns
    -------
    diag_indices : torch.Tensor
        Sparse indices for diagonal blocks, shape [2, size * learned_d].
    non_diag_indices : torch.Tensor
        Sparse indices for non-diagonal blocks, shape [2, num_edges * learned_d].
    """
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d * row.unsqueeze(1)).reshape(
        1, -1
    )
    non_diag_col_indices = (col_template + total_d * col.unsqueeze(1)).reshape(
        1, -1
    )
    non_diag_indices = torch.cat(
        (non_diag_row_indices, non_diag_col_indices), dim=0
    )

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d * diag.unsqueeze(1)).reshape(
        1, -1
    )
    diag_col_indices = (col_template + total_d * diag.unsqueeze(1)).reshape(
        1, -1
    )
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


def mergesp(index1, value1, index2, value2):
    """
    Merge two sparse matrices with disjoint indices into one.

    Concatenates two sets of sparse matrix indices and values, assuming
    the indices are disjoint (no overlapping entries).

    Parameters
    ----------
    index1 : torch.Tensor
        First set of sparse indices, shape [2, num_entries1].
    value1 : torch.Tensor
        First set of values, shape [num_entries1].
    index2 : torch.Tensor
        Second set of sparse indices, shape [2, num_entries2].
    value2 : torch.Tensor
        Second set of values, shape [num_entries2].

    Returns
    -------
    index : torch.Tensor
        Merged sparse indices, shape [2, num_entries1 + num_entries2].
    val : torch.Tensor
        Merged values, shape [num_entries1 + num_entries2].
    """
    assert index1.dim() == 2 and index2.dim() == 2
    assert value1.dim() == 1 and value2.dim() == 1
    assert index1.size(1) == value1.numel()
    assert index2.size(1) == value2.numel()
    assert index1.size(0) == 2 and index2.size(0) == 2

    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val
