r"""Structural index utilities for the directed sheaf Laplacian.

The sheaf Laplacian of Eq. 2-3 is a block matrix with one :math:`d \times d`
block per ordered node pair. Assembling it sparsely needs three pieces of
bookkeeping, all independent of the learned values and all computed here:

1. a **symmetrized support**: the skeleton of :math:`\Gamma(u)` in Eq. 3, which
   is incident to :math:`u` regardless of direction. Directions are carried
   separately, as a per-pair phase, by :mod:`.phase`. This is the
   MagNet/SigMaNet recipe: build structure on the symmetrized graph, read the
   phase off the directed one.
2. for every unordered pair :math:`\{a, b\}` with :math:`a < b`, the rows of
   the support holding the arc :math:`(a, b)` and its reverse :math:`(b, a)`,
   so the two restriction maps of the edge can be gathered.
3. sparse element indices for the block layout.

Both (1) and (2) are :math:`O(m \log m)`. The Neural Sheaf Diffusion port in
TopoBench does (2) with an ``[m/2, m, 2]`` broadcast comparison
(``reverse_pairs.unsqueeze(1) == edge_pairs.unsqueeze(0)``), which is
:math:`O(m^2)` in memory: a 16-graph GraphUniverse batch at average degree 4-5
reaches roughly 24k arcs and therefore hundreds of megabytes per forward pass,
and the synthetic experiment of the paper's Fig. 2 (~5.3e5 arcs) would need
hundreds of gigabytes. Here the reverse arc is found by sorting integer keys
``u * num_nodes + v`` and running :func:`torch.searchsorted`.

That port also assumes a self-loop-free graph -- its ``source < target`` mask
drops self-loops and then asserts the survivors number exactly half the arcs,
which fails as soon as two self-loops exist. :func:`symmetrize_support` removes
self-loops up front: a self-loop carries no direction
(:math:`A_{uu} - A_{uu} = 0`), so it contributes nothing the phase can act on.

The block element layout is row-major and matches that port exactly, because
the Hermitian assembly in :mod:`.laplacian_builders` depends on it: swapping
the element-level row and column indices of a block transposes that block, so
swap-and-conjugate produces the conjugate transpose required by Eq. 2.
"""

import torch


def edge_keys(edge_index, num_nodes: int):
    """Encode each arc as a single integer key ``u * num_nodes + v``.

    Injective as long as every node index is below ``num_nodes``, which lets
    set operations on arcs use integer sorting instead of pair comparisons.

    Parameters
    ----------
    edge_index : torch.Tensor
        Arc indices of shape ``[2, num_edges]``.
    num_nodes : int
        Number of nodes; the radix of the encoding.

    Returns
    -------
    torch.Tensor
        Keys of shape ``[num_edges]`` and dtype ``torch.int64``.
    """
    src, dst = edge_index[0].long(), edge_index[1].long()
    return src * num_nodes + dst


def remove_self_loops(edge_index):
    """Drop arcs whose endpoints coincide.

    Parameters
    ----------
    edge_index : torch.Tensor
        Arc indices of shape ``[2, num_edges]``.

    Returns
    -------
    torch.Tensor
        Arc indices of shape ``[2, num_kept]``.
    """
    return edge_index[:, edge_index[0] != edge_index[1]]


def symmetrize_support(edge_index, num_nodes: int):
    """Build the deduplicated, self-loop-free, symmetric arc support.

    Parameters
    ----------
    edge_index : torch.Tensor
        Arc indices of shape ``[2, num_edges]``, possibly directed, possibly
        containing self-loops and duplicates.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Arc indices of shape ``[2, 2 * num_pairs]``, sorted by key, in which
        every arc is accompanied by its reverse and no arc repeats.
    """
    loop_free = remove_self_loops(edge_index)
    both = torch.cat([loop_free, loop_free.flip(0)], dim=1)
    keys = torch.unique(edge_keys(both, num_nodes))
    return torch.stack(
        [
            torch.div(keys, num_nodes, rounding_mode="floor"),
            keys % num_nodes,
        ]
    )


def compute_left_right_map_index(edge_index, num_nodes: int):
    r"""Pair every undirected edge with the two support rows carrying it.

    For each unordered pair :math:`\{a, b\}` with :math:`a < b`, returns the
    row of ``edge_index`` holding the arc :math:`(a, b)` -- whose predicted
    map is :math:`\mathcal{F}^{0}_{a \lhd e}` -- and the row holding
    :math:`(b, a)`, whose predicted map is
    :math:`\mathcal{F}^{0}_{b \lhd e}`. The off-diagonal block of Eq. 2 is then
    :math:`-\mathcal{F}^{0\top}_{a \lhd e} \mathcal{F}^{0}_{b \lhd e}` scaled
    by the pair's phase.

    Parameters
    ----------
    edge_index : torch.Tensor
        A symmetric, deduplicated, self-loop-free support of shape
        ``[2, 2 * num_pairs]``, as produced by :func:`symmetrize_support`.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    left_right_index : torch.Tensor
        Shape ``[2, num_pairs]``; row 0 indexes the arcs ``(a, b)`` with
        ``a < b`` and row 1 indexes their reverses.
    pair_index : torch.Tensor
        Shape ``[2, num_pairs]`` listing the pairs ``(a, b)`` with ``a < b``.

    Raises
    ------
    ValueError
        If any arc of ``edge_index`` has no reverse in ``edge_index``.
    """
    src, dst = edge_index[0], edge_index[1]
    lower = torch.nonzero(src < dst, as_tuple=False).flatten()
    if lower.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return empty, empty

    keys = edge_keys(edge_index, num_nodes)
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    reverse_keys = dst[lower].long() * num_nodes + src[lower].long()
    slot = torch.searchsorted(sorted_keys, reverse_keys)
    slot = slot.clamp(max=sorted_keys.numel() - 1)
    right_index = order[slot]

    if not bool(torch.all(keys[right_index] == reverse_keys)):
        raise ValueError(
            "Every arc must have its reverse in the support; pass the output "
            "of symmetrize_support()."
        )

    left_right_index = torch.stack([lower, right_index])
    pair_index = torch.stack([src[lower], dst[lower]])
    return left_right_index, pair_index


def has_directed_edge(edge_index, num_nodes: int, query):
    r"""Test membership of arcs in the binary adjacency of ``edge_index``.

    Definition 1 builds the phase from the **binary** adjacency, so this
    reports presence rather than multiplicity: a duplicated arc must not
    contribute twice, or a tripled arc would yield
    :math:`e^{i 6 \pi q}` instead of :math:`e^{i 2 \pi q}`.

    Parameters
    ----------
    edge_index : torch.Tensor
        Arc indices of shape ``[2, num_edges]`` defining the adjacency.
    num_nodes : int
        Number of nodes in the graph.
    query : torch.Tensor
        Arcs to test, of shape ``[2, num_queries]``.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[num_queries]``, true where the arc is
        present in ``edge_index``.
    """
    num_queries = query.size(1)
    if edge_index.numel() == 0 or num_queries == 0:
        return torch.zeros(num_queries, dtype=torch.bool, device=query.device)

    keys = torch.sort(edge_keys(edge_index, num_nodes)).values
    query_keys = edge_keys(query, num_nodes)
    slot = torch.searchsorted(keys, query_keys).clamp(max=keys.numel() - 1)
    return keys[slot] == query_keys


def _block_templates(d: int, device, diagonal: bool):
    """Build the row/column element offsets of one block.

    Parameters
    ----------
    d : int
        Stalk dimension, i.e. block side length.
    device : torch.device
        Device on which to allocate.
    diagonal : bool
        If true, describe only the ``d`` diagonal entries of the block;
        otherwise all ``d * d`` entries in row-major order.

    Returns
    -------
    row_template : torch.Tensor
        Row offsets, flattened.
    col_template : torch.Tensor
        Column offsets, flattened.
    """
    within = torch.arange(0, d, device=device)
    if diagonal:
        return within, within.clone()
    row_template = within.view(-1, 1).tile(1, d).reshape(-1)
    col_template = within.view(1, -1).tile(d, 1).reshape(-1)
    return row_template, col_template


def block_indices(size: int, pair_index, d: int, diagonal: bool = False):
    """Build sparse element indices for the diagonal and off-diagonal blocks.

    Parameters
    ----------
    size : int
        Number of nodes, so the operator is ``size * d`` square.
    pair_index : torch.Tensor
        Node pairs ``(a, b)`` with ``a < b``, of shape ``[2, num_pairs]``.
    d : int
        Stalk dimension.
    diagonal : bool, optional
        If true, each block carries only its ``d`` diagonal entries. Default
        is False.

    Returns
    -------
    diag_indices : torch.Tensor
        Element indices of the block-diagonal entries, shape ``[2, n_diag]``.
    non_diag_indices : torch.Tensor
        Element indices of the blocks at ``(a, b)``, shape ``[2, n_off]``.

    Raises
    ------
    ValueError
        If ``pair_index`` is non-empty and not strictly increasing per column.
    """
    device = pair_index.device
    row_template, col_template = _block_templates(d, device, diagonal)

    if pair_index.numel() and not bool(
        torch.all(pair_index[0] < pair_index[1])
    ):
        raise ValueError(
            "pair_index must satisfy pair_index[0] < pair_index[1]"
        )

    rows, cols = pair_index[0].long(), pair_index[1].long()
    off_rows = (row_template.view(1, -1) + d * rows.view(-1, 1)).reshape(1, -1)
    off_cols = (col_template.view(1, -1) + d * cols.view(-1, 1)).reshape(1, -1)
    non_diag_indices = torch.cat([off_rows, off_cols], dim=0)

    nodes = torch.arange(0, size, device=device)
    diag_rows = (row_template.view(1, -1) + d * nodes.view(-1, 1)).reshape(
        1, -1
    )
    diag_cols = (col_template.view(1, -1) + d * nodes.view(-1, 1)).reshape(
        1, -1
    )
    diag_indices = torch.cat([diag_rows, diag_cols], dim=0)

    return diag_indices, non_diag_indices


def flip_index(index):
    """Swap row and column element indices, transposing every block.

    Because :func:`block_indices` lays each block out in row-major order, an
    element-level swap moves the entry at ``(i + d*a, j + d*b)`` to
    ``(j + d*b, i + d*a)``, i.e. it places the transpose of the block at
    ``(a, b)`` into position ``(b, a)``. Reusing the values therefore yields a
    symmetric matrix, and conjugating them yields a Hermitian one.

    Parameters
    ----------
    index : torch.Tensor
        Sparse element indices of shape ``[2, nnz]``.

    Returns
    -------
    torch.Tensor
        Indices of shape ``[2, nnz]`` with rows and columns exchanged.
    """
    return torch.stack([index[1], index[0]])


def node_degree(edge_index, num_nodes: int, dtype=torch.float32):
    """Count arcs leaving each node.

    Parameters
    ----------
    edge_index : torch.Tensor
        Arc indices of shape ``[2, num_edges]``.
    num_nodes : int
        Number of nodes in the graph.
    dtype : torch.dtype, optional
        Floating dtype of the result. Default is ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Degrees of shape ``[num_nodes]``.
    """
    counts = torch.bincount(edge_index[0].long(), minlength=num_nodes)
    return counts.to(dtype)


def mergesp(index1, value1, index2, value2):
    """Concatenate two sparse patterns assumed to have disjoint indices.

    Parameters
    ----------
    index1 : torch.Tensor
        First index set of shape ``[2, nnz1]``.
    value1 : torch.Tensor
        First value set of shape ``[nnz1]``.
    index2 : torch.Tensor
        Second index set of shape ``[2, nnz2]``.
    value2 : torch.Tensor
        Second value set of shape ``[nnz2]``.

    Returns
    -------
    index : torch.Tensor
        Merged indices of shape ``[2, nnz1 + nnz2]``.
    value : torch.Tensor
        Merged values of shape ``[nnz1 + nnz2]``.
    """
    return (
        torch.cat([index1, index2], dim=1),
        torch.cat([value1, value2]),
    )


def spmm(index, value, num_rows: int, x):
    """Multiply a sparse matrix by a dense matrix.

    Implemented as gather-multiply-scatter with :meth:`torch.Tensor.index_add`,
    which keeps the new code free of the compiled ``torch_sparse`` and
    ``torch_scatter`` extensions and is differentiable with respect to both
    ``value`` and ``x``. Repeated indices accumulate, so a pattern need not be
    coalesced.

    Parameters
    ----------
    index : torch.Tensor
        Sparse indices of shape ``[2, nnz]`` as ``(row, col)``.
    value : torch.Tensor
        Sparse values of shape ``[nnz]``.
    num_rows : int
        Number of rows of the sparse matrix, i.e. of the output.
    x : torch.Tensor
        Dense operand of shape ``[num_cols, num_features]``.

    Returns
    -------
    torch.Tensor
        Product of shape ``[num_rows, num_features]``.
    """
    row, col = index[0], index[1]
    out = torch.zeros(num_rows, x.size(-1), dtype=x.dtype, device=x.device)
    return out.index_add(0, row, x.index_select(0, col) * value.unsqueeze(-1))
