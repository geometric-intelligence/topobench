"""Batched structure builders for the SMCN backbone.

SMCN processes, besides the usual cell-complex neighborhoods, a *bag of
marked subcomplexes* over the rank pair (0, 1): one copy of the node set
per edge, marked with hop distances to that edge. The reference
implementation precomputes these structures per sample; here they are
built on the fly from the batched incidence matrices, which keeps the
data pipeline unchanged and works for any batch composition.

The construction follows the official SMCN implementation
(https://github.com/yoavgelberg/SMCN, ``data/utils.py``): bag rows are
laid out as ``row(u, e) = u * n_edges + e`` within each graph, the bag
low-adjacency replicates the node adjacency once per edge copy, and the
marking of row ``(u, e)`` is the shortest-path distance from node ``u``
to the closest endpoint of edge ``e``, bucketed as in the reference
(distances above ``max_dist`` map to ``max_dist``, disconnected pairs to
``max_dist + 1``).

References
----------
Eitan et al. "Topological Blindspots: Understanding and Extending
Topological Deep Learning Through the Lens of Expressivity." ICLR 2025.
https://arxiv.org/abs/2408.05486
"""

from dataclasses import dataclass

import torch


@dataclass
class SMCNStructures:
    """Connectivity and subcomplex-bag structures for one batch.

    All index tensors are global with respect to the batch: cell indices
    are offset per graph, and bag-row indices are offset by the number of
    bag rows of the preceding graphs.

    Attributes
    ----------
    a01_pairs : torch.Tensor
        Ordered node pairs adjacent through an edge, shape ``[2, P0]``.
    a01_bridge : torch.Tensor
        Mediating edge index of every pair in ``a01_pairs``, shape ``[P0]``.
    a12_pairs : torch.Tensor
        Ordered edge pairs adjacent through a 2-cell, shape ``[2, P1]``.
    a12_bridge : torch.Tensor
        Mediating 2-cell index of every pair in ``a12_pairs``, shape ``[P1]``.
    inc01_pairs : torch.Tensor
        Node-edge incidence pairs, shape ``[2, I0]``.
    inc12_pairs : torch.Tensor
        Edge-2-cell incidence pairs, shape ``[2, I1]``.
    inc02_pairs : torch.Tensor
        Node-2-cell incidence pairs, shape ``[2, I2]``.
    bag_low_index : torch.Tensor
        Node index of every bag row, shape ``[R]``.
    bag_high_index : torch.Tensor
        Edge index of every bag row, shape ``[R]``.
    bag_marking : torch.Tensor
        Bucketed hop-distance marking of every bag row, shape ``[R]``.
    bag_low_adj_pairs : torch.Tensor
        Bag-row pairs of the replicated node adjacency, shape ``[2, Q]``.
    bag_low_adj_bridge : torch.Tensor
        Mediating edge index of every bag adjacency pair, shape ``[Q]``.
    bag_inc_pairs : torch.Tensor
        Bag-row pairs broadcasting each marked row to the rows of the
        same node, shape ``[2, S]``.
    bag_batch : torch.Tensor
        Graph index of every bag row, shape ``[R]``.
    """

    a01_pairs: torch.Tensor
    a01_bridge: torch.Tensor
    a12_pairs: torch.Tensor
    a12_bridge: torch.Tensor
    inc01_pairs: torch.Tensor
    inc12_pairs: torch.Tensor
    inc02_pairs: torch.Tensor
    bag_low_index: torch.Tensor
    bag_high_index: torch.Tensor
    bag_marking: torch.Tensor
    bag_low_adj_pairs: torch.Tensor
    bag_low_adj_bridge: torch.Tensor
    bag_inc_pairs: torch.Tensor
    bag_batch: torch.Tensor


def sparse_pairs(matrix):
    """Return the (row, col) index pairs of a sparse or dense matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse (COO/CSR) or dense matrix.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[2, nnz]`` with row/column indices.
    """
    if matrix.is_sparse or matrix.layout == torch.sparse_csr:
        coo = matrix.to_sparse_coo().coalesce()
        return coo.indices().long()
    return torch.nonzero(matrix, as_tuple=False).t().long()


def adjacency_from_incidence(incidence, expected_size=None):
    """Build ordered cell pairs mediated by a higher-rank cell.

    Two rank-``r`` cells are adjacent whenever they are both contained in
    a common rank-``(r + 1)`` cell; that common cell is the *bridge* and
    its features are used as edge features by the message passing.

    Parameters
    ----------
    incidence : torch.Tensor
        Incidence matrix of shape ``[n_low, n_high]`` (sparse or dense).
    expected_size : int, optional
        If given, every column must have exactly this many nonzero
        entries (e.g. 2 for a node-edge incidence). Default is None.

    Returns
    -------
    pairs : torch.Tensor
        Ordered pairs of low-rank cells, shape ``[2, P]``.
    bridge : torch.Tensor
        Mediating high-rank cell of each pair, shape ``[P]``.

    Raises
    ------
    ValueError
        If ``expected_size`` is given and some column violates it.
    """
    idx = sparse_pairs(incidence)
    if idx.numel() == 0:
        empty = idx.new_zeros(0)
        return idx.new_zeros(2, 0), empty
    rows, cols = idx[0], idx[1]
    order = torch.argsort(cols, stable=True)
    rows, cols = rows[order], cols[order]
    counts = torch.bincount(cols, minlength=int(cols.max()) + 1)
    counts = counts[counts > 0]
    if expected_size is not None and not bool((counts == expected_size).all()):
        raise ValueError(
            "every column of the incidence must have exactly "
            f"{expected_size} nonzero entries"
        )
    pairs_src, pairs_dst, bridges = [], [], []
    unique_cols = torch.unique_consecutive(cols)
    start = 0
    sizes = counts.tolist()
    for col, size in zip(unique_cols.tolist(), sizes, strict=False):
        members = rows[start : start + size]
        start += size
        if size < 2:
            continue
        grid_a = members.repeat_interleave(size)
        grid_b = members.repeat(size)
        keep = grid_a != grid_b
        pairs_src.append(grid_a[keep])
        pairs_dst.append(grid_b[keep])
        bridges.append(torch.full_like(grid_a[keep], fill_value=col))
    if not pairs_src:
        empty = idx.new_zeros(0)
        return idx.new_zeros(2, 0), empty
    pairs = torch.stack([torch.cat(pairs_src), torch.cat(pairs_dst)], dim=0)
    return pairs, torch.cat(bridges)


def hop_distance_buckets(adjacency, max_dist):
    """Compute bucketed all-pairs hop distances of one graph.

    Follows the reference marking semantics: exact distances up to
    ``max_dist``; nodes that are connected but farther than ``max_dist``
    receive ``max_dist``; disconnected pairs receive ``max_dist + 1``.

    Parameters
    ----------
    adjacency : torch.Tensor
        Dense boolean adjacency of shape ``[n, n]``.
    max_dist : int
        Largest exact distance to resolve.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[n, n]`` with values in
        ``[0, max_dist + 1]``.
    """
    n = adjacency.size(0)
    device = adjacency.device
    dist = torch.full((n, n), max_dist + 1, dtype=torch.long, device=device)
    dist.fill_diagonal_(0)
    reach = torch.eye(n, dtype=torch.bool, device=device)
    adj = adjacency.bool() | reach
    frontier = reach
    for d in range(1, max_dist + 1):
        frontier = (frontier.float() @ adj.float()) > 0
        newly = frontier & (dist == max_dist + 1)
        dist[newly] = d
    # Distinguish "connected but farther" from "disconnected" by closing
    # the reachability transitively (log-doubling).
    closure = frontier
    steps = max(1, (n - 1).bit_length())
    for _ in range(steps):
        closure = (closure.float() @ closure.float()) > 0
    far = closure & (dist == max_dist + 1)
    dist[far] = max_dist
    return dist


def build_smcn_structures(
    incidence_1,
    incidence_2,
    batch_0,
    batch_1,
    batch_2,
    max_dist=10,
):
    """Build all SMCN connectivity/bag structures for a batch.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Block-diagonal node-edge incidence ``[n_nodes, n_edges]``.
    incidence_2 : torch.Tensor
        Block-diagonal edge-2-cell incidence ``[n_edges, n_cells]``.
    batch_0 : torch.Tensor
        Graph index of every node, shape ``[n_nodes]``.
    batch_1 : torch.Tensor
        Graph index of every edge, shape ``[n_edges]``.
    batch_2 : torch.Tensor
        Graph index of every 2-cell, shape ``[n_cells]``.
    max_dist : int, optional
        Marking distance cutoff. Default is 10.

    Returns
    -------
    SMCNStructures
        The assembled batch structures.
    """
    device = batch_0.device
    num_graphs = int(batch_0.max()) + 1 if batch_0.numel() else 0

    inc01 = sparse_pairs(incidence_1)
    inc12 = (
        sparse_pairs(incidence_2)
        if incidence_2 is not None and incidence_2.numel()
        else batch_0.new_zeros(2, 0)
    )
    a01_pairs, a01_bridge = adjacency_from_incidence(
        incidence_1, expected_size=2
    )
    a12_pairs, a12_bridge = adjacency_from_incidence(incidence_2)

    # Node-to-2-cell incidence via the edge memberships.
    if inc12.numel():
        dense_1 = incidence_1
        if dense_1.is_sparse or dense_1.layout == torch.sparse_csr:
            dense_1 = dense_1.to_dense()
        dense_2 = incidence_2
        if dense_2.is_sparse or dense_2.layout == torch.sparse_csr:
            dense_2 = dense_2.to_dense()
        membership = (dense_1.abs() @ dense_2.abs()) > 0
        inc02 = torch.nonzero(membership, as_tuple=False).t().long()
    else:
        inc02 = batch_0.new_zeros(2, 0)

    n0 = torch.bincount(batch_0, minlength=num_graphs)
    n1 = torch.bincount(batch_1, minlength=num_graphs)
    node_off = torch.cumsum(n0, 0) - n0
    edge_off = torch.cumsum(n1, 0) - n1
    rows_per_graph = n0 * n1
    row_off = torch.cumsum(rows_per_graph, 0) - rows_per_graph

    bag_low, bag_high, bag_mark, bag_batch = [], [], [], []
    bl_pairs, bl_bridge, bi_pairs = [], [], []

    pair_graph = batch_1[a01_bridge] if a01_bridge.numel() else a01_bridge
    inc_graph = batch_1[inc01[1]] if inc01.numel() else inc01.new_zeros(0)

    for g in range(num_graphs):
        nl, nh = int(n0[g]), int(n1[g])
        if nl == 0 or nh == 0:
            continue
        n_offset, e_offset, r_offset = (
            int(node_off[g]),
            int(edge_off[g]),
            int(row_off[g]),
        )
        rows = torch.arange(nl * nh, device=device)
        bag_low.append(rows // nh + n_offset)
        bag_high.append(rows % nh + e_offset)
        bag_batch.append(
            torch.full((nl * nh,), g, dtype=torch.long, device=device)
        )

        # Local structures of this graph.
        pair_mask = pair_graph == g
        loc_pairs = a01_pairs[:, pair_mask] - n_offset
        loc_bridge = a01_bridge[pair_mask] - e_offset
        inc_mask = inc_graph == g
        loc_inc = inc01[:, inc_mask].clone()
        loc_inc[0] -= n_offset
        loc_inc[1] -= e_offset

        # Distance marking: min over the endpoints of each edge copy.
        adj = torch.zeros(nl, nl, dtype=torch.bool, device=device)
        adj[loc_pairs[0], loc_pairs[1]] = True
        dist = hop_distance_buckets(adj, max_dist)
        endpoints = torch.full((nh, 2), -1, dtype=torch.long, device=device)
        edge_sorted = torch.argsort(loc_inc[1], stable=True)
        sorted_nodes = loc_inc[0][edge_sorted]
        endpoints[:, 0] = sorted_nodes[0::2]
        endpoints[:, 1] = sorted_nodes[1::2]
        marking = dist[:, endpoints].min(dim=-1).values  # [nl, nh]
        bag_mark.append(marking.reshape(-1))

        # Bag low-adjacency: node adjacency replicated once per edge copy.
        copies = torch.arange(nh, device=device)
        src = (loc_pairs[0].unsqueeze(1) * nh + copies).reshape(-1)
        dst = (loc_pairs[1].unsqueeze(1) * nh + copies).reshape(-1)
        bl_pairs.append(torch.stack([src, dst], dim=0) + r_offset)
        bl_bridge.append((loc_bridge + e_offset).repeat_interleave(nh))

        # Bag incidence: marked row (u, e) broadcasts to all rows of u.
        marked = loc_inc[0] * nh + loc_inc[1]
        targets = (loc_inc[0].unsqueeze(1) * nh + copies).reshape(-1)
        sources = marked.repeat_interleave(nh)
        bi_pairs.append(torch.stack([sources, targets], dim=0) + r_offset)

    def _cat(parts, like, width=None):
        """Concatenate chunks, or make an empty tensor when none exist.

        Parameters
        ----------
        parts : list of torch.Tensor
            Per-graph chunks to concatenate along the last dimension.
        like : torch.Tensor
            Tensor providing the dtype and device for the empty case.
        width : int, optional
            First dimension of the empty tensor, used for pair tensors.

        Returns
        -------
        torch.Tensor
            Concatenated tensor, or an empty one if ``parts`` is empty.
        """
        if parts:
            return torch.cat(parts, dim=-1)
        if width is not None:
            return like.new_zeros(width, 0)
        return like.new_zeros(0)

    return SMCNStructures(
        a01_pairs=a01_pairs,
        a01_bridge=a01_bridge,
        a12_pairs=a12_pairs,
        a12_bridge=a12_bridge,
        inc01_pairs=inc01,
        inc12_pairs=inc12,
        inc02_pairs=inc02,
        bag_low_index=_cat(bag_low, batch_0),
        bag_high_index=_cat(bag_high, batch_0),
        bag_marking=_cat(bag_mark, batch_0),
        bag_low_adj_pairs=_cat(bl_pairs, batch_0, width=2),
        bag_low_adj_bridge=_cat(bl_bridge, batch_0),
        bag_inc_pairs=_cat(bi_pairs, batch_0, width=2),
        bag_batch=_cat(bag_batch, batch_0),
    )
