"""R-neighbourhood structural transform for the loopy models."""

import networkx as nx
import numpy as np
import torch
import torch_geometric


def _bounded_simple_cycles(graph: nx.Graph, max_length: int) -> list:
    """Enumerate simple cycles of length 3 to ``max_length`` inclusive.

    Parameters
    ----------
    graph : networkx.Graph
        The undirected input graph.
    max_length : int
        Maximum cycle length to enumerate.

    Returns
    -------
    list
        One representative node list per undirected simple cycle.
    """
    adjacency = {node: sorted(graph.neighbors(node)) for node in graph.nodes}
    found = {}
    for start in sorted(graph.nodes):
        # Forcing ``start`` to be the smallest node of every cycle it closes
        # means each cycle is discovered from a single starting node.
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour == start and len(path) >= 3:
                    reverse = (path[0], *path[:0:-1])
                    found[min(path, reverse)] = list(min(path, reverse))
                elif (
                    neighbour > start
                    and neighbour not in path
                    and len(path) < max_length
                ):
                    stack.append((neighbour, (*path, neighbour)))
    return list(found.values())


def r_neighborhood(
    G: nx.Graph,
    r: int,
) -> tuple[dict, dict]:
    """Compute the r-neighbourhood cycles and pairwise node distances.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    r : int
        Maximal neighbourhood order.

    Returns
    -------
    paths : dict
        Mapping from order ``L`` to the array of length ``L + 2`` simple
        cycles, of shape ``(num_cycles, L + 2)``.
    hops : dict
        Mapping from a pair of nodes ``(s, t)`` to their shortest path
        distance in the graph.
    edge_attr_idx : dict
        Unused placeholder kept for compatibility, one empty array per order.
    """
    cycles = _bounded_simple_cycles(
        G.to_undirected() if G.is_directed() else G, max_length=r + 2
    )
    distances = dict(nx.all_pairs_shortest_path_length(G))
    paths = {}
    hops = {
        (source, target): d
        for source, target_dict in distances.items()
        for target, d in target_dict.items()
    }
    edge_attr_idx = {}
    for L in range(2, r + 3):
        # Initialization
        paths[L - 2] = np.zeros((0, L), dtype=int)
        edge_attr_idx[L - 2] = np.zeros((L - 1, 0))
        # Dividing the simple cycles depending on the length
        L_long_cycles = [cycle for cycle in cycles if len(cycle) == L]
        if L_long_cycles:
            # Adding center of neighborhood as last element
            paths[L - 2] = np.array(sorted(L_long_cycles))

    return paths, hops, edge_attr_idx


class RNeighbourhood(torch_geometric.transforms.BaseTransform):
    r"""Precompute the loopy r-neighbourhood path tensors on a graph.

    For every order ``L`` in ``0 .. r`` it adds ``loopyN{L}`` (the paths),
    ``loopyA{L}`` (the hop distances) and ``loopyNcount{L}`` (the per-graph
    path count) to each data object.

    Parameters
    ----------
    r : int, optional
        Maximal neighbourhood order; the longest path has ``r + 2`` nodes.
    **kwargs : dict, optional
        Extra arguments forwarded by the TopoBench transform dispatch (e.g.
        ``transform_name``); stored but unused.
    """

    def __init__(self, r: int = 2, **kwargs):
        super().__init__()
        self.r = r
        self.parameters = kwargs

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Add the loopy path tensors to the data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph. Only ``edge_index`` is read; node indices in the
            output refer to the rows of ``data.x_0`` / ``data.x``.

        Returns
        -------
        torch_geometric.data.Data
            The same object with ``loopyN{L}``, ``loopyA{L}`` and
            ``loopyNcount{L}`` fields added for every ``L`` in ``0 .. r``.
        """
        graph = nx.Graph()
        graph.add_edges_from(data.edge_index.t().cpu().tolist())
        paths, hops, _ = r_neighborhood(graph, r=self.r)

        for order in range(self.r + 1):
            length = order + 2
            # Map each directed path (node tuple) to its hop row.
            rows = {}
            if length == 2:
                # Direct neighbours: both orientations of every edge.
                for centre in graph.nodes:
                    for neighbour in graph.neighbors(centre):
                        rows[(centre, neighbour)] = [0, 1]
            else:
                # One rotation per node as centre;
                for cycle in paths[order]:
                    cycle = [int(node) for node in cycle]
                    for shift in range(length):
                        rolled = cycle[shift:] + cycle[:shift]
                        path = tuple(rolled)
                        centre = path[0]
                        rows[path] = [hops[centre, node] for node in path]

            ordered = sorted(rows)
            if ordered:
                node_rows = np.asarray(ordered, dtype=np.int64)
                hop_rows = np.asarray(
                    [rows[path] for path in ordered], dtype=np.int64
                )
            else:
                node_rows = np.zeros((0, length), dtype=np.int64)
                hop_rows = np.zeros((0, length), dtype=np.int64)

            data[f"loopyN{order}"] = torch.from_numpy(node_rows)
            data[f"loopyA{order}"] = torch.from_numpy(hop_rows)
            data[f"loopyNcount{order}"] = torch.tensor(
                [node_rows.shape[0]], dtype=torch.long
            )
        return data
