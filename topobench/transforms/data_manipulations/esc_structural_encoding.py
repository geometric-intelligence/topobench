"""ESC structural cache used by challenge model."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from topobench.data.utils.esc import (
    ESC_DEGREE_BINS,
    ESC_DISTANCE_BASE,
    ESC_DISTANCE_OFFSET,
    ESC_EDGE_OFFSET,
    ESC_HOP_RADIUS,
    validate_esc_tensors,
)


def _root_distances(
    nodes: Sequence[int], root: int, neighbors: Sequence[set[int]]
) -> dict[int, int]:
    """Get root distances inside one-hop union.

    Parameters
    ----------
    nodes : sequence of int
        Nodes in rooted union.
    root : int
        Root for distance labels.
    neighbors : sequence of set of int
        Adjacency sets by node ID.

    Returns
    -------
    dict[int, int]
        Distance label for each node.
    """
    return {
        node: 0 if node == root else 1 if node in neighbors[root] else 2
        for node in nodes
    }


def _edge_code(
    first_label: tuple[int, int], second_label: tuple[int, int]
) -> int:
    """Pack endpoint labels into one code ID.

    Parameters
    ----------
    first_label : tuple[int, int]
        First endpoint's two root distances.
    second_label : tuple[int, int]
        Second endpoint's two root distances.

    Returns
    -------
    int
        Code ID for sorted endpoint-label pair.
    """
    first_label, second_label = sorted((first_label, second_label))
    coordinates = (*first_label, *second_label)
    packed = 0
    for coordinate in coordinates:
        packed = ESC_DISTANCE_BASE * packed + coordinate
    return ESC_EDGE_OFFSET + packed


def _directed_histogram(
    nodes: Sequence[int],
    internal_edges: Sequence[tuple[int, int]],
    degrees: Mapping[int, int],
    first_distances: Mapping[int, int],
    second_distances: Mapping[int, int],
) -> list[tuple[int, int]]:
    """Build sparse histogram for one directed root edge.

    Parameters
    ----------
    nodes : sequence of int
        Nodes in rooted subgraph.
    internal_edges : sequence of tuple[int, int]
        Sorted undirected edges inside subgraph.
    degrees : mapping of int to int
        Local degree by node.
    first_distances : mapping of int to int
        Distance to first root by node.
    second_distances : mapping of int to int
        Distance to second root by node.

    Returns
    -------
    list[tuple[int, int]]
        Positive ``(code_id, count)`` pairs, sorted by ID.
    """
    histogram: Counter[int] = Counter()

    for degree in degrees.values():
        histogram[degree] += 1
    for node in nodes:
        histogram[ESC_DISTANCE_OFFSET + first_distances[node]] += 1
        histogram[
            ESC_DISTANCE_OFFSET + ESC_DISTANCE_BASE + second_distances[node]
        ] += 1
    for first, second in internal_edges:
        histogram[
            _edge_code(
                (first_distances[first], second_distances[first]),
                (first_distances[second], second_distances[second]),
            )
        ] += 1

    return sorted(histogram.items())


class ESCStructuralEncoding(BaseTransform):
    r"""Cache ESC histograms for each directed edge.

    Radius stays at one. Uses paper's induced union, not narrower release
    mask. Full 387-code segm keeps packing and embedding IDs stable, even when
    some codes stay unreachable.

    Parameters
    ----------
    hop_radius : int, optional
        Root radius. Only ``1`` works.
    degree_bins : int, optional
        Degree bins. Only ``300`` works.
    rooted_subgraph : str, optional
        Rooted-subgraph rule. Must be ``"induced_union"``.
    include_resistance_distance : bool, optional
        Resistance-distance flag. Must be ``False``.
    add_self_loops : bool, optional
        Self-loop flag. Must be ``False``.
    internal_edge_orientation : str, optional
        Edge sorting rule. Must be ``"lexicographic"``.
    encoder_version : str, optional
        Cache version. Must be ``"esc-paper-induced-v1"``.
    **kwargs : Any
        Extra transform metadata.

    References
    ----------
    Zuoyu Yan et al., "An Efficient Subgraph GNN with Provable Substructure
    Counting Power," KDD 2024, Section 5.1.

    ESC-GNN preprocessing, ``utils_edge_efficient.py`` at commit
    ``ade0a538e561cf46bd5805b412d96b9f9ba265cf``. Used to check release-mask
    difference. No source copied.
    """

    def __init__(
        self,
        hop_radius: int = ESC_HOP_RADIUS,
        degree_bins: int = ESC_DEGREE_BINS,
        rooted_subgraph: str = "induced_union",
        include_resistance_distance: bool = False,
        add_self_loops: bool = False,
        internal_edge_orientation: str = "lexicographic",
        encoder_version: str = "esc-paper-induced-v1",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        expected = {
            "hop_radius": ESC_HOP_RADIUS,
            "degree_bins": ESC_DEGREE_BINS,
            "rooted_subgraph": "induced_union",
            "include_resistance_distance": False,
            "add_self_loops": False,
            "internal_edge_orientation": "lexicographic",
            "encoder_version": "esc-paper-induced-v1",
        }
        resolved = {
            "hop_radius": hop_radius,
            "degree_bins": degree_bins,
            "rooted_subgraph": rooted_subgraph,
            "include_resistance_distance": include_resistance_distance,
            "add_self_loops": add_self_loops,
            "internal_edge_orientation": internal_edge_orientation,
            "encoder_version": encoder_version,
        }
        for name, expected_value in expected.items():
            if resolved[name] != expected_value:
                raise ValueError(
                    f"ESCStructuralEncoding requires {name}={expected_value!r}; "
                    f"received {resolved[name]!r}"
                )

        self.hop_radius = hop_radius
        self.degree_bins = degree_bins
        self.rooted_subgraph = rooted_subgraph
        self.include_resistance_distance = include_resistance_distance
        self.add_self_loops = add_self_loops
        self.internal_edge_orientation = internal_edge_orientation
        self.encoder_version = encoder_version
        self.type = kwargs.get("transform_type", "data manipulation")
        self.parameters = {**resolved, **kwargs}

    def __repr__(self) -> str:
        """Show params that define cache identity."""
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"parameters={self.parameters!r})"
        )

    def forward(self, data: Data) -> Data:
        """Attach sparse ESC cache to graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Simple graph with reciprocal directed edges.

        Returns
        -------
        torch_geometric.data.Data
            Same graph with three ESC cache tensors.

        Raises
        ------
        ValueError
            When graph or generated cache is invalid.
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError(
                "ESCStructuralEncoding requires data.num_nodes to be defined"
            )
        num_nodes = int(num_nodes)
        if num_nodes < 0:
            raise ValueError("ESCStructuralEncoding requires num_nodes >= 0")

        directed_columns, neighbors = self._validate_graph(
            edge_index, num_nodes
        )
        num_edges = edge_index.size(1)
        entries_by_edge: list[list[tuple[int, int]]] = [
            [] for _ in range(num_edges)
        ]

        for first_root, second_root in sorted(directed_columns):
            nodes = sorted(
                {first_root, second_root}
                | neighbors[first_root]
                | neighbors[second_root]
            )
            node_set = set(nodes)
            internal_edges = [
                (node, neighbor)
                for node in nodes
                for neighbor in sorted(neighbors[node] & node_set)
                if node < neighbor
            ]
            degrees = {node: 0 for node in nodes}
            for first, second in internal_edges:
                degrees[first] += 1
                degrees[second] += 1

            first_distances = _root_distances(nodes, first_root, neighbors)
            second_distances = _root_distances(nodes, second_root, neighbors)
            forward_entries = _directed_histogram(
                nodes,
                internal_edges,
                degrees,
                first_distances,
                second_distances,
            )
            reverse_entries = _directed_histogram(
                nodes,
                internal_edges,
                degrees,
                second_distances,
                first_distances,
            )
            columns = directed_columns[(first_root, second_root)]
            entries_by_edge[columns[(first_root, second_root)]] = (
                forward_entries
            )
            entries_by_edge[columns[(second_root, first_root)]] = (
                reverse_entries
            )

        code_ids: list[int] = []
        code_counts: list[float] = []
        nnz_per_edge: list[int] = []
        for entries in entries_by_edge:
            nnz_per_edge.append(len(entries))
            code_ids.extend(code_id for code_id, _ in entries)
            code_counts.extend(float(count) for _, count in entries)

        device = edge_index.device
        data.esc_code_id = torch.tensor(
            code_ids, dtype=torch.long, device=device
        )
        data.esc_code_count = torch.tensor(
            code_counts, dtype=torch.float32, device=device
        )
        data.esc_nnz_per_edge = torch.tensor(
            nnz_per_edge, dtype=torch.long, device=device
        )
        validate_esc_tensors(
            edge_index,
            data.esc_code_id,
            data.esc_code_count,
            data.esc_nnz_per_edge,
            context="ESCStructuralEncoding output",
        )
        return data

    def _validate_graph(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[
        dict[tuple[int, int], dict[tuple[int, int], int]],
        list[set[int]],
    ]:
        """Check graph and map reciprocal edge columns.

        Parameters
        ----------
        edge_index : torch.Tensor
            Directed edges, shape ``[2, E]``.
        num_nodes : int
            Node count.

        Returns
        -------
        grouped : dict
            Directed columns grouped by sorted root pair.
        neighbors : list[set[int]]
            Checked adjacency sets by node ID.

        Raises
        ------
        ValueError
            When graph is not simple and reciprocal, or degree exceeds bins.
        """
        if not isinstance(edge_index, torch.Tensor):
            raise ValueError(
                "ESCStructuralEncoding requires edge_index to be a torch.Tensor"
            )
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "ESCStructuralEncoding requires edge_index with shape [2, E]"
            )
        if edge_index.dtype != torch.long:
            raise ValueError(
                "ESCStructuralEncoding requires edge_index with dtype "
                "torch.long"
            )

        rows = edge_index[0].detach().cpu().tolist()
        columns = edge_index[1].detach().cpu().tolist()
        directed: dict[tuple[int, int], int] = {}
        neighbors = [set() for _ in range(num_nodes)]
        for column, (source, target) in enumerate(
            zip(rows, columns, strict=True)
        ):
            if (
                source < 0
                or source >= num_nodes
                or target < 0
                or target >= num_nodes
            ):
                raise ValueError(
                    "ESCStructuralEncoding found a node ID outside "
                    f"[0, {num_nodes})"
                )
            if source == target:
                raise ValueError(
                    "ESCStructuralEncoding does not support self-loops"
                )
            edge = (source, target)
            if edge in directed:
                raise ValueError(
                    "ESCStructuralEncoding found a duplicate directed edge "
                    f"record {edge}"
                )
            directed[edge] = column
            neighbors[source].add(target)

        for source, target in directed:
            if (target, source) not in directed:
                raise ValueError(
                    "ESCStructuralEncoding requires exactly one reciprocal "
                    f"record for directed edge {(source, target)}"
                )
        if any(len(adjacency) >= self.degree_bins for adjacency in neighbors):
            raise ValueError(
                "ESCStructuralEncoding encountered a rooted-subgraph degree "
                f"outside [0, {self.degree_bins})"
            )

        grouped: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        for edge, column in directed.items():
            source, target = edge
            root = (min(source, target), max(source, target))
            grouped.setdefault(root, {})[edge] = column
        return grouped, neighbors


__all__ = ["ESCStructuralEncoding"]
