"""Unit tests for the RNeighbourhood (loopy) transform."""

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms import TRANSFORMS
from topobench.transforms.data_manipulations.r_neighbourhood import (
    RNeighbourhood,
    _bounded_simple_cycles,
    r_neighborhood,
)


def _undirected(edges, num_nodes, feat=4):
    """Build a Data object with both edge orientations."""
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(
        x_0=torch.randn(num_nodes, feat),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


TRIANGLE = ([[0, 1], [1, 2], [2, 0]], 3)
SQUARE = ([[0, 1], [1, 2], [2, 3], [3, 0]], 4)
TRIANGLE_TAIL = ([[0, 1], [1, 2], [2, 0], [2, 3]], 4)
PATH = ([[0, 1], [1, 2], [2, 3]], 4)


class TestBoundedSimpleCycles:
    """Test the environment-safe cycle enumeration."""

    def _graph(self, edges):
        g = nx.Graph()
        g.add_edges_from(edges)
        return g

    def test_triangle_found(self):
        cycles = _bounded_simple_cycles(self._graph(TRIANGLE[0]), 4)
        assert len(cycles) == 1
        assert sorted(cycles[0]) == [0, 1, 2]

    def test_square_found(self):
        cycles = _bounded_simple_cycles(self._graph(SQUARE[0]), 4)
        assert len(cycles) == 1
        assert len(cycles[0]) == 4

    def test_length_bound_excludes_longer(self):
        # A 4-cycle is not returned when the bound only allows length 3.
        cycles = _bounded_simple_cycles(self._graph(SQUARE[0]), 3)
        assert cycles == []

    def test_path_has_no_cycles(self):
        assert _bounded_simple_cycles(self._graph(PATH[0]), 4) == []

    def test_each_cycle_returned_once(self):
        # Two triangles sharing nothing -> exactly two cycles, deduped.
        edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]
        cycles = _bounded_simple_cycles(self._graph(edges), 4)
        assert len(cycles) == 2

    def test_min_node_first(self):
        cycles = _bounded_simple_cycles(self._graph(TRIANGLE[0]), 4)
        assert cycles[0][0] == min(cycles[0])


class TestRNeighborhoodFunction:
    """Test the ``r_neighborhood`` helper."""

    def _graph(self, edges):
        g = nx.Graph()
        g.add_edges_from(edges)
        return g

    def test_returns_three_dicts(self):
        paths, hops, edge_attr_idx = r_neighborhood(
            self._graph(TRIANGLE[0]), r=2
        )
        assert isinstance(paths, dict)
        assert isinstance(hops, dict)
        assert isinstance(edge_attr_idx, dict)

    def test_paths_keyed_by_order(self):
        paths, _, _ = r_neighborhood(self._graph(SQUARE[0]), r=2)
        assert set(paths.keys()) == {0, 1, 2}

    def test_triangle_appears_at_order_one(self):
        paths, _, _ = r_neighborhood(self._graph(TRIANGLE[0]), r=2)
        assert paths[1].shape == (1, 3)  # one length-3 cycle
        assert paths[2].shape[0] == 0  # no length-4 cycle

    def test_square_appears_at_order_two(self):
        paths, _, _ = r_neighborhood(self._graph(SQUARE[0]), r=2)
        assert paths[2].shape == (1, 4)
        assert paths[1].shape[0] == 0

    def test_hops_are_symmetric_distances(self):
        _, hops, _ = r_neighborhood(self._graph(TRIANGLE[0]), r=2)
        assert hops[0, 0] == 0
        assert hops[0, 1] == 1
        assert hops[1, 2] == 1

    def test_directed_graph_is_undirected(self):
        digraph = nx.DiGraph()
        digraph.add_edges_from(TRIANGLE[0])
        paths, _, _ = r_neighborhood(digraph, r=2)
        assert paths[1].shape == (1, 3)


class TestRNeighbourhoodTransform:
    """Test the ``RNeighbourhood`` transform."""

    def test_init_stores_params(self):
        t = RNeighbourhood(r=3, transform_name="RNeighbourhood", foo="bar")
        assert t.r == 3
        assert t.parameters["transform_name"] == "RNeighbourhood"
        assert t.parameters["foo"] == "bar"

    def test_registered_in_transforms(self):
        assert "RNeighbourhood" in TRANSFORMS

    def test_adds_all_orders(self):
        data = _undirected(*TRIANGLE_TAIL)
        out = RNeighbourhood(r=2)(data)
        for order in range(3):
            assert f"loopyN{order}" in out
            assert f"loopyA{order}" in out
            assert f"loopyNcount{order}" in out

    def test_order_zero_is_directed_edges(self):
        data = _undirected(*TRIANGLE)
        out = RNeighbourhood(r=2)(data)
        # 3 undirected edges -> 6 directed (centre, neighbour) rows.
        assert out["loopyN0"].shape == (6, 2)
        # Every hop row of a direct neighbour is [0, 1].
        assert torch.equal(
            out["loopyA0"], torch.tensor([[0, 1]]).repeat(6, 1)
        )

    def test_shapes_and_alignment(self):
        data = _undirected(*SQUARE)
        out = RNeighbourhood(r=2)(data)
        for order in range(3):
            n, a = out[f"loopyN{order}"], out[f"loopyA{order}"]
            assert n.shape == a.shape
            assert n.shape[1] == order + 2
            assert out[f"loopyNcount{order}"].item() == n.shape[0]

    def test_square_produces_four_rotations(self):
        data = _undirected(*SQUARE)
        out = RNeighbourhood(r=2)(data)
        assert out["loopyN2"].shape == (4, 4)  # one 4-cycle, four centres
        assert out["loopyN1"].shape[0] == 0  # no triangle

    def test_centre_hop_is_zero(self):
        data = _undirected(*TRIANGLE)
        out = RNeighbourhood(r=2)(data)
        # Column 0 is the centre, whose hop distance to itself is 0.
        assert torch.all(out["loopyA1"][:, 0] == 0)

    def test_node_indices_in_range(self):
        data = _undirected(*TRIANGLE_TAIL)
        out = RNeighbourhood(r=2)(data)
        for order in range(3):
            paths = out[f"loopyN{order}"]
            if paths.numel():
                assert paths.min() >= 0
                assert paths.max() < data.num_nodes

    def test_dtype_is_long(self):
        out = RNeighbourhood(r=2)(_undirected(*TRIANGLE))
        assert out["loopyN0"].dtype == torch.long
        assert out["loopyA0"].dtype == torch.long
        assert out["loopyNcount0"].dtype == torch.long

    @pytest.mark.parametrize("r", [0, 1, 2, 3])
    def test_orders_match_r(self, r):
        out = RNeighbourhood(r=r)(_undirected(*SQUARE))
        assert all(f"loopyN{order}" in out for order in range(r + 1))
        assert f"loopyN{r + 1}" not in out

    def test_order_zero_only_when_r_zero(self):
        out = RNeighbourhood(r=0)(_undirected(*TRIANGLE))
        assert out["loopyN0"].shape[0] == 6
        assert "loopyN1" not in out

    def test_empty_graph(self):
        data = Data(
            x_0=torch.randn(3, 4),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=3,
        )
        out = RNeighbourhood(r=2)(data)
        for order in range(3):
            assert out[f"loopyN{order}"].shape == (0, order + 2)
            assert out[f"loopyNcount{order}"].item() == 0

    def test_no_dict_fields(self):
        out = RNeighbourhood(r=2)(_undirected(*SQUARE))
        assert not any(
            isinstance(out[key], dict) for key in out.keys()
        )

    def test_dispatch_via_data_transform(self):
        from topobench.transforms.data_transform import DataTransform

        t = DataTransform(
            transform_name="RNeighbourhood",
            transform_type="data manipulation",
            r=2,
        )
        out = t(_undirected(*TRIANGLE))
        assert "loopyN1" in out
