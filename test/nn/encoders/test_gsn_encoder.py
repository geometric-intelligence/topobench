"""Unit tests for the GSNFeatureEncoder and its helper functions."""

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.encoders import GSNFeatureEncoder
from topobench.nn.encoders.gsn_encoder import (
    invert_injective_dict,
    make_contiguous,
    normalized_edge,
    nx_to_pyg,
    pyg_to_nx,
)


def triangle_data(dtype=torch.float32) -> Data:
    """Build a single-triangle graph (nodes 0-1-2, all pairwise edges).

    Parameters
    ----------
    dtype : torch.dtype
        Dtype of the node feature matrix ``x``.

    Returns
    -------
    torch_geometric.data.Data
        A 3-node triangle with a constant node feature matrix.
    """
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
    )
    return Data(x=torch.ones(3, 2, dtype=dtype), edge_index=edge_index)


class TestMakeContiguous:
    """Tests for `make_contiguous`."""

    def test_remaps_to_contiguous_preserving_order(self):
        """Values become 0-based contiguous ids in ascending value order."""
        out = make_contiguous({"a": 5, "b": 2, "c": 5, "d": 9})
        # 2 -> 0, 5 -> 1, 9 -> 2; keys unchanged
        assert out == {"a": 1, "b": 0, "c": 1, "d": 2}

    def test_already_contiguous_is_unchanged(self):
        """An already 0-based contiguous mapping is returned unchanged."""
        ii = {"x": 0, "y": 1, "z": 2}
        assert make_contiguous(ii) == ii

    def test_single_value(self):
        """All-equal values collapse to a single id (0)."""
        assert make_contiguous({0: 7, 1: 7, 2: 7}) == {0: 0, 1: 0, 2: 0}

    def test_keys_preserved_and_input_not_mutated(self):
        """Keys are preserved and the input dict is not mutated in place."""
        ii = {"a": 3, "b": 1}
        out = make_contiguous(ii)
        assert set(out.keys()) == {"a", "b"}
        assert ii == {"a": 3, "b": 1}  # unchanged


class TestInvertInjectiveDict:
    """Tests for `invert_injective_dict`."""

    def test_swaps_keys_and_values(self):
        """Keys and values are exchanged."""
        assert invert_injective_dict({1: "a", 2: "b"}) == {"a": 1, "b": 2}

    def test_double_inversion_is_identity(self):
        """Inverting twice recovers the original mapping."""
        ii = {"u": 10, "v": 20, "w": 30}
        assert invert_injective_dict(invert_injective_dict(ii)) == ii

    def test_empty(self):
        """An empty mapping inverts to an empty mapping."""
        assert invert_injective_dict({}) == {}


class TestNormalizedEdge:
    """Tests for `normalized_edge`."""

    def test_orders_endpoints_ascending(self):
        """The smaller endpoint always comes first."""
        assert normalized_edge(2, 0) == (0, 2)
        assert normalized_edge(0, 2) == (0, 2)

    def test_self_loop(self):
        """Equal endpoints are returned as-is."""
        assert normalized_edge(3, 3) == (3, 3)


class TestPygToNx:
    """Tests for `pyg_to_nx`."""

    def test_undirected_and_self_loops_removed(self):
        """Conversion yields an undirected graph without self-loops."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]], dtype=torch.long)
        data = Data(x=torch.randn(3, 4), edge_index=edge_index)
        G = pyg_to_nx(data)
        assert isinstance(G, nx.Graph)
        assert not G.is_directed()
        assert nx.number_of_selfloops(G) == 0
        assert set(G.nodes) == {0, 1, 2}

    def test_copies_node_edge_and_graph_attributes(self):
        """Node, edge, and graph-level attributes are carried over by name."""
        data = Data(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr=torch.tensor([[7.0], [8.0]]),
            y=torch.tensor([1]),  # graph-level
        )
        G = pyg_to_nx(data)
        assert "x" in G.nodes[0]
        _, _, edge_data = next(iter(G.edges(data=True)))
        assert "edge_attr" in edge_data
        assert "y" in G.graph

    def test_edge_index_and_num_nodes_not_treated_as_attrs(self):
        """`edge_index` / `num_nodes` are not copied as node/edge attributes."""
        data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=3,
        )
        G = pyg_to_nx(data)
        assert "edge_index" not in G.nodes[0]
        assert "num_nodes" not in G.graph


class TestNxToPyg:
    """Tests for `nx_to_pyg`."""

    def test_round_trips_node_and_edge_attributes(self):
        """Node and edge attributes are picked up by name."""
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {n: [float(n)] for n in G.nodes}, "feat")
        nx.set_edge_attributes(G, {e: [1.0] for e in G.edges}, "ea")
        data = nx_to_pyg(G)
        assert data.feat.shape[0] == 3
        assert hasattr(data, "ea")

    def test_graph_attr_converted_to_tensor(self):
        """A tensor-convertible graph attribute becomes a tensor."""
        G = nx.path_graph(2)
        G.graph["num"] = 5
        data = nx_to_pyg(G)
        assert torch.is_tensor(data.num)
        assert int(data.num) == 5

    def test_non_tensor_graph_attr_stored_as_is(self):
        """A non-convertible graph attribute is stored unchanged."""
        G = nx.path_graph(2)
        G.graph["name"] = "abc"
        data = nx_to_pyg(G)
        assert data.name == "abc"


class TestGSNFeatureEncoder:
    """Test suite for the `GSNFeatureEncoder`."""

    def test_orbit_partition_triangle(self):
        """A triangle has one node orbit, one edge orbit, |Aut| = 6."""
        info = GSNFeatureEncoder._get_substructure_orbit_partition(
            nx.complete_graph(3)
        )
        assert set(info.node_orbit_partition.values()) == {0}
        assert set(info.edge_orbit_partition.values()) == {0}
        assert info.number_automorphisms == 6

    def test_orbit_partition_path(self):
        """A P3 path: endpoints share an orbit, the center is distinct."""
        info = GSNFeatureEncoder._get_substructure_orbit_partition(
            nx.path_graph(3)
        )
        node_orbits = info.node_orbit_partition
        # endpoints 0 and 2 share, center 1 differs
        assert node_orbits[0] == node_orbits[2]
        assert node_orbits[0] != node_orbits[1]
        # both edges belong to one orbit
        assert set(info.edge_orbit_partition.values()) == {0}
        assert info.number_automorphisms == 2

    def test_precompute_global_numbering(self):
        """Channels and normalization vectors align across substructures."""
        enc = GSNFeatureEncoder(
            [nx.complete_graph(3), nx.path_graph(3)], lazy=False
        )
        # K3 -> 1 node orbit, P3 -> 2 node orbits => 3 node channels
        assert enc._node_channels == 3
        # K3 -> 1 edge orbit, P3 -> 1 edge orbit => 2 edge channels
        assert enc._edge_channels == 2
        # normalization = |Aut| per orbit: K3 -> 6, P3 -> 2
        assert enc._normalization_vector_n.tolist() == [6.0, 2.0, 2.0]
        assert enc._normalization_vector_e.tolist() == [6.0, 2.0]

    def test_forward_triangle_counts(self):
        """Each node/edge of a triangle participates in exactly one triangle."""
        enc = GSNFeatureEncoder([nx.complete_graph(3)], lazy=False)
        out = enc._encode(triangle_data())
        assert hasattr(out, "node_gsn_encodings")
        assert hasattr(out, "edge_gsn_encodings")
        # one node orbit channel, all counts normalized to 1.0
        assert out.node_gsn_encodings.shape == (3, 1)
        assert torch.allclose(out.node_gsn_encodings, torch.ones(3, 1))
        # undirected triangle -> 6 directed edges, each count 1.0
        assert out.edge_gsn_encodings.shape == (6, 1)
        assert torch.allclose(out.edge_gsn_encodings, torch.ones(6, 1))

    def test_forward_induced_semantics(self):
        """Counting P3 inside a triangle yields zeros (induced counting)."""
        enc = GSNFeatureEncoder([nx.path_graph(3)], lazy=False)
        out = enc._encode(triangle_data())
        assert torch.count_nonzero(out.node_gsn_encodings) == 0

    def test_forward_dtype(self):
        """Encodings adopt the node-feature dtype."""
        enc = GSNFeatureEncoder([nx.complete_graph(3)], lazy=False)
        out = enc._encode(triangle_data(dtype=torch.float64))
        # encodings adopt the (round-tripped) node-feature dtype
        assert out.node_gsn_encodings.dtype == out.x.dtype
        assert out.edge_gsn_encodings.dtype == out.x.dtype

    def test_forward_dtype_falls_back_to_x_0(self):
        """With no `x` (features under `x_0`, as in TopoBench), dtype from x_0.

        The encoder must not crash on a missing ``x`` and must derive the
        encoding dtype from ``x_0`` instead.
        """
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
        )
        # TopoBench-style graph: node features live under x_0, `x` is unset
        data = Data(edge_index=edge_index, num_nodes=3)
        data.x_0 = torch.ones(3, 2)
        assert getattr(data, "x", None) is None

        enc = GSNFeatureEncoder([nx.complete_graph(3)], lazy=False)
        out = enc._encode(
            data
        )  # would AttributeError if it assumed `data.x` existed
        # encodings adopt the (round-tripped) x_0 dtype
        assert out.node_gsn_encodings.dtype == out.x_0.dtype
        assert out.edge_gsn_encodings.dtype == out.x_0.dtype

    def test_lazy_matches_eager(self):
        """Lazy and eager precomputation give identical encodings."""
        lazy = GSNFeatureEncoder([nx.complete_graph(3)], lazy=True)
        eager = GSNFeatureEncoder([nx.complete_graph(3)], lazy=False)
        out_lazy = lazy._encode(triangle_data())
        out_eager = eager._encode(triangle_data())
        assert torch.allclose(
            out_lazy.node_gsn_encodings, out_eager.node_gsn_encodings
        )

    def test_parallel_matches_serial(self):
        """Parallel encoding (n_jobs=-1) matches serial (n_jobs=1) exactly.

        Guards the joblib parallelization over substructures: the merged
        counts must be identical to the in-process sequential path. Uses
        several substructures and a graph containing triangles and a
        4-cycle so more than one job actually contributes.
        """
        subs = [nx.complete_graph(3), nx.cycle_graph(4), nx.path_graph(3)]
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2],
                [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 2, 4],
            ],
            dtype=torch.long,
        )
        data = Data(x=torch.ones(5, 2), edge_index=edge_index)

        serial = GSNFeatureEncoder(subs, lazy=False, n_jobs=1)._encode(
            data.clone()
        )
        parallel = GSNFeatureEncoder(subs, lazy=False, n_jobs=-1)._encode(
            data.clone()
        )

        assert torch.allclose(
            serial.node_gsn_encodings, parallel.node_gsn_encodings
        )
        assert torch.allclose(
            serial.edge_gsn_encodings, parallel.edge_gsn_encodings
        )
        # sanity: the substructures actually match something
        assert torch.count_nonzero(serial.node_gsn_encodings) > 0

    def test_regression_non_canonical_edge_orientation(self):
        """Motifs stored non-ascending must not KeyError and count correctly.

        Regression for the edge-orbit keying bug: a substructure whose edges
        are stored as ``(2, 0)`` etc. previously raised ``KeyError``.
        """
        H = nx.Graph()
        H.add_edge(2, 0)
        H.add_edge(0, 1)
        H.add_edge(1, 2)
        enc = GSNFeatureEncoder([H], lazy=False)
        out = enc._encode(triangle_data())
        # identical to the canonical complete_graph(3) triangle motif
        assert torch.allclose(out.node_gsn_encodings, torch.ones(3, 1))

    def test_regression_edgeless_graph(self):
        """A graph with no edges runs and yields empty edge encodings.

        Regression for the ``KeyError: 'edge_gsn_encodings'`` on edgeless
        graphs.
        """
        enc = GSNFeatureEncoder([nx.complete_graph(3)], lazy=False)
        data = Data(
            x=torch.ones(3, 2), edge_index=torch.empty(2, 0, dtype=torch.long)
        )
        out = enc._encode(data)
        assert out.edge_gsn_encodings.shape == (0, enc._edge_channels)
        assert out.node_gsn_encodings.shape == (3, enc._node_channels)

    def test_forward_passthrough_when_encodings_present(self):
        """As a feature encoder, `forward` returns a batch that already has
        encodings unchanged (the pre-transform is the producer)."""
        enc = GSNFeatureEncoder(pyg_kword="gsn_encodings")
        data = triangle_data()
        data["node_gsn_encodings"] = torch.zeros(3, 1)
        assert enc.forward(data) is data

    def test_forward_warns_and_recomputes_when_encodings_missing(self):
        """`forward` falls back to recomputing when encodings are missing,
        warning that the `GSNEncodings` pre-transform should be applied."""
        enc = GSNFeatureEncoder([nx.complete_graph(3)], lazy=False)
        with pytest.warns(UserWarning, match="node_gsn_encodings"):
            out = enc.forward(triangle_data())
        # the fallback produces the same encodings as the pre-transform path
        assert torch.allclose(out.node_gsn_encodings, torch.ones(3, 1))
        assert torch.allclose(out.edge_gsn_encodings, torch.ones(6, 1))
