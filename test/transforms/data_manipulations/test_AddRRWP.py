"""Unit tests for the AddRRWP transform (RRWP positional encodings)."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.transforms.data_manipulations import DATA_MANIPULATIONS
from topobench.transforms.data_manipulations.rrwp_positional_encodings import (
    AddRRWP,
    compute_rrwp,
)
from topobench.transforms.data_transform import DataTransform


def _cycle_graph(num_nodes: int) -> Data:
    """Create an undirected cycle graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the cycle.

    Returns
    -------
    torch_geometric.data.Data
        The cycle graph data object.
    """
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.cat(
        [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
    )
    return Data(x=torch.randn(num_nodes, 3), edge_index=edge_index)


def _path_graph(num_nodes: int) -> Data:
    """Create an undirected path graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    torch_geometric.data.Data
        The path graph data object.
    """
    src = torch.arange(num_nodes - 1)
    dst = src + 1
    edge_index = torch.cat(
        [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
    )
    return Data(x=torch.randn(num_nodes, 3), edge_index=edge_index)


class TestComputeRRWP:
    """Test the compute_rrwp helper function."""

    def test_shapes(self):
        """Test output shapes for a small graph."""
        data = _cycle_graph(6)
        walk_length = 4
        abs_pe, rel_index, rel_val, deg = compute_rrwp(
            data.edge_index, data.num_nodes, walk_length
        )

        assert abs_pe.shape == (6, walk_length)
        assert rel_index.shape[0] == 2
        assert rel_val.shape == (rel_index.shape[1], walk_length)
        assert deg.shape == (6,)

    def test_identity_channel(self):
        """Test that the first channel is the identity matrix."""
        data = _cycle_graph(5)
        abs_pe, rel_index, rel_val, _ = compute_rrwp(
            data.edge_index, data.num_nodes, 3
        )

        assert torch.allclose(abs_pe[:, 0], torch.ones(5))
        self_loops = rel_index[0] == rel_index[1]
        assert torch.allclose(
            rel_val[self_loops, 0], torch.ones(int(self_loops.sum()))
        )
        assert torch.allclose(
            rel_val[~self_loops, 0], torch.zeros(int((~self_loops).sum()))
        )

    def test_one_step_probabilities(self):
        """Test that the second channel matches D^{-1} A exactly."""
        data = _cycle_graph(4)
        _, rel_index, rel_val, deg = compute_rrwp(
            data.edge_index, data.num_nodes, 2
        )

        # On a cycle, every node has degree 2 and hops to each neighbor
        # with probability 1/2. Pair orientation is (source=j, target=i)
        # with value P_{i, j}.
        assert torch.allclose(deg, torch.full((4,), 2.0))
        for source, target, val in zip(
            rel_index[0], rel_index[1], rel_val, strict=True
        ):
            if source == target:
                expected = torch.tensor([1.0, 0.0])
            else:
                expected = torch.tensor([0.0, 0.5])
            assert torch.allclose(val, expected)

    def test_matches_dense_matrix_powers(self):
        """Test the values against explicitly computed matrix powers."""
        data = _path_graph(5)
        walk_length = 4
        _, rel_index, rel_val, _ = compute_rrwp(
            data.edge_index, data.num_nodes, walk_length
        )

        adj = torch.zeros(5, 5)
        adj[data.edge_index[0], data.edge_index[1]] = 1.0
        transition = adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        dense = torch.stack(
            [torch.matrix_power(transition, k) for k in range(walk_length)],
            dim=-1,
        )

        source, target = rel_index[0], rel_index[1]
        assert torch.allclose(rel_val, dense[target, source], atol=1e-6)

        # All non-zero entries of the dense tensor must be represented.
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[target, source] = True
        assert torch.all(dense.abs().sum(-1)[~mask] == 0)

    def test_sparsity_of_relative_encodings(self):
        """Test that unreachable node pairs are not materialized."""
        data = _path_graph(6)
        _, rel_index, _, _ = compute_rrwp(data.edge_index, data.num_nodes, 2)

        # With K=2 (identity + one hop), only self-loops and direct
        # neighbors can appear.
        distance = (rel_index[0] - rel_index[1]).abs()
        assert torch.all(distance <= 1)

    def test_isolated_node(self):
        """Test a graph containing an isolated node."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        abs_pe, rel_index, rel_val, deg = compute_rrwp(edge_index, 3, 3)

        assert deg[2] == 0
        # The isolated node only appears in its own identity pair.
        expected_abs = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(abs_pe[2], expected_abs)
        node_2_pairs = (rel_index == 2).any(dim=0)
        assert int(node_2_pairs.sum()) == 1
        assert torch.allclose(rel_val[node_2_pairs][0], expected_abs)

    def test_empty_graph(self):
        """Test a graph with no edges."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        abs_pe, rel_index, rel_val, deg = compute_rrwp(edge_index, 3, 4)

        assert abs_pe.shape == (3, 4)
        assert torch.all(deg == 0)
        # Only identity pairs remain.
        assert rel_index.shape[1] == 3
        assert torch.equal(rel_index[0], rel_index[1])


class TestAddRRWP:
    """Test the AddRRWP transform."""

    def test_registration(self):
        """Test that the transform is discoverable by TopoBench."""
        assert "AddRRWP" in DATA_MANIPULATIONS

    def test_data_transform_instantiation(self):
        """Test instantiation through TopoBench's DataTransform."""
        transform = DataTransform(
            transform_name="AddRRWP",
            transform_type="data manipulation",
            walk_length=4,
        )
        data = transform(_cycle_graph(5))
        assert data.rrwp.shape == (5, 4)

    def test_invalid_walk_length(self):
        """Test that too-short walk lengths are rejected."""
        with pytest.raises(ValueError, match="walk_length"):
            AddRRWP(walk_length=1)

    def test_repr(self):
        """Test the string representation."""
        assert repr(AddRRWP(walk_length=6)) == "AddRRWP(walk_length=6)"

    def test_attached_attributes(self, simple_graph_0):
        """Test the attributes attached to the data object.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        walk_length = 5
        data = AddRRWP(walk_length=walk_length)(simple_graph_0)

        num_nodes = simple_graph_0.num_nodes
        assert data.rrwp.shape == (num_nodes, walk_length)
        assert data.rrwp_index.shape[0] == 2
        assert data.rrwp_val.shape == (
            data.rrwp_index.shape[1],
            walk_length,
        )
        assert data.log_deg.shape == (num_nodes,)

    def test_log_deg(self):
        """Test that log_deg equals log(1 + degree)."""
        data = AddRRWP(walk_length=3)(_cycle_graph(4))
        assert torch.allclose(data.log_deg, torch.full((4,), 3.0).log())

    def test_diagonal_matches_abs_encoding(self):
        """Test that rrwp equals the diagonal of the relative encodings."""
        data = AddRRWP(walk_length=4)(_cycle_graph(6))

        self_loops = data.rrwp_index[0] == data.rrwp_index[1]
        diag_nodes = data.rrwp_index[0, self_loops]
        diag_vals = data.rrwp_val[self_loops]
        assert torch.allclose(data.rrwp[diag_nodes], diag_vals)

    def test_batching_offsets_pair_index(self):
        """Test that rrwp_index is offset like edge_index when batching."""
        transform = AddRRWP(walk_length=3)
        data_0 = transform(_cycle_graph(4))
        data_1 = transform(_path_graph(3))
        batch = Batch.from_data_list([data_0, data_1])

        assert batch.rrwp.shape == (7, 3)
        assert batch.rrwp_index.max() == 6
        # Pairs of the second graph must be offset by 4 nodes.
        num_pairs_0 = data_0.rrwp_index.shape[1]
        assert torch.equal(
            batch.rrwp_index[:, num_pairs_0:], data_1.rrwp_index + 4
        )
        assert torch.allclose(
            batch.rrwp_val, torch.cat([data_0.rrwp_val, data_1.rrwp_val])
        )

    def test_deterministic(self):
        """Test that the transform is deterministic."""
        transform = AddRRWP(walk_length=4)
        graph = _cycle_graph(5)
        data_a = transform(graph.clone())
        data_b = transform(graph.clone())

        assert torch.equal(data_a.rrwp, data_b.rrwp)
        assert torch.equal(data_a.rrwp_index, data_b.rrwp_index)
        assert torch.equal(data_a.rrwp_val, data_b.rrwp_val)
