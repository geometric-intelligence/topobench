"""Test RWSE (Random Walk Structural Encoding) Transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import RWSE


class TestRWSE:
    """Test RWSE (Random Walk Structural Encoding) transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = RWSE(max_pe_dim=8, concat_to_x=True)

    def test_initialization(self):
        """Test initialization of the transform."""
        assert self.transform.max_pe_dim == 8
        assert self.transform.concat_to_x is True

        # Test with different parameters
        transform = RWSE(max_pe_dim=16, concat_to_x=False)
        assert transform.max_pe_dim == 16
        assert transform.concat_to_x is False

    def test_forward_simple_graph(self):
        """Test transform on a simple graph."""
        # Create a simple 3-node graph with a cycle
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that RWSE was concatenated to x
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3  # num_nodes
        assert transformed.x.shape[1] == 1 + self.transform.max_pe_dim  # original + RWSE

        # Check that original features are preserved
        assert torch.equal(transformed.x[:, 0:1], x)

        # Check that RWSE features are not all zeros
        rwse_features = transformed.x[:, 1:]
        assert not torch.allclose(rwse_features, torch.zeros_like(rwse_features))

    def test_forward_no_concat(self):
        """Test transform when concat_to_x is False."""
        transform = RWSE(max_pe_dim=8, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        # Check that RWSE is stored separately
        assert hasattr(transformed, "RWSE")
        assert transformed.RWSE.shape == (3, 8)

        # Check that original x is unchanged
        assert torch.equal(transformed.x, x)

    def test_forward_no_features(self):
        """Test transform when data.x is None."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that x is created with RWSE
        assert transformed.x is not None
        assert transformed.x.shape == (3, self.transform.max_pe_dim)

    def test_empty_graph(self):
        """Test transform on an empty graph (no edges)."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=5)

        transformed = self.transform(data)

        # Check that RWSE is all zeros for empty graph
        assert transformed.x is not None
        assert transformed.x.shape == (5, self.transform.max_pe_dim)
        assert torch.allclose(transformed.x, torch.zeros(5, self.transform.max_pe_dim))

    def test_single_node_graph(self):
        """Test transform on a graph with a single node."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=1)

        transformed = self.transform(data)

        # Check that RWSE is all zeros for single node
        assert transformed.x is not None
        assert transformed.x.shape == (1, self.transform.max_pe_dim)
        assert torch.allclose(transformed.x, torch.zeros(1, self.transform.max_pe_dim))

    def test_disconnected_graph(self):
        """Test transform on a disconnected graph."""
        # Two disconnected edges: 0-1 and 2-3
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data)

        # Check that RWSE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (4, self.transform.max_pe_dim)

    def test_self_loop_graph(self):
        """Test transform on a graph with self-loops."""
        # Graph with self-loops
        edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])
        x = torch.tensor([[1.0], [2.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=2)

        transformed = self.transform(data)

        # Check that RWSE is computed correctly
        assert transformed.x is not None
        assert transformed.x.shape[0] == 2
        assert transformed.x.shape[1] == 1 + self.transform.max_pe_dim

    def test_complete_graph(self):
        """Test transform on a complete graph."""
        # Complete graph on 4 nodes
        edges = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges).t()
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data)

        # Check that RWSE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (4, self.transform.max_pe_dim)

        # In a complete graph, random walk should distribute evenly
        # Return probability should converge to 1/n for each node
        rwse = transformed.x
        # For later steps, values should approach 1/4 = 0.25
        later_steps = rwse[:, -3:]  # last 3 steps
        # All nodes should have similar return probabilities
        assert torch.allclose(later_steps, later_steps.mean(dim=0, keepdim=True), atol=0.1)

    def test_directed_graph(self):
        """Test transform on a directed graph (unidirectional edges)."""
        # Directed path: 0 -> 1 -> 2 -> 3
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data)

        # Check that RWSE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (4, self.transform.max_pe_dim)

        # Node 3 has no outgoing edges, so return probability should be 0
        assert torch.allclose(transformed.x[3], torch.zeros(self.transform.max_pe_dim))

    def test_isolated_node(self):
        """Test transform with isolated nodes."""
        # Graph with node 2 isolated
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that RWSE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (3, self.transform.max_pe_dim)

        # Isolated node (node 2) should have zero RWSE
        assert torch.allclose(transformed.x[2], torch.zeros(self.transform.max_pe_dim))

    def test_different_pe_dimensions(self):
        """Test transform with different max_pe_dim values."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        for dim in [1, 4, 8, 16, 32]:
            transform = RWSE(max_pe_dim=dim, concat_to_x=False)
            transformed = transform(data)
            assert transformed.RWSE.shape == (3, dim)

    def test_return_probability_properties(self):
        """Test mathematical properties of return probabilities."""
        # Simple cycle graph: 0 -> 1 -> 2 -> 0
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transform = RWSE(max_pe_dim=12, concat_to_x=False)
        transformed = transform(data)

        rwse = transformed.RWSE

        # All return probabilities should be between 0 and 1
        assert torch.all(rwse >= 0.0)
        assert torch.all(rwse <= 1.0)

        # For a cycle of length 3, return probability at step 3 should be 1
        # (you must return to starting node after 3 steps)
        assert torch.allclose(rwse[:, 2], torch.ones(3), atol=0.01)

        # At step 6, should also return (multiple of cycle length)
        assert torch.allclose(rwse[:, 5], torch.ones(3), atol=0.01)

    def test_undirected_graph_symmetry(self):
        """Test RWSE on undirected graph exhibits expected symmetries."""
        # Undirected path: 0-1-2-1-0 (bidirectional edges)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transform = RWSE(max_pe_dim=8, concat_to_x=False)
        transformed = transform(data)

        rwse = transformed.RWSE

        # Nodes 0 and 2 should have similar RWSE (they're symmetric in this graph)
        # Node 1 is in the center, so it will be different
        assert torch.allclose(rwse[0], rwse[2], atol=0.1)

    def test_device_consistency(self):
        """Test that RWSE respects the device of input data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]]).cuda()
        x = torch.tensor([[1.0], [2.0], [3.0]]).cuda()
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that output is on the same device
        assert transformed.x.device == x.device
        assert transformed.x.is_cuda

    def test_backward_compatibility(self):
        """Test that all original data attributes are preserved."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        y = torch.tensor([0, 1, 0])
        custom_attr = torch.tensor([10, 20, 30])

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            custom_attr=custom_attr,
            num_nodes=3
        )

        transformed = self.transform(data)

        # Check that all attributes are preserved
        assert hasattr(transformed, "y")
        assert hasattr(transformed, "custom_attr")
        assert torch.equal(transformed.y, y)
        assert torch.equal(transformed.custom_attr, custom_attr)
        assert torch.equal(transformed.edge_index, edge_index)

    def test_large_graph_performance(self):
        """Test transform on a larger graph to ensure it runs without errors."""
        # Create a larger random graph
        num_nodes = 100
        num_edges = 300
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        data = Data(edge_index=edge_index, num_nodes=num_nodes)

        transformed = self.transform(data)

        # Check that RWSE is computed for all nodes
        assert transformed.x is not None
        assert transformed.x.shape == (num_nodes, self.transform.max_pe_dim)

    def test_batch_processing(self):
        """Test that transform works with batch data (if applicable)."""
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        # Apply transform
        transformed = self.transform(data)

        # Verify basic properties
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] > 1

    @pytest.mark.parametrize("max_pe_dim", [1, 2, 4, 8, 16])
    def test_parametrized_dimensions(self, max_pe_dim):
        """Parametrized test for different RWSE dimensions.

        Parameters
        ----------
        max_pe_dim : int
            The maximum positional encoding dimension to test.
        """
        transform = RWSE(max_pe_dim=max_pe_dim, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        assert transformed.RWSE.shape == (3, max_pe_dim)
        assert torch.all(transformed.RWSE >= 0.0)
        assert torch.all(transformed.RWSE <= 1.0)
