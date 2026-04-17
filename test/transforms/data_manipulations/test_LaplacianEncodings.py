"""Test LapPE (Laplacian Positional Encoding) Transform."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import LapPE


class TestLapPE:
    """Test LapPE (Laplacian Positional Encoding) transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = LapPE(max_pe_dim=8, concat_to_x=True)

    def test_initialization(self):
        """Test initialization of the transform."""
        assert self.transform.max_pe_dim == 8
        assert self.transform.concat_to_x is True
        assert self.transform.include_eigenvalues is False
        assert self.transform.include_first is False

        # Test with different parameters
        transform = LapPE(
            max_pe_dim=16,
            include_eigenvalues=True,
            include_first=True,
            concat_to_x=False
        )
        assert transform.max_pe_dim == 16
        assert transform.include_eigenvalues is True
        assert transform.include_first is True
        assert transform.concat_to_x is False

    def test_forward_simple_graph(self):
        """Test transform on a simple graph."""
        # Create a simple 3-node graph with a cycle
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that LapPE was concatenated to x
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3  # num_nodes
        assert transformed.x.shape[1] == 1 + self.transform.max_pe_dim  # original + LapPE

        # Check that original features are preserved
        assert torch.equal(transformed.x[:, 0:1], x)

        # Check that LapPE features are not all zeros
        lappe_features = transformed.x[:, 1:]
        assert not torch.allclose(lappe_features, torch.zeros_like(lappe_features))

    def test_forward_no_concat(self):
        """Test transform when concat_to_x is False."""
        transform = LapPE(max_pe_dim=8, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        # Check that LapPE is stored separately
        assert hasattr(transformed, "LapPE")
        assert transformed.LapPE.shape == (3, 8)

        # Check that original x is unchanged
        assert torch.equal(transformed.x, x)

    def test_forward_no_features(self):
        """Test transform when data.x is None."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that x is created with LapPE
        assert transformed.x is not None
        assert transformed.x.shape == (3, self.transform.max_pe_dim)

    def test_empty_graph(self):
        """Test transform on an empty graph (no edges)."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=5)

        transformed = self.transform(data)

        # Check that LapPE is all zeros for empty graph
        assert transformed.x is not None
        assert transformed.x.shape == (5, self.transform.max_pe_dim)
        assert torch.allclose(transformed.x, torch.zeros(5, self.transform.max_pe_dim))

    def test_single_node_graph(self):
        """Test transform on a graph with a single node."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(edge_index=edge_index, num_nodes=1)

        transformed = self.transform(data)

        # Check that LapPE is all zeros for single node
        assert transformed.x is not None
        assert transformed.x.shape == (1, self.transform.max_pe_dim)
        assert torch.allclose(transformed.x, torch.zeros(1, self.transform.max_pe_dim))

    def test_disconnected_graph(self):
        """Test transform on a disconnected graph."""
        # Two disconnected edges: 0-1 and 2-3
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data)

        # Check that LapPE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (4, self.transform.max_pe_dim)

    def test_self_loop_graph(self):
        """Test transform on a graph with self-loops."""
        # Graph with self-loops
        edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])
        x = torch.tensor([[1.0], [2.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=2)

        transformed = self.transform(data)

        # Check that LapPE is computed correctly
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

        # Check that LapPE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (4, self.transform.max_pe_dim)

    def test_directed_graph(self):
        """Test transform on a directed graph (unidirectional edges)."""
        # Directed path: 0 -> 1 -> 2 -> 3
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data)

        # Check that LapPE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (4, self.transform.max_pe_dim)

    def test_isolated_node(self):
        """Test transform with isolated nodes."""
        # Graph with node 2 isolated
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data)

        # Check that LapPE is computed
        assert transformed.x is not None
        assert transformed.x.shape == (3, self.transform.max_pe_dim)

    def test_different_pe_dimensions(self):
        """Test transform with different max_pe_dim values."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        for dim in [1, 2, 4, 8, 16]:
            transform = LapPE(max_pe_dim=dim, concat_to_x=False)
            transformed = transform(data)
            assert transformed.LapPE.shape == (3, dim)

    def test_include_eigenvalues(self):
        """Test transform with include_eigenvalues=True."""
        transform = LapPE(max_pe_dim=8, include_eigenvalues=True, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        # When include_eigenvalues=True, dimension should be 2 * max_pe_dim
        assert transformed.LapPE.shape == (3, 2 * 8)

    def test_include_first_eigenvector(self):
        """Test transform with include_first=True vs False."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        # Test with include_first=False (default)
        transform_exclude = LapPE(max_pe_dim=4, include_first=False, concat_to_x=False)
        transformed_exclude = transform_exclude(data)

        # Test with include_first=True
        transform_include = LapPE(max_pe_dim=4, include_first=True, concat_to_x=False)
        transformed_include = transform_include(data)

        # Both should have the same shape (due to padding)
        assert transformed_exclude.LapPE.shape == (3, 4)
        assert transformed_include.LapPE.shape == (3, 4)

        # The encodings should be different
        assert not torch.allclose(transformed_exclude.LapPE, transformed_include.LapPE)

    def test_orthogonality_of_eigenvectors(self):
        """Test that eigenvectors are approximately orthogonal."""
        transform = LapPE(max_pe_dim=4, include_first=False, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = transform(data)
        lappe = transformed.LapPE

        # Compute Gram matrix (should be approximately identity)
        gram = lappe.T @ lappe
        # Check if non-diagonal elements are close to zero
        gram_offdiag = gram - torch.diag(torch.diag(gram))
        assert torch.allclose(gram_offdiag, torch.zeros_like(gram_offdiag), atol=0.1)

    def test_sign_consistency(self):
        """Test that sign ambiguity resolution is applied."""
        transform = LapPE(max_pe_dim=4, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        # Run transform
        transformed = transform(data.clone())

        # Check that for each eigenvector column, the maximum absolute value
        # element has consistent sign (the sign convention used in the code)
        lappe = transformed.LapPE

        for col in range(lappe.shape[1]):
            if lappe[:, col].abs().sum() > 1e-6:  # Skip zero columns (padding)
                max_idx = torch.argmax(lappe[:, col].abs())
                # The sign convention ensures the element with max abs value is positive
                assert lappe[max_idx, col] >= 0

    def test_padding_behavior(self):
        """Test that padding is applied when num_nodes < max_pe_dim."""
        # Small graph with max_pe_dim larger than possible eigenvalues
        transform = LapPE(max_pe_dim=10, concat_to_x=False)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, num_nodes=2)

        transformed = transform(data)

        # Should still have max_pe_dim dimensions (with padding)
        assert transformed.LapPE.shape == (2, 10)

        # Some columns should be all zeros (padding)
        non_zero_cols = (transformed.LapPE.abs().sum(dim=0) > 1e-6).sum()
        assert non_zero_cols < 10  # Some padding should exist

    def test_chain_graph(self):
        """Test transform on a chain/path graph."""
        # Chain: 0-1-2-3-4
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3]
        ])
        data = Data(edge_index=edge_index, num_nodes=5)

        transform = LapPE(max_pe_dim=4, concat_to_x=False)
        transformed = transform(data)

        # Check that LapPE is computed
        assert transformed.LapPE.shape == (5, 4)

        # For a symmetric chain, end nodes should have similar encodings
        # (though potentially with opposite signs)
        assert torch.allclose(
            transformed.LapPE[0].abs(),
            transformed.LapPE[4].abs(),
            atol=0.1
        )

    def test_star_graph(self):
        """Test transform on a star graph (one central node connected to all others)."""
        # Star graph: node 0 is center, connected to nodes 1, 2, 3
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3],
            [1, 0, 2, 0, 3, 0]
        ])
        data = Data(edge_index=edge_index, num_nodes=4)

        transform = LapPE(max_pe_dim=4, concat_to_x=False)
        transformed = transform(data)

        # Check that LapPE is computed
        assert transformed.LapPE.shape == (4, 4)

        # Leaf nodes should have similar magnitude encodings (up to sign)
        # In a star graph, leaf nodes are structurally equivalent
        norms_leaves = torch.norm(transformed.LapPE[1:], dim=1)
        assert torch.allclose(norms_leaves, norms_leaves.mean() * torch.ones_like(norms_leaves), atol=0.15)

    def test_cycle_graph(self):
        """Test transform on a cycle graph."""
        # Cycle: 0-1-2-3-0
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3]
        ])
        data = Data(edge_index=edge_index, num_nodes=4)

        transform = LapPE(max_pe_dim=4, concat_to_x=False)
        transformed = transform(data)

        # Check that LapPE is computed
        assert transformed.LapPE.shape == (4, 4)

        # All nodes in a regular cycle should have similar magnitude encodings
        norms = torch.norm(transformed.LapPE, dim=1)
        assert torch.allclose(norms, norms.mean() * torch.ones_like(norms), atol=0.1)

    def test_device_consistency(self):
        """Test that LapPE respects the device of input data."""
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

        # Check that LapPE is computed for all nodes
        assert transformed.x is not None
        assert transformed.x.shape == (num_nodes, self.transform.max_pe_dim)

    def test_eigenvalue_ordering(self):
        """Test that eigenvalues are in ascending order."""
        transform = LapPE(
            max_pe_dim=4,
            include_eigenvalues=True,
            include_first=True,
            concat_to_x=False
        )
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        data = Data(edge_index=edge_index, num_nodes=4)

        transformed = transform(data)

        # Extract eigenvalues (second half of the encoding)
        eigenvalues = transformed.LapPE[0, 4:]  # All nodes should have same eigenvalues

        # Check that non-zero eigenvalues are in ascending order
        non_zero_mask = eigenvalues > 1e-6
        if non_zero_mask.sum() > 1:
            non_zero_eigenvalues = eigenvalues[non_zero_mask]
            assert torch.all(non_zero_eigenvalues[1:] >= non_zero_eigenvalues[:-1] - 1e-5)

    def test_numerical_stability(self):
        """Test transform on potentially problematic cases for numerical stability."""
        # Very small graph
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, num_nodes=2)

        transform = LapPE(max_pe_dim=4, concat_to_x=False)
        transformed = transform(data)

        # Should not contain NaN or Inf
        assert not torch.isnan(transformed.LapPE).any()
        assert not torch.isinf(transformed.LapPE).any()

    def test_undirected_graph_symmetry(self):
        """Test LapPE on undirected graph exhibits expected symmetries."""
        # Symmetric graph: 0-1-2-1-0 (bidirectional edges forming a triangle)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ])
        data = Data(edge_index=edge_index, num_nodes=3)

        transform = LapPE(max_pe_dim=4, concat_to_x=False)
        transformed = transform(data)

        lappe = transformed.LapPE

        # Check that LapPE is computed (triangle graph symmetry is complex
        # due to the eigenvector structure, so we just verify computation)
        assert not torch.allclose(lappe, torch.zeros_like(lappe))
        assert not torch.isnan(lappe).any()
        assert not torch.isinf(lappe).any()

    def test_batch_processing(self):
        """Test that transform works with individual graphs."""
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        # Apply transform
        transformed = self.transform(data)

        # Verify basic properties
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] > 1

    def test_max_pe_dim_larger_than_graph(self):
        """Test when max_pe_dim is larger than the number of nodes."""
        transform = LapPE(max_pe_dim=20, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        # Should still have max_pe_dim dimensions (with padding)
        assert transformed.LapPE.shape == (3, 20)

        # Most dimensions should be zero (padding)
        non_zero_cols = (transformed.LapPE.abs().sum(dim=0) > 1e-6).sum()
        assert non_zero_cols <= 3  # At most num_nodes eigenvectors

    @pytest.mark.parametrize("max_pe_dim", [1, 2, 4, 8, 16])
    def test_parametrized_dimensions(self, max_pe_dim):
        """Parametrized test for different LapPE dimensions.

        Parameters
        ----------
        max_pe_dim : int
            The maximum dimension for LapPE.
        """
        transform = LapPE(max_pe_dim=max_pe_dim, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        assert transformed.LapPE.shape == (3, max_pe_dim)
        # Check for NaN or Inf
        assert not torch.isnan(transformed.LapPE).any()
        assert not torch.isinf(transformed.LapPE).any()

    @pytest.mark.parametrize("include_first", [True, False])
    def test_parametrized_include_first(self, include_first):
        """Parametrized test for include_first parameter.

        Parameters
        ----------
        include_first : bool
            Whether to include the first eigenvector in the encoding.
        """
        transform = LapPE(
            max_pe_dim=4,
            include_first=include_first,
            concat_to_x=False
        )
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        assert transformed.LapPE.shape == (3, 4)
        assert not torch.isnan(transformed.LapPE).any()

    @pytest.mark.parametrize("include_eigenvalues", [True, False])
    def test_parametrized_include_eigenvalues(self, include_eigenvalues):
        """Parametrized test for include_eigenvalues parameter.

        Parameters
        ----------
        include_eigenvalues : bool
            Whether to include eigenvalues in the encoding.
        """
        max_pe_dim = 4
        transform = LapPE(
            max_pe_dim=max_pe_dim,
            include_eigenvalues=include_eigenvalues,
            concat_to_x=False
        )
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(edge_index=edge_index, num_nodes=3)

        transformed = transform(data)

        expected_dim = 2 * max_pe_dim if include_eigenvalues else max_pe_dim
        assert transformed.LapPE.shape == (3, expected_dim)
