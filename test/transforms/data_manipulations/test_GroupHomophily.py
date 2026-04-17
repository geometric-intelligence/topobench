"""Test GroupCombinatorialHomophily transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import GroupCombinatorialHomophily


class TestGroupCombinatorialHomophily:
    """Test GroupCombinatorialHomophily transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = GroupCombinatorialHomophily(top_k=2)

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test default initialization
        default_transform = GroupCombinatorialHomophily()
        assert default_transform.type == "calcualte_group_combinatorial_homophily"
        assert default_transform.top_k == 3

        # Test custom initialization
        custom_transform = GroupCombinatorialHomophily(top_k=5)
        assert custom_transform.top_k == 5

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.transform)
        assert "GroupCombinatorialHomophily" in repr_str
        assert "calcualte_group_combinatorial_homophily" in repr_str

    def test_simple_hypergraph(self):
        """Test transform on a simple hypergraph."""
        # Create a simple hypergraph with 4 nodes, 2 hyperedges
        incidence = torch.zeros((4, 2))
        # First hyperedge contains nodes 0,1
        incidence[0:2, 0] = 1
        # Second hyperedge contains nodes 2,3
        incidence[2:4, 1] = 1

        data = Data(
            incidence_hyperedges=incidence.to_sparse(),
            y=torch.tensor([0, 0, 1, 1])  # Two classes
        )

        transformed = self.transform(data)

        assert "group_combinatorial_homophily" in transformed
        result = transformed["group_combinatorial_homophily"]

        # Should have entry for hyperedges of size 2
        assert "he_card=2" in result
        # Check matrices exist
        assert "D" in result["he_card=2"]
        assert "Dt" in result["he_card=2"]
        assert "Bt" in result["he_card=2"]
        assert "num_hyperedges" in result["he_card=2"]

    def test_mixed_size_hyperedges(self):
        """Test transform with hyperedges of different sizes."""
        # Create hypergraph with 5 nodes and 3 hyperedges of different sizes
        incidence = torch.zeros((5, 3))
        # Size 2 hyperedge
        incidence[0:2, 0] = 1
        # Size 3 hyperedge
        incidence[2:5, 1] = 1
        # Size 2 hyperedge
        incidence[1:3, 2] = 1

        data = Data(
            incidence_hyperedges=incidence.to_sparse(),
            y=torch.tensor([0, 0, 1, 1, 1])
        )

        transformed = self.transform(data)
        result = transformed["group_combinatorial_homophily"]

        # Should capture the two most frequent sizes
        assert len(result) <= self.transform.top_k
        assert any("he_card=2" in key for key in result.keys())

    def test_single_class(self):
        """Test transform when all nodes belong to the same class."""
        incidence = torch.zeros((3, 2))
        incidence[0:2, 0] = 1
        incidence[1:3, 1] = 1

        data = Data(
            incidence_hyperedges=incidence.to_sparse(),
            y=torch.tensor([0, 0, 0])  # All same class
        )

        transformed = self.transform(data)
        result = transformed["group_combinatorial_homophily"]

        for size_data in result.values():
            assert size_data["Dt"].shape[0] == 1  # Only one class
            assert not torch.isnan(size_data["Dt"]).any()
            assert not torch.isnan(size_data["Bt"]).any()

    def test_empty_hypergraph(self):
        """Test transform on empty hypergraph."""
        data = Data(
            incidence_hyperedges=torch.zeros((0, 0)).to_sparse(),
            y=torch.tensor([])
        )

        transformed = self.transform(data)
        assert "group_combinatorial_homophily" in transformed
        assert isinstance(transformed["group_combinatorial_homophily"], dict)

    def test_affinity_score(self):
        """Test affinity score calculation."""
        n_nodes = 10
        x_mod = 5  # nodes in class
        t = 2  # type-t degree
        k = 3  # hyperedge size

        score = self.transform.calculate_affinity_score(n_nodes, x_mod, t, k)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Should be a probability

    def test_matrix_calculations(self):
        """Test D and Bt matrix calculations."""
        # Create small hypergraph
        incidence = torch.zeros((4, 2))
        incidence[0:2, 0] = 1
        incidence[2:4, 1] = 1
        labels = torch.tensor([0, 0, 1, 1])

        unique_labels = {0: 2, 1: 2}  # Two nodes of each class
        class_node_idxs = {
            0: torch.tensor([0, 1]),
            1: torch.tensor([2, 3])
        }
        he_cardinalities = torch.tensor([2, 2])

        Dt, D = self.transform.calculate_D_matrix(
            incidence,
            labels,
            he_cardinalities,
            unique_labels,
            class_node_idxs
        )

        assert not torch.isnan(Dt).any()
        assert not torch.isnan(D).any()
        assert Dt.shape[1] == max(he_cardinalities)
        assert D.shape[0] == incidence.shape[0]

    def test_attribute_preservation(self):
        """Test that other attributes are preserved."""
        incidence = torch.zeros((3, 2))
        incidence[0:2, 0] = 1
        incidence[1:3, 1] = 1

        data = Data(
            incidence_hyperedges=incidence.to_sparse(),
            y=torch.tensor([0, 1, 0]),
            custom_attr="test"
        )

        transformed = self.transform(data)
        assert transformed.custom_attr == "test"
        assert torch.equal(transformed.y, data.y)
