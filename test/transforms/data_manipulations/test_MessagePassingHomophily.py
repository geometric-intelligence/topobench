
"""Test MessagePassingHomophily transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import MessagePassingHomophily


class TestMessagePassingHomophily:
   """Test MessagePassingHomophily transform."""

   def setup_method(self):
       """Set up test fixtures before each test method."""
       self.transform = MessagePassingHomophily(num_steps=2)

   def test_initialization(self):
       """Test initialization with different parameters."""
       # Test default initialization
       default_transform = MessagePassingHomophily()
       assert default_transform.type == "calcualte_message_passing_homophily"
       assert default_transform.num_steps == 3
       assert default_transform.incidence_field == "incidence_hyperedges"

       # Test custom initialization
       custom_transform = MessagePassingHomophily(
           num_steps=5,
           incidence_field="incidence_1"
       )
       assert custom_transform.num_steps == 5
       assert custom_transform.incidence_field == "incidence_1"

   def test_invalid_incidence_field(self):
       """Test initialization with invalid incidence field."""
       with pytest.raises(AssertionError):
           MessagePassingHomophily(incidence_field="invalid_field")

   def test_repr(self):
       """Test string representation."""
       repr_str = repr(self.transform)
       assert "MessagePassingHomophily" in repr_str
       assert "calcualte_message_passing_homophily" in repr_str

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

       # Check output structure
       assert "mp_homophily" in transformed
       result = transformed["mp_homophily"]

       # Check Ep and Np matrices
       assert "Ep" in result
       assert "Np" in result

       # Check dimensions
       assert result["Ep"].shape == (2, 2, 2)  # num_steps x num_edges x num_classes
       assert result["Np"].shape == (2, 4, 2)  # num_steps x num_nodes x num_classes

       # Check probability distributions sum to 1
       assert torch.allclose(result["Ep"].sum(dim=2), torch.ones(2, 2))
       assert torch.allclose(result["Np"].sum(dim=2), torch.ones(2, 4))

   def test_different_incidence_fields(self):
       """Test transform with different incidence fields."""
       incidence = torch.zeros((4, 2))
       incidence[0:2, 0] = 1
       incidence[2:4, 1] = 1

       data = Data(
           incidence_0=incidence.to_sparse(),
           incidence_1=incidence.to_sparse(),
           incidence_2=incidence.to_sparse(),
           y=torch.tensor([0, 0, 1, 1])
       )

       # Test each valid incidence field
       for field in ['incidence_0', 'incidence_1', 'incidence_2']:
           transform = MessagePassingHomophily(incidence_field=field)
           transformed = transform(data)
           result = transformed["mp_homophily"]

           assert result["Ep"].shape[1] == 2  # num_edges
           assert result["Np"].shape[1] == 4  # num_nodes

   def test_single_class(self):
       """Test transform when all nodes belong to same class."""
       incidence = torch.zeros((3, 2))
       incidence[0:2, 0] = 1
       incidence[1:3, 1] = 1

       data = Data(
           incidence_hyperedges=incidence.to_sparse(),
           y=torch.tensor([0, 0, 0])  # All same class
       )

       transformed = self.transform(data)
       result = transformed["mp_homophily"]

       # Check dimensions
       assert result["Ep"].shape == (2, 2, 1)  # Single class
       assert result["Np"].shape == (2, 3, 1)  # Single class

       # All probabilities should be 1 since there's only one class
       assert torch.all(result["Ep"] == 1.0)
       assert torch.all(result["Np"] == 1.0)

   def test_structured_hypergraph(self):
       """Test transform on structured hypergraph with known patterns."""
       n_nodes = 6
       n_edges = 3
       n_classes = 2

       # Create structured incidence matrix
       incidence = torch.zeros((n_nodes, n_edges))
       # First hyperedge: nodes 0,1,2 (class 0)
       incidence[0:3, 0] = 1
       # Second hyperedge: nodes 2,3,4 (mixed)
       incidence[2:5, 1] = 1
       # Third hyperedge: nodes 3,4,5 (class 1)
       incidence[3:6, 2] = 1

       # Define class labels
       labels = torch.tensor([0, 0, 0, 1, 1, 1])

       data = Data(
           incidence_hyperedges=incidence.to_sparse(),
           y=labels
       )

       transformed = self.transform(data)
       result = transformed["mp_homophily"]

       # Check dimensions
       assert result["Ep"].shape == (2, n_edges, n_classes)
       assert result["Np"].shape == (2, n_nodes, n_classes)

       # Check first step probabilities for first hyperedge (all class 0)
       expected_ep_step0_edge0 = torch.tensor([1.0, 0.0])
       assert torch.allclose(result["Ep"][0, 0], expected_ep_step0_edge0)

   def test_empty_hypergraph(self):
       """Test transform on empty hypergraph."""
       data = Data(
           incidence_hyperedges=torch.zeros((0, 0)).to_sparse(),
           y=torch.tensor([])
       )

       transformed = self.transform(data)
       result = transformed["mp_homophily"]

       assert result["Ep"].shape[0] == 2  # num_steps
       assert result["Np"].shape[0] == 2  # num_steps
       assert result["Ep"].shape[1] == 0  # no edges
       assert result["Np"].shape[1] == 0  # no nodes

   def test_attribute_preservation(self):
       """Test that other attributes are preserved."""
       incidence = torch.zeros((3, 2))
       incidence[0:2, 0] = 1
       incidence[1:3, 1] = 1

       data = Data(
           incidence_hyperedges=incidence.to_sparse(),
           y=torch.tensor([0, 1, 0]),
           custom_attr="test",
           edge_attr=torch.tensor([[1.], [2.]])
       )

       transformed = self.transform(data)

       # Check original attributes are preserved
       assert transformed.custom_attr == "test"
       assert torch.equal(transformed.y, data.y)
       assert torch.equal(transformed.edge_attr, data.edge_attr)
