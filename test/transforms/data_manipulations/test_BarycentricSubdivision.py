"""Unit tests for the BarycentricSubdivisionTransform class."""

import pytest
import torch
from topobench.transforms.data_manipulations import BarycentricSubdivisionTransform


class TestBarycentricSubdivisionTransform:
    """Test the BarycentricSubdivisionTransform class."""

    def test_barycentric_subdivision_transform(self, sg1_clique_lifted):
        """Test the forward method of the BarycentricSubdivisionTransform class.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A simple graph data object with a clique lifting.
        """
        # Define the parameters for the transform
        parameters = {
            "keys_to_keep": ["x", "y", "num_nodes"],
            "complex_dim": 3,
            "neighborhoods": ["up_incidence-0", "up_incidence-1", "up_incidence-2"],
            "signed": True,
        }

        # Create the transform
        transform = BarycentricSubdivisionTransform(**parameters)

        # Apply the transform
        data = sg1_clique_lifted.clone()
        out = transform(data)

        # Check if the output has the expected keys
        assert "up_incidence-0" in out
        assert "up_incidence-1" in out
        assert "up_incidence-2" in out
        assert "shape" in out
        assert "x" in out
        assert "y" in out

        # Note: topobench.data.utils.utils.select_neighborhoods_of_interest
        # always keeps keys that contain "incidence" and no hyphen.
        assert "incidence_0" in out
        assert "incidence_1" in out
        assert "incidence_2" in out
        assert "incidence_3" in out

        # Check if the subdivision changed the number of simplices
        # The number of simplices in Sd(K) is much larger than in K
        assert out.x.shape[0] > sg1_clique_lifted.num_nodes

    def test_barycentric_subdivision_transform_empty_neighborhoods(self, sg1_clique_lifted):
        """Test the forward method with empty neighborhoods.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A simple graph data object with a clique lifting.
        """
        # Define the parameters for the transform
        parameters = {
            "keys_to_keep": ["x", "y", "num_nodes"],
            "complex_dim": 3,
            "neighborhoods": [],
            "signed": True,
        }

        # Create the transform
        transform = BarycentricSubdivisionTransform(**parameters)

        # Apply the transform
        data = sg1_clique_lifted.clone()
        out = transform(data)
        assert "shape" in out
        assert "incidence_1" in out

    def test_repr(self):
        """Test the __repr__ method."""
        parameters = {
            "keys_to_keep": ["x"],
            "complex_dim": 2,
            "neighborhoods": [],
            "signed": False,
        }
        transform = BarycentricSubdivisionTransform(**parameters)
        repr_str = transform.__repr__()
        assert "BarycentricSubdivisionTransform" in repr_str
        assert "domain2domain" in repr_str
