"""Test the CofaceCCLifting module."""


import pytest
import torch
from torch_geometric.data import Data

from topobench.data.utils.utils import load_manual_simplicial_complex
from topobench.transforms.liftings.simplicial2combinatorial.coface_cc_lifting import (
    CofaceCCLifting,
)


class TestCofaceCCLifting:
    """Test the CofaceCCLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_simplicial_complex()

        # Initialise the CofaceCCLifting class
        self.coface_lift = CofaceCCLifting()
        self.coface_lift_keep = CofaceCCLifting(keep_features=True)

    def test_feature_preservation(self):
        """Test that features are preserved when lifting."""
        x_0 = torch.rand(3, 5)
        x_1 = torch.rand(3, 6)
        x_2 = torch.rand(1, 7)

        incidence_1_dense = torch.tensor([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=torch.float)

        incidence_2_dense = torch.tensor([
            [1],
            [1],
            [1]
        ], dtype=torch.float)

        # Convert to sparse
        incidence_1 = incidence_1_dense.to_sparse()
        incidence_2 = incidence_2_dense.to_sparse()

        data = Data(x_0=x_0, x_1=x_1, x_2=x_2, incidence_1=incidence_1, incidence_2=incidence_2)

        lifted_data = self.coface_lift_keep(data)

        assert torch.allclose(lifted_data.x_0, x_0)
        assert torch.allclose(lifted_data.x_1, x_1)
        assert torch.allclose(lifted_data.x_2, x_2)

    def test_lift_topology(self):
        """Test that lifted topology."""
        # Test the lift_topology method
        lifted_data = self.coface_lift.forward(self.data.clone())

        expected_n_3_cells = 3

        expected_incidence_3 = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        assert (
            expected_incidence_3 == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 ."
        assert (
            expected_n_3_cells == lifted_data.x_3.size(0)
        ), "Something is wrong with the number of 3-cells."
