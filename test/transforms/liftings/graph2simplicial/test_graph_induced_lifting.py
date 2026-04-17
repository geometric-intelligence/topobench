"""Test the SimplicialGraphInducedLifting class."""

import torch

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2simplicial.graph_induced_lifting import (
    SimplicialGraphInducedLifting,
)


class TestSimplicialGraphInducedLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SimplicialGraphInducedLifting class
        self.lifting_signed = SimplicialGraphInducedLifting(complex_dim=3, signed=True)
        self.lifting_unsigned = SimplicialGraphInducedLifting(
            complex_dim=3, signed=False
        )

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1_singular_values_unsigned = torch.tensor(
            [3.7417, 2.4495, 2.4495, 2.4495, 2.4495, 2.4495, 2.4495, 2.4495]
        )

        expected_incidence_1_singular_values_signed = torch.tensor(
            [
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                6.8993e-08,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_1.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_1.to_dense())

        assert torch.allclose(
            expected_incidence_1_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert torch.allclose(
            expected_incidence_1_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2_singular_values_unsigned = torch.tensor(
            [
                4.1190,
                3.1623,
                3.1623,
                3.1623,
                3.0961,
                3.0000,
                3.0000,
                2.7564,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                1.7321,
                1.6350,
                1.4142,
                1.4142,
                1.0849,
            ]
        )

        expected_incidence_2_singular_values_signed = torch.tensor(
            [
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.6458e00,
                2.6458e00,
                2.2361e00,
                1.7321e00,
                1.7321e00,
                9.3758e-07,
                4.7145e-07,
                4.3417e-07,
                4.0241e-07,
                3.1333e-07,
                2.2512e-07,
                1.9160e-07,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_2.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_2.to_dense())
        assert torch.allclose(
            expected_incidence_2_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert torch.allclose(
            expected_incidence_2_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_2 (edges to triangles)."

        expected_incidence_3_singular_values_unsigned = torch.tensor(
            [
                3.8466,
                3.1379,
                3.0614,
                2.8749,
                2.8392,
                2.8125,
                2.5726,
                2.3709,
                2.2858,
                2.2369,
                2.1823,
                2.0724,
                2.0000,
                2.0000,
                2.0000,
                1.8937,
                1.7814,
                1.7321,
                1.7256,
                1.5469,
                1.5340,
                1.4834,
                1.4519,
                1.4359,
                1.4142,
                1.0525,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.9837,
                0.9462,
                0.8853,
                0.7850,
            ]
        )

        expected_incidence_3_singular_values_signed = torch.tensor(
            [
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.6933e00,
                2.6458e00,
                2.6458e00,
                2.6280e00,
                2.4495e00,
                2.3040e00,
                1.9475e00,
                1.7321e00,
                1.7321e00,
                1.7321e00,
                1.4823e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                1.0000e00,
                7.3584e-01,
                2.7959e-07,
                2.1776e-07,
                1.4498e-07,
                5.5373e-08,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_3.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_3.to_dense())

        assert torch.allclose(
            expected_incidence_3_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_3 (triangles to tetrahedrons)."
        assert torch.allclose(
            expected_incidence_3_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_3 (triangles to tetrahedrons)."
