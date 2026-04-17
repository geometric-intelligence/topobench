"""Test the message passing module."""

import torch

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2simplicial.vietoris_rips_lifting import (
    SimplicialVietorisRipsLifting,
)


class TestSimplicialVietorisRipsLifting:
    """Test the SimplicialVietorisRipsLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = SimplicialVietorisRipsLifting(complex_dim=3, signed=True)
        self.lifting_unsigned = SimplicialVietorisRipsLifting(
            complex_dim=3, signed=False
        )

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1_singular_values_unsigned = torch.tensor(
            [2.8412, 2.1378, 1.9299, 1.7321, 1.5809, 1.4142, 1.2617, 0.7358]
        )

        expected_incidence_1_singular_values_signed = torch.tensor(
            [
                2.6622e00,
                2.2830e00,
                2.0000e00,
                2.0000e00,
                1.6697e00,
                1.4142e00,
                9.5538e-01,
                1.7728e-07,
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
            [2.5207, 2.0000, 1.5886, 1.4142, 1.4142, 1.0595]
        )

        expected_incidence_2_singular_values_signed = torch.tensor(
            [2.2738e00, 2.0000e00, 2.0000e00, 1.8196e00, 1.2324e00, 3.2893e-08]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_2.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_2.to_dense())

        assert torch.allclose(
            expected_incidence_2_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert torch.allclose(
            expected_incidence_2_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_2 (edges to triangles)."

        expected_incidence_3_singular_values_unsigned = torch.tensor([2.0000])

        expected_incidence_3_singular_values_signed = torch.tensor([2.0000])

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_3.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_3.to_dense())

        assert torch.allclose(
            expected_incidence_3_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_3 (triangles to tetrahedrons)."
        assert torch.allclose(
            expected_incidence_3_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_3 (triangles to tetrahedrons)."
