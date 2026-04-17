"""Test the message passing module."""

import numpy as np
import torch
from torch_geometric.data import Data

from topobench.transforms.liftings.pointcloud2hypergraph.mogmst_lifting import (
    MoGMSTLifting,
)


class TestMoGMSTLifting:
    """Test the MoGMSTLifting class."""

    def setup_method(self):
        """Set up the test."""
        # Load the graph
        y = torch.tensor([0 for i in range(8)], dtype=torch.int32)
        pos = torch.tensor(
            [
                [-1.44, -1.55],
                [-2, -2],
                [-1.18, -2.38],
                [-1.26, 3.28],
                [-0.59, 3.68],
                [-0.7, 3.33],
                [0.52, 0.09],
                [0.16, 0.45],
            ]
        )
        self.data = Data(x=pos, y=torch.tensor(y))

        # Initialise the HypergraphKHopLifting class
        self.lifting = MoGMSTLifting(min_components=3, random_state=0)
        self.lifting2 = MoGMSTLifting(random_state=0)

    def test_find_mog(self):
        """Test the find_mog method."""
        labels, num_components, means = self.lifting.find_mog(
            self.data.clone().x.numpy()
        )

        assert num_components == 3, "Wrong number of components"

        assert (
            labels[0] == labels[1] == labels[2]
            and labels[3] == labels[4] == labels[5]
            and labels[6] == labels[7]
            and labels[0] != labels[3]
            and labels[3] != labels[6]
            and labels[0] != labels[6]
        ), "Labels have not been assigned correctly"

        labels, num_components, means = self.lifting2.find_mog(
            self.data.clone().x.numpy()
        )

        assert num_components == 4, "Wrong number of components"

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_k = self.lifting.forward(self.data.clone())

        expected_n_hyperedges = 6

        assert (
            lifted_data_k.num_hyperedges == expected_n_hyperedges
        ), "Wrong number of hyperedges (k=1)"

        incidence_np = lifted_data_k.incidence_hyperedges.to_dense().numpy()
        asg_inc = incidence_np[:, :3]
        mst_inc = incidence_np[:, 3:]

        assert (
            (
                (asg_inc[:3] == asg_inc[0]).all()
                and (asg_inc[3:6] == asg_inc[3]).all()
                and (asg_inc[6] == asg_inc[7]).all()
            )
            and not (asg_inc[0] == asg_inc[3]).all()
            and not (asg_inc[0] == asg_inc[6]).all()
            and not (asg_inc[3] == asg_inc[6]).all()
        ), "Something went wrong with point assignment to means"

        assert (
            (mst_inc[:6] == mst_inc[0]).all()
            and (mst_inc[6] == mst_inc[7]).all()
            and np.sum(mst_inc[0]) == 1
            and np.sum(mst_inc[6]) == 2
        ), "Something went wrong with MST calculation/incidence matrix creation"
