"""Test the message passing module."""


from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.pointcloud2simplicial.random_flag_complex import (
    RandomFlagComplexLifting,
)


class TestRandomFlagComplexLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()
        del self.data["edge_attr"]
        del self.data["edge_index"]

        self.lifting_p_0 = RandomFlagComplexLifting(steps=10, p=0)
        self.lifting_p_1 = RandomFlagComplexLifting(steps=1, p=1)
        self.lifting_hp = RandomFlagComplexLifting(steps=100, alpha=0.01)


    def test_empty(self):
        """Test that the lifted topology is empty."""
        lifted_data = self.lifting_p_0.forward(self.data.clone())
        assert(lifted_data.x_1.size(0) == 0)
