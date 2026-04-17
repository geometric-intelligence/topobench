"""Test AbstractLifting module."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.liftings import AbstractLifting

class TestAbstractLifting:
    """Test the AbstractLifting class."""

    def setup_method(self):
        """Set up test fixtures for each test method.

        Creates a concrete subclass of AbstractLifting for testing purposes.
        """
        class ConcreteLifting(AbstractLifting):
            """Concrete implementation of AbstractLifting for testing."""

            def lift_topology(self, data):
                """Implementation of abstract method that calls parent's method.

                Parameters
                ----------
                data : torch_geometric.data.Data
                    The input data to be lifted.

                Returns
                -------
                dict
                    Empty dictionary as this is just for testing.

                Raises
                ------
                NotImplementedError
                    Always raises this error as it calls the parent's abstract method.
                """
                return super().lift_topology(data)

        self.lifting = ConcreteLifting(feature_lifting=None)

    def test_lift_topology_raises_not_implemented(self):
        """Test that the abstract lift_topology method raises NotImplementedError.

        Verifies that calling lift_topology on an abstract class implementation
        raises NotImplementedError as expected.
        """
        dummy_data = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )

        with pytest.raises(NotImplementedError):
            self.lifting.lift_topology(dummy_data)
