"""Tests for the NoReadOut layer."""

import pytest
import torch
import torch_geometric.data as tg_data
from topobench.nn.readouts.base import AbstractZeroCellReadOut
from topobench.nn.readouts.identical import NoReadOut


class TestNoReadOut:
    """Tests for the NoReadOut layer."""
    @pytest.fixture
    def base_kwargs(self):
        """Fixture providing the required base parameters.

        Returns
        -------
        dict
            The base parameters for the NoReadOut layer.
        """
        return {
            'hidden_dim': 64,
            'out_channels': 32,
            'task_level': 'graph'
        }

    @pytest.fixture
    def readout_layer(self, base_kwargs):
        """Fixture to create a NoReadOut instance for testing.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.

        Returns
        -------
        NoReadOut
            A NoReadOut instance for testing.
        """
        return NoReadOut(**base_kwargs)

    @pytest.fixture
    def sample_model_output(self):
        """Fixture to create a sample model output dictionary.

        Returns
        -------
        dict
            A sample model output dictionary.
        """
        return {
            'x_0': torch.randn(10, 64),  # Required key for model output
            'edge_indices': torch.randint(0, 10, (2, 15)),
            'other_data': torch.randn(10, 32)
        }

    @pytest.fixture
    def sample_batch(self):
        """Fixture to create a sample batch of graph data.

        Returns
        -------
        torch_geometric.data.Data
            A sample batch of graph data.
        """
        return tg_data.Data(
            x=torch.randn(10, 32),
            edge_index=torch.randint(0, 10, (2, 15)),
            batch_0=torch.zeros(10, dtype=torch.long)  # Required key for batch data
        )

    def test_initialization(self, base_kwargs):
        """Test that NoReadOut initializes correctly with required parameters.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        readout = NoReadOut(**base_kwargs)
        assert isinstance(readout, NoReadOut)
        assert isinstance(readout, AbstractZeroCellReadOut)

    def test_forward_pass_returns_unchanged_output(self, readout_layer, sample_model_output, sample_batch):
        """Test that forward pass returns the model output without modifications.

        Parameters
        ----------
        readout_layer : NoReadOut
            A fixture to create a NoReadOut instance for testing.
        sample_model_output : dict
            A fixture to create a sample model output dictionary.
        sample_batch : torch_geometric.data.Data
            A fixture to create a sample batch of graph data.
        """
        original_output = sample_model_output.copy()
        output = readout_layer(sample_model_output, sample_batch)

        # The output should contain the original data plus the computed logits
        for key in original_output:
            assert key in output
            assert torch.equal(output[key], original_output[key])
        assert 'logits' in output

    def test_invalid_task_level(self, base_kwargs):
        """Test that initialization fails with invalid task_level.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        invalid_kwargs = base_kwargs.copy()
        invalid_kwargs['task_level'] = 'invalid_level'
        with pytest.raises(AssertionError, match="Invalid task_level"):
            NoReadOut(**invalid_kwargs)

    def test_repr(self, readout_layer):
        """Test the string representation of the NoReadOut layer.

        Parameters
        ----------
        readout_layer : NoReadOut
            A fixture to create a NoReadOut instance for testing.
        """
        assert str(readout_layer) == "NoReadOut()"
        assert repr(readout_layer) == "NoReadOut()"

    def test_forward_pass_with_different_batch_sizes(self, readout_layer):
        """Test that forward pass works with different batch sizes.

        Parameters
        ----------
        readout_layer : NoReadOut
            A fixture to create a NoReadOut instance for testing.
        """
        # Test with single graph
        single_batch = tg_data.Data(
            x=torch.randn(5, 32),
            edge_index=torch.randint(0, 5, (2, 8)),
            batch_0=torch.zeros(5, dtype=torch.long)
        )
        single_output = {
            'x_0': torch.randn(5, 64),
            'embeddings': torch.randn(5, 64)
        }
        result = readout_layer(single_output, single_batch)
        assert 'logits' in result

        # Test with multiple graphs
        multi_batch = tg_data.Data(
            x=torch.randn(15, 32),
            edge_index=torch.randint(0, 15, (2, 25)),
            batch_0=torch.cat([torch.zeros(5), torch.ones(5), torch.ones(5) * 2]).long()
        )
        multi_output = {
            'x_0': torch.randn(15, 64),
            'embeddings': torch.randn(15, 64)
        }
        result = readout_layer(multi_output, multi_batch)
        assert 'logits' in result

    def test_kwargs_handling(self, base_kwargs):
        """Test that the layer correctly handles both required and additional keyword arguments.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        additional_kwargs = {
            'random_param': 42,
            'another_param': 'test',
            'pooling_type': 'mean'  # Valid additional parameter
        }
        kwargs = {**base_kwargs, **additional_kwargs}
        readout = NoReadOut(**kwargs)
        assert isinstance(readout, NoReadOut)
