"""Unit tests for the DGMStructureFeatureEncoder module."""

import pytest
import torch
import torch_geometric
import numpy as np

from topobench.nn.encoders import DGMStructureFeatureEncoder
from topobench.nn.encoders.kdgm import DGM_d

class TestDGMStructureFeatureEncoder:
    """Test suite for the DGMStructureFeatureEncoder class.

    This test class covers various aspects of the DGMStructureFeatureEncoder,
    including initialization, forward pass, selective encoding, and
    configuration settings.
    """

    @pytest.fixture
    def sample_data(self):
        """Create a sample PyG Data object for testing.

        Returns
        -------
        torch_geometric.data.Data
            A data object with simulated multi-dimensional features and batch information.
        """
        data = torch_geometric.data.Data()

        # Simulate multi-dimensional features
        data.x_0 = torch.randn(10, 5)  # 10 nodes, 5 features
        data.x_1 = torch.randn(10, 7)  # 10 nodes, 7 features
        data.x_2 = torch.randn(10, 9)  # 10 nodes, 9 features

        # Add batch information
        data.batch_0 = torch.zeros(10, dtype=torch.long)
        data.batch_1 = torch.zeros(10, dtype=torch.long)
        data.batch_2 = torch.zeros(10, dtype=torch.long)

        return data

    def test_initialization(self, sample_data):
        """Test encoder initialization with different configurations.

        Parameters
        ----------
        sample_data : torch_geometric.data.Data
            Fixture providing sample graph data for testing.
        """
        # Test with default settings
        encoder = DGMStructureFeatureEncoder(
            in_channels=[5, 7, 9],
            out_channels=64
        )

        # Test __repr__ method
        repr_str = encoder.__repr__()

        # Check basic attributes
        assert encoder.in_channels == [5, 7, 9]
        assert encoder.out_channels == 64
        assert len(encoder.dimensions) == 3

    def test_forward_pass(self, sample_data):
        """Test forward pass of the encoder.

        Parameters
        ----------
        sample_data : torch_geometric.data.Data
            Fixture providing sample graph data for testing.
        """
        encoder = DGMStructureFeatureEncoder(
            in_channels=[5, 7, 9],
            out_channels=64,
            selected_dimensions=[0, 1, 2]
        )

        # Perform forward pass
        output_data = encoder(sample_data)

        # Check output attributes
        for i in [0, 1, 2]:
            # Check encoded features exist
            assert hasattr(output_data, f'x_{i}')
            assert output_data[f'x_{i}'].shape[1] == 64

            # Check auxiliary attributes
            assert hasattr(output_data, f'x_aux_{i}')
            assert hasattr(output_data, f'logprobs_{i}')

        # Check edges index exists
        assert 'edges_index' in output_data

    def test_selective_encoding(self, sample_data):
        """Test encoding only specific dimensions.

        Parameters
        ----------
        sample_data : torch_geometric.data.Data
            Fixture providing sample graph data for testing.
        """
        encoder = DGMStructureFeatureEncoder(
            in_channels=[5, 7, 9],
            out_channels=64,
            selected_dimensions=[0, 1]  # Only encode the first two dimensions
        )

        # Perform forward pass
        output_data = encoder(sample_data)

        # Verify encoding for selected dimensions
        assert hasattr(output_data, 'x_1')
        assert output_data['x_0'].shape[1] == 64
        assert output_data['x_1'].shape[1] == 64
        assert output_data['x_2'].shape[1] == 9

    def test_dropout_configuration(self):
        """Test dropout configuration for the encoder."""
        # Test with non-zero dropout
        encoder = DGMStructureFeatureEncoder(
            in_channels=[5, 7, 9],
            out_channels=64,
            proj_dropout=0.5
        )

        # Check dropout value
        for i in encoder.dimensions:
            encoder_module = getattr(encoder, f'encoder_{i}')
            assert encoder_module.base_enc.dropout.p == 0.5
            assert encoder_module.embed_f.dropout.p == 0.5

    @pytest.mark.parametrize("in_channels", [
        [5],  # Single dimension
        [5, 7, 9],  # Multiple dimensions
        [10, 20, 30, 40]  # More dimensions
    ])
    def test_variable_input_dimensions(self, sample_data, in_channels):
        """Test encoder with varying input dimensions.

        Parameters
        ----------
        sample_data : torch_geometric.data.Data
            Fixture providing sample graph data for testing.
        in_channels : list
            List of input channel dimensions to test.
        """
        encoder = DGMStructureFeatureEncoder(
            in_channels=in_channels,
            out_channels=64
        )

        # Prepare data dynamically
        data = torch_geometric.data.Data()
        for i, channel in enumerate(in_channels):
            setattr(data, f'x_{i}', torch.randn(10, channel))
            setattr(data, f'batch_{i}', torch.zeros(10, dtype=torch.long))

        # Perform forward pass
        output_data = encoder(data)

        # Verify encoding for each dimension
        for i in range(len(in_channels)):
            assert hasattr(output_data, f'x_{i}')
            assert output_data[f'x_{i}'].shape[1] == 64

def pytest_configure():
    """Custom pytest configuration.

    Sets up default configuration values for testing.
    """
    pytest.in_channels = [5, 7, 9]
    pytest.out_channels = 64
