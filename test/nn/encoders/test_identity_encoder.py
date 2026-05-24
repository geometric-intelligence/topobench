"""Unit tests for IdentityFeatureEncoder."""

import torch
import torch_geometric

from topobench.nn.encoders import IdentityFeatureEncoder


def test_identity_encoder_passthrough():
    """Test that IdentityFeatureEncoder returns input unchanged."""
    in_channels = [16]
    out_channels = 16
    encoder = IdentityFeatureEncoder(in_channels, out_channels)

    batch = torch_geometric.data.Data(
        x_0=torch.randn(10, 16),
        x=torch.randn(10, 16),
    )
    result = encoder(batch)
    assert torch.equal(result.x_0, batch.x_0)


def test_identity_encoder_out_channels():
    """Test that out_channels attribute is set correctly."""
    encoder = IdentityFeatureEncoder([32], 32)
    assert encoder.out_channels == 32


def test_identity_encoder_different_dims():
    """Test with mismatched in_channels and out_channels."""
    encoder = IdentityFeatureEncoder([128], 64)
    batch = torch_geometric.data.Data(
        x_0=torch.randn(10, 128),
        x=torch.randn(10, 128),
    )
    # Should still pass through without error
    result = encoder(batch)
    assert result.x_0.shape == (10, 128)
