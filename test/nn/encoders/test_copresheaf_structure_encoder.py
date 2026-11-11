"""Tests for the copresheaf structural feature encoder."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.encoders.copresheaf_structure_encoder import (
    CopresheafStructureFeatureEncoder,
)


def _data():
    return Data(
        x_0=torch.randn(3, 5),
        x_1=torch.randn(2, 5),
        structure_0=torch.randn(3, 4),
        structure_1=torch.randn(2, 4),
        batch_0=torch.zeros(3, dtype=torch.long),
        batch_1=torch.zeros(2, dtype=torch.long),
    )


def test_structure_encoder_combines_signal_and_statistics():
    """The encoder projects every rank without changing cell counts."""
    encoder = CopresheafStructureFeatureEncoder(
        in_channels=[5, 5],
        out_channels=8,
        selected_dimensions=[0, 1],
    )

    output = encoder(_data())

    assert output.x_0.shape == (3, 8)
    assert output.x_1.shape == (2, 8)


def test_structure_encoder_requires_lifting_statistics():
    """Missing structural inputs fail with the relevant rank name."""
    data = _data()
    del data.structure_1
    encoder = CopresheafStructureFeatureEncoder(
        in_channels=[5, 5], out_channels=8, selected_dimensions=[0, 1]
    )

    with pytest.raises(KeyError, match="structure_1"):
        encoder(data)
