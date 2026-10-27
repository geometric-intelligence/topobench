"""Tests for the TRAWL feature-encoder adapter and component registration."""

import torch
from torch import nn

from topobench.nn.encoders.trawl import TRAWLFeatureEncoder


def test_native_component_registration():
    """Native discovery and direct Hydra imports share the TRAWL classes."""
    from topobench.loss import TBLoss
    from topobench.loss.loss import TBLoss as DirectTBLoss
    from topobench.nn.backbones import MODEL_CLASSES
    from topobench.nn.backbones.combinatorial import BACKBONE_CLASSES
    from topobench.nn.backbones.combinatorial.trawl import TRAWL
    from topobench.nn.encoders import FEATURE_ENCODERS
    from topobench.nn.readouts import READOUT_CLASSES
    from topobench.nn.wrappers import WRAPPER_CLASSES
    from topobench.transforms import TRANSFORMS
    from topobench.transforms.data_manipulations import DATA_MANIPULATIONS

    assert MODEL_CLASSES["TRAWL"] is TRAWL
    assert TBLoss is DirectTBLoss
    assert TBLoss.__module__ == "topobench.loss.loss"
    assert BACKBONE_CLASSES["TRAWL"] is TRAWL
    # Building blocks in trawl_utils are not registered as backbones.
    assert "NeighborhoodFusion" not in MODEL_CLASSES
    assert "PureTorchMambaBlock" not in MODEL_CLASSES
    assert FEATURE_ENCODERS["TRAWLFeatureEncoder"] is TRAWLFeatureEncoder
    assert "TRAWLReadout" in READOUT_CLASSES
    assert "TRAWLWrapper" in WRAPPER_CLASSES
    assert TRANSFORMS["TRAWLTransform"] is DATA_MANIPULATIONS["TRAWLTransform"]


def test_encoded_features_replace_trawl_signals():
    class Double(nn.Module):
        def forward(self, batch):
            for rank in range(2):
                batch[f"x_{rank}"] = 2 * batch[f"x_{rank}"]
            return batch

    batch = {"x_0": torch.ones(3, 2), "x_1": torch.ones(2, 2)}
    out = TRAWLFeatureEncoder(Double(), [0, 1])(batch)
    torch.testing.assert_close(out["trawl_signal_0"], torch.full((3, 2), 2.0))
    assert out["trawl_signal_1"] is out["x_1"]
