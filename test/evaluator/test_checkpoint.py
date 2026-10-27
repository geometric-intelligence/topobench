"""Tests for top-K checkpoint averaging and prediction ensembles."""

import pytest
import torch
from torch import nn

from topobench.evaluator.checkpoint import (
    PassThroughReadout,
    PredictionEnsemble,
    average_checkpoints,
)


def test_checkpoint_buffer_averaging(tmp_path):
    paths = [tmp_path / f"{i}.ckpt" for i in range(2)]
    for i, path in enumerate(paths):
        torch.save(
            {
                "state_dict": {
                    "weight": torch.tensor([float(i * 2)]),
                    "counter": torch.tensor(i + 3),
                }
            },
            path,
        )
    result = average_checkpoints(paths)
    assert result["weight"].item() == 1
    # Integer buffers are taken from the first (best) checkpoint.
    assert result["counter"].item() == 3
    with pytest.raises(ValueError):
        average_checkpoints([])


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, batch):
        return {"x_0": self.linear(batch["x"])}


class _Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, model_out, batch):
        model_out["logits"] = self.linear(model_out["x_0"])
        return model_out


class _Batch(dict):
    def clone(self):
        return _Batch(self)


def test_prediction_ensemble_averages_member_logits(tmp_path):
    torch.manual_seed(0)
    model = nn.Module()
    model.feature_encoder = nn.Identity()
    model.backbone = _Backbone()
    model.readout = _Readout()
    batch = _Batch(x=torch.randn(4, 3))
    paths, predictions = [], []
    for i in range(2):
        with torch.no_grad():
            model.readout.linear.bias.add_(i)
        paths.append(tmp_path / f"{i}.ckpt")
        torch.save({"state_dict": model.state_dict()}, paths[-1])
        predictions.append(
            model.readout(model.backbone(batch), batch)["logits"].detach()
        )
    ensemble = PredictionEnsemble(model, paths).eval()
    output = PassThroughReadout("graph")(ensemble(batch), batch)
    torch.testing.assert_close(
        output["logits"], torch.stack(predictions).mean(0)
    )
