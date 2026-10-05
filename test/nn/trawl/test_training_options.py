"""Checkpoint, split and integration boundary regression tests."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from topobench.data.utils.split_utils import (
    imported_split,
    seeded_stratified_split,
)
from topobench.dataloader.samplers import EpochRandomSampler
from topobench.evaluator.checkpoint import (
    PredictionEnsemble,
    average_checkpoints,
)
from topobench.nn.backbones.combinatorial.trawl import WalkGraphLayer
from topobench.nn.readouts.trawl import TRAWLReadout

from .test_trawl import collate, model, prepare


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
    assert result["counter"].item() == 3
    with pytest.raises(ValueError):
        average_checkpoints([])


def test_prediction_ensemble(tmp_path):
    graph = prepare()

    class Pipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_encoder = nn.Identity()
            self.backbone = model([graph])
            self.readout = TRAWLReadout(16, 2, graph_dim=32)

    net = Pipeline().eval()
    paths, predictions = [], []
    batch = collate([graph])
    for i in range(2):
        with torch.no_grad():
            net.readout.head[-1].bias.add_(i)
        paths.append(tmp_path / f"{i}.ckpt")
        torch.save({"state_dict": net.state_dict()}, paths[-1])
        predictions.append(
            net.readout(net.backbone(batch), batch)["logits"].detach()
        )
    ensemble = PredictionEnsemble(net, paths).eval()
    torch.testing.assert_close(
        ensemble(batch)["logits"], torch.stack(predictions).mean(0)
    )


def test_split_and_epoch_order(tmp_path):
    labels = np.tile([0, 1], 20)
    config = OmegaConf.create({"train_prop": 0.5, "data_seed": 42})
    splits = seeded_stratified_split(labels, config)
    assert [len(splits[key]) for key in ["train", "valid", "test"]] == [
        20,
        10,
        10,
    ]
    path = tmp_path / "split.npz"
    np.savez(path, **splits)
    config.split_file = str(path)
    imported = imported_split(labels, config)
    np.testing.assert_array_equal(imported["train"], splits["train"])
    splits["test"][0] = splits["train"][0]
    np.savez(path, **splits)
    with pytest.raises(ValueError):
        imported_split(labels, config)
    sampler = EpochRandomSampler(range(20), 42)
    sampler.set_epoch(3)
    assert (
        list(sampler)
        == torch.randperm(
            20, generator=torch.Generator().manual_seed(46)
        ).tolist()
    )


def test_graph_adapter_never_crosses_walks():
    class SumNeighbors(nn.Module):
        def forward(self, x, edge_index):
            return torch.zeros_like(x).index_add(
                0, edge_index[1], x[edge_index[0]]
            )

    layer = WalkGraphLayer(SumNeighbors(), bidirectional=False)
    x = torch.arange(6.0).reshape(2, 3, 1)
    torch.testing.assert_close(
        layer(x).flatten(), torch.tensor([0.0, 0.0, 1.0, 0.0, 3.0, 4.0])
    )
