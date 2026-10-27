"""Tests for self-supervised TRAWL pretraining."""

import pytest
import torch
from torch import nn

from test._utils.trawl import collate, complex_data, network, prepare
from topobench.model.pretraining.trawl import (
    FullPrecisionValidation,
    PretrainingCheckpoint,
    TRAWLPretrainer,
    validation_frequency,
)
from topobench.nn.encoders.trawl import TRAWLFeatureEncoder


def test_pretraining_uses_no_labels():
    graphs = [prepare(), prepare(complex_data(empty=True))]
    pretrainer = TRAWLPretrainer(
        network(graphs),
        objectives={"features": 1.0, "colors": 1.0, "topology": 1.0},
    )
    batch = collate(graphs)
    del batch.y
    loss = pretrainer._step(batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_base_pretraining_masks_differ_per_microbatch():
    graphs = [prepare(), prepare()]
    pretrainer = TRAWLPretrainer(network(graphs), mask_probability=0.5)
    pretrainer.log = lambda *args, **kwargs: None
    batch = collate(graphs)

    def loss(batch_idx):
        # Fix the walk step so only the mask seed can change.
        pretrainer.backbone.reset_sampling_step()
        return pretrainer._step(batch, batch_idx=batch_idx).item()

    first, again, second = loss(0), loss(0), loss(1)
    assert first == again and first != second


def test_pretraining_updates_native_feature_encoder():
    class RankEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.ModuleList(
                [nn.Linear(width, 5) for width in [2, 3, 1]]
            )

        def forward(self, batch):
            for rank, layer in enumerate(self.projections):
                batch[f"x_{rank}"] = layer(batch[f"x_{rank}"])
            return batch

    data = prepare()
    net = network([data], in_channels=[5, 5, 5])
    encoder = TRAWLFeatureEncoder(RankEncoder(), [0, 1, 2])
    trainer = TRAWLPretrainer(
        net, feature_encoder=encoder, target_widths=[2, 3, 1]
    )
    trainer.log = lambda *args, **kwargs: None
    trainer._step(collate([data])).backward()
    assert encoder.encoder.projections[0].weight.grad is not None


class _Strategy:
    @staticmethod
    def reduce_boolean_decision(decision):
        return decision


class _Trainer:
    strategy = _Strategy()


def test_pretraining_checkpoint_min_delta(tmp_path):
    callback = PretrainingCheckpoint(
        dirpath=tmp_path, monitor="loss", mode="min", min_delta=1e-3
    )
    callback.best_k_models = {"a": torch.tensor(1.0)}
    callback.kth_best_model_path = "a"
    assert not callback.check_monitor_top_k(_Trainer(), torch.tensor(0.9995))
    assert callback.check_monitor_top_k(_Trainer(), torch.tensor(0.998))
    with pytest.raises(ValueError):
        PretrainingCheckpoint(dirpath=tmp_path, min_delta=-1.0)


def test_pretraining_resume_keeps_best_across_run_directories(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "epoch=3-step=9.ckpt").write_bytes(b"best")
    state = {
        "dirpath": str(old),
        "best_model_path": str(old / "moved" / "epoch=3-step=9.ckpt"),
        "best_model_score": torch.tensor(0.25),
        "best_k_models": {},
        "kth_best_model_path": "",
        "kth_value": torch.tensor(0.25),
        "last_model_path": "",
    }
    callback = PretrainingCheckpoint(
        dirpath=new, monitor="loss", mode="min", resume_dir=old
    )
    callback.load_state_dict(state)
    target = new / "epoch=3-step=9.ckpt"
    assert target.read_bytes() == b"best"
    assert callback.best_model_path == str(target)
    assert callback.best_k_models == {str(target): torch.tensor(0.25)}
    # A later, worse epoch must not replace the restored best.
    assert not callback.check_monitor_top_k(_Trainer(), torch.tensor(0.3))
    missing = PretrainingCheckpoint(dirpath=tmp_path / "other")
    state["best_model_path"] = str(tmp_path / "absent.ckpt")
    with pytest.raises(FileNotFoundError):
        missing.load_state_dict(state)


def test_validation_frequency():
    class Holder:
        check_val_every_n_epoch = 5

    assert validation_frequency(Holder()) == 5
    assert validation_frequency(None) == 1


def _cpu_autocast_enabled():
    # ``torch.is_autocast_enabled(device_type)`` is unavailable before torch 2.4.
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


def test_full_precision_validation():
    class Probe(FullPrecisionValidation, nn.Module):
        device = torch.device("cpu")

        def _step(self, batch, validation=False, batch_idx=0):
            return _cpu_autocast_enabled()

    probe = Probe()
    probe.evaluation_autocast = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert probe.validation_step(None, 0) is False
