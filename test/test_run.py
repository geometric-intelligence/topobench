"""Tests for helpers of the training entry point."""

import os

import torch
from omegaconf import OmegaConf

from topobench.run import (
    apply_determinism,
    enable_unused_parameter_detection,
    log_late_metrics,
)

# cuBLAS reads its workspace configuration when first used in a process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def test_models_can_request_unused_parameter_detection():
    cfg = OmegaConf.create(
        {
            "model": {"ddp_find_unused_parameters": True},
            "trainer": {"strategy": "ddp"},
        }
    )
    enable_unused_parameter_detection(cfg)
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
    other = OmegaConf.create(
        {"model": {"model_name": "gcn"}, "trainer": {"strategy": "ddp"}}
    )
    enable_unused_parameter_detection(other)
    assert other.trainer.strategy == "ddp"


def test_late_metrics_keep_existing_csv_columns(tmp_path):
    from lightning.pytorch.loggers import CSVLogger

    # The parent's writer exists before a spawned fit, whose workers write
    # rows through their own writer that the parent never sees.
    parent = CSVLogger(tmp_path, name="csv", version=0)
    assert parent.experiment is not None
    worker = CSVLogger(tmp_path, name="csv", version=0)
    worker.log_metrics({"val/accuracy": 0.5, "epoch": 0}, step=0)
    worker.save()
    log_late_metrics([parent], {"test_best_rerun/accuracy": 0.75})
    header = (
        (tmp_path / "csv/version_0/metrics.csv").read_text().splitlines()[0]
    )
    for key in ("val/accuracy", "epoch", "test_best_rerun/accuracy"):
        assert key in header


def test_deterministic_flag_survives_trainer_construction():
    from lightning import Trainer

    cfg = OmegaConf.create({"deterministic": "strict", "trainer": {}})
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        apply_determinism(cfg)
        Trainer(
            logger=False,
            enable_checkpointing=False,
            deterministic=cfg.trainer.deterministic,
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
