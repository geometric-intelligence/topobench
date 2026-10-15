"""Main entry point for training and testing models."""

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, open_dict

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from topobench.utils.config_resolvers import register_all_resolvers

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


# Register custom resolvers before Hydra initialization
register_all_resolvers()


def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


def log_late_metrics(loggers: list[Logger], metrics: dict) -> None:
    """Log and flush metrics after fitting has finalized the loggers.

    A spawned fit writes CSV rows from its workers, so the parent's CSV writer
    does not know the existing header. Seed it from the file before saving,
    otherwise Lightning rewrites the file with only the new columns.

    Parameters
    ----------
    loggers : list[Logger]
        Run loggers.
    metrics : dict
        Metrics to append.
    """
    for lgr in loggers:
        lgr.log_metrics(metrics)
        experiment = lgr.experiment if rank_zero_only.rank == 0 else None
        path = getattr(experiment, "metrics_file_path", None)
        keys = getattr(experiment, "metrics_keys", None)
        if path and keys is not None and os.path.isfile(path):
            with open(path, newline="") as stream:
                header = next(csv.reader(stream), [])
            keys.extend(key for key in header if key not in keys)
        lgr.save()


def apply_determinism(cfg: DictConfig) -> None:
    """Configure deterministic kernels when ``cfg.deterministic`` is set.

    ``True`` warns on operations without a deterministic implementation;
    ``"strict"`` raises instead. The choice is also written to
    ``cfg.trainer.deterministic`` so Lightning keeps it.

    Parameters
    ----------
    cfg : DictConfig
        Run configuration; ``cfg.trainer`` is updated in place.
    """
    deterministic = cfg.get("deterministic", False)
    if deterministic:
        # Enable cudnn deterministic algorithms for reproducibility.
        # "strict" raises on operations without a deterministic kernel
        # instead of warning, so a run is either repeatable or fails.
        strict = deterministic == "strict"
        # cuBLAS needs a fixed workspace for repeatable reductions; it must
        # be set before the first cuBLAS handle is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        # Every Lightning Trainer applies its own ``deterministic`` flag and
        # would otherwise reset this (the default trainer config sets False).
        with open_dict(cfg):
            cfg.trainer.deterministic = True if strict else "warn"
        log.info(
            "Enabled cudnn.deterministic and torch.use_deterministic_algorithms"
            f" ({'strict' if strict else 'warn only'})"
        )


DDP_STRATEGIES = {"ddp", "ddp_spawn", "ddp_fork", "ddp_notebook"}


def enable_unused_parameter_detection(cfg: DictConfig) -> None:
    """Let TRAWL train under Lightning DDP string strategies.

    Which TRAWL parameters receive gradients depends on the configuration
    and data (pretraining decoders, fusion for union walks, absent ranks).
    Plain DDP aborts on such parameters, so TRAWL runs select Lightning's
    registered ``*_find_unused_parameters_true`` variant instead. Spawned
    runs on Linux also share tensors through the file system.

    Parameters
    ----------
    cfg : DictConfig
        Run configuration; ``cfg.trainer.strategy`` is updated in place.
    """
    strategy = cfg.trainer.get("strategy")
    if cfg.model.get("model_name") != "trawl" or not isinstance(strategy, str):
        return
    if strategy.startswith("ddp_spawn") and sys.platform.startswith("linux"):
        # Spawned workers receive the datasets through shared memory. The
        # default file-descriptor strategy keeps one descriptor per tensor
        # in the parent, which exhausts ordinary limits on real datasets.
        torch.multiprocessing.set_sharing_strategy("file_system")
    if strategy not in DDP_STRATEGIES:
        return
    replacement = f"{strategy}_find_unused_parameters_true"
    log.info(f"TRAWL uses trainer.strategy={replacement} under DDP")
    with open_dict(cfg):
        cfg.trainer.strategy = replacement


@task_wrapper
def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    apply_determinism(cfg)

    enable_unused_parameter_detection(cfg)

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.dataset.loader._target_}>")
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    # Preprocess dataset and load the splits
    log.info("Instantiating preprocessor...")
    transform_config = (
        hydra.utils.instantiate(cfg.transforms)
        if cfg.get("transforms", None) is not None
        else None
    )
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg.dataset.split_params)
    )
    # Prepare datamodule
    log.info("Instantiating datamodule...")
    if cfg.dataset.parameters.task_level in ["node", "graph"]:
        datamodule = TBDataloader(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            **cfg.dataset.get("dataloader_params", {}),
        )
    else:
        raise ValueError("Invalid task_level")

    # Model for us is Network + logic: inputs backbone, readout, losses
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )

    if hasattr(model.backbone, "initialize"):
        model.backbone.initialize(dataset_train.data_lst)

    model.configure_compilation()

    if cfg.get("pretraining", {}).get("enabled", False) and cfg.get("train"):
        if cfg.get("ckpt_path"):
            log.info("Resuming supervised checkpoint; skipping pretraining.")
        else:
            from topobench.model.trawl_pretraining import run_pretraining

            run_pretraining(
                model,
                datamodule,
                cfg.pretraining,
                cfg.trainer,
                output_dir=cfg.paths.output_dir,
            )

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Log to wandb preprocessor time
    if logger:
        for log_temp in logger:
            if isinstance(log_temp, L.pytorch.loggers.wandb.WandbLogger):
                log_temp.log_metrics(
                    {
                        "preprocessor_time": preprocessor.preprocessing_time,
                    }
                )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,  # Log metrics every step (Lightning requires >=1)
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.model.get("model_name") == "trawl":
        from topobench.utils.trawl_provenance import write_run_manifest

        if rank_zero_only.rank == 0:
            write_run_manifest(cfg, model, datamodule)

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

    train_metrics = trainer.callback_metrics

    test_metrics = {}
    if cfg.get("test"):
        log.info("Starting testing!")

        test_metrics = rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=datamodule,
            device=model.device,
            callbacks=callbacks,
            logger=logger,
        )

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def rerun_best_model_checkpoint(
    checkpoint_model: LightningModule,
    cfg: DictConfig,
    datamodule: LightningModule,
    device: torch.device,
    callbacks: list[Callback],
    logger: list[Logger],
) -> dict:
    """Rerun the best model checkpoint on validation and test datasets to log final metrics.

    This function iterates through the callbacks to locate the `ModelCheckpoint`, loads the
    best model weights, and runs a test pass on both the validation and test dataloaders.
    Metrics are logged with `val_best_rerun/` and `test_best_rerun/` prefixes to ensure
    metrics reflect the best model state rather than the final epoch.

    Parameters
    ----------
    checkpoint_model : LightningModule
        The model instance to load weights into.
    cfg : DictConfig
        Configuration composed by Hydra.
    datamodule : LightningModule
        The data module providing `val_dataloader` and `test_dataloader`.
    device : torch.device
        The target device (CPU/GPU) for the model.
    callbacks : list[Callback]
        A list of callbacks to search for the `ModelCheckpoint`.
    logger : list[Logger]
        A list of loggers (e.g., WandbLogger) to record the re-run metrics.

    Returns
    -------
    dict
        Re-run metrics with `val_best_rerun/` and `test_best_rerun/` prefixes.
    """
    final_metrics = {}
    selected_checkpoints = []
    model_path = None
    if not cfg.get("train", True) and cfg.get("ckpt_path"):
        model_path = Path(cfg.ckpt_path)
        strategy = cfg.get("evaluation", {}).get("checkpoint", "best")
        if strategy == "best":
            state = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            checkpoint_model.load_state_dict(state["state_dict"], strict=True)
            checkpoint_model.to(device)
            selected_checkpoints = [
                {"path": str(model_path), "validation_score": None}
            ]
            callbacks = []
        else:
            from topobench.callbacks.model_checkpoint import (
                RankedModelCheckpoint,
            )

            callback = next(
                (
                    item
                    for item in callbacks
                    if isinstance(item, RankedModelCheckpoint)
                ),
                None,
            )
            if callback is None:
                raise ValueError(
                    "Top-K checkpoint replay requires RankedModelCheckpoint"
                )
            callback.best_model_path = str(model_path)
            callback.best_k_models = {}
            callback.restore_ranking(relocate=True)
            callbacks = [callback]
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            if hasattr(callback, "restore_ranking"):
                callback.restore_ranking()
            if not callback.best_model_path:
                continue
            evaluation = cfg.get("evaluation", {})
            strategy = evaluation.get("checkpoint", "best")
            if strategy not in {"best", "weight_average", "logit_ensemble"}:
                raise ValueError(f"Unknown checkpoint strategy: {strategy}")
            if strategy != "best":
                from topobench.evaluator.checkpoint import (
                    PassThroughReadout,
                    PredictionEnsemble,
                    average_checkpoints,
                )

                k = int(evaluation.get("top_k", 5))
                ranked = sorted(
                    callback.best_k_models,
                    key=lambda path: float(callback.best_k_models[path]),
                    reverse=callback.mode == "max",
                )
                if k < 1 or len(ranked) < k:
                    raise ValueError(
                        f"Requested {k} checkpoints, but only {len(ranked)} are available; set callbacks.model_checkpoint.save_top_k and train enough epochs"
                    )
                paths = ranked[:k]
                selected_checkpoints = [
                    {
                        "path": str(path),
                        "validation_score": float(
                            callback.best_k_models[path]
                        ),
                    }
                    for path in paths
                ]
                if strategy == "weight_average":
                    checkpoint_model.load_state_dict(
                        average_checkpoints(paths), strict=True
                    )
                else:
                    ensemble = PredictionEnsemble(checkpoint_model, paths)
                    checkpoint_model.feature_encoder = torch.nn.Identity()
                    checkpoint_model.backbone = ensemble
                    checkpoint_model.readout = PassThroughReadout(
                        checkpoint_model.task_level
                    )
                checkpoint_model.to(device)
                break
            log.info(
                f"Loading best model from checkpoint at {callback.best_model_path}"
            )
            model_path = Path(callback.best_model_path)
            selected_checkpoints = [
                {
                    "path": str(model_path),
                    "validation_score": float(callback.best_model_score)
                    if callback.best_model_score is not None
                    else None,
                }
            ]
            ckpt = torch.load(
                model_path, map_location="cpu", weights_only=False
            )

            checkpoint_model.load_state_dict(ckpt["state_dict"], strict=True)
            checkpoint_model.to(device)
            break  # there is only one checkpoint callback

    walk_views = cfg.get("evaluation", {}).get("walk_views")
    if walk_views is not None:
        if int(walk_views) < 1:
            raise ValueError("evaluation.walk_views must be positive")
        for module in checkpoint_model.modules():
            if hasattr(module, "eval_views"):
                module.eval_views = int(walk_views)

    # New trainer to log final metrics on validation set
    # Because wandb displays validation metrics from the final, not the best epoch.
    checkpoint_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        num_sanity_val_steps=0,
        enable_progress_bar=cfg.trainer.get("enable_progress_bar", True),
        logger=False,
    )

    log.info("Re-testing best model checkpoint on validation set!")
    # The datamodule builds loaders inside the trainer, so multi-process
    # reruns shard each split exactly once instead of padding duplicates.
    results = checkpoint_trainer.validate(
        model=checkpoint_model, datamodule=datamodule
    )
    if results:
        logged = {}
        for k, v in results[0].items():
            suffix = k.split("/", 1)[1] if "/" in k else k
            logged[f"val_best_rerun/{suffix}"] = v
        log.info(logged)
        final_metrics.update(logged)
        log_late_metrics(logger, logged)

    log.info("Re-testing best model checkpoint on test set!")
    results = checkpoint_trainer.test(
        model=checkpoint_model, datamodule=datamodule
    )
    if results:
        logged = {}
        for k, v in results[0].items():
            suffix = k.split("/", 1)[1] if "/" in k else k
            logged[f"test_best_rerun/{suffix}"] = v
        log.info(logged)
        final_metrics.update(logged)
        log_late_metrics(logger, logged)
    if cfg.model.get("model_name") == "trawl" and rank_zero_only.rank == 0:
        report = {
            "checkpoint_strategy": cfg.get("evaluation", {}).get(
                "checkpoint", "best"
            ),
            "selected_checkpoints": selected_checkpoints,
            "checkpoint_count": len(selected_checkpoints),
            "walk_views": walk_views,
            "metrics": final_metrics,
        }
        destination = Path(cfg.paths.output_dir) / "trawl_evaluation.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    if (
        cfg.get("delete_checkpoint_after_test", False)
        and model_path
        and model_path.exists()
    ):
        log.info(f"Cleaning up: Deleting checkpoint at {model_path}")
        try:
            model_path.unlink()
        except Exception as e:
            log.warning(
                f"Failed to delete checkpoint at {model_path}. Error: {e}"
            )
    return final_metrics


def count_number_of_parameters(
    model: torch.nn.Module, only_trainable: bool = True
) -> int:
    """Count the number of trainable params.

    If all params, specify only_trainable = False.

    Ref:
        - https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=brando_miranda
        - https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    only_trainable : bool, optional
        If True, only count trainable parameters (default: True).

    Returns
    -------
    int
        The number of parameters.
    """
    if only_trainable:
        num_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:  # counts trainable and none-traibale
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    assert num_params > 0, f"Err: {num_params=}"
    return int(num_params)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
