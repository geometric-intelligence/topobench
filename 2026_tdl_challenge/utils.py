"""GraphUniverse Challenge: Evaluation pipeline for synthetic graph datasets."""

from __future__ import annotations

import copy
import io
import itertools
import json
import logging
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from matplotlib import colors as mcolors
from matplotlib import patheffects as mpe
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf, open_dict

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.run import run
from topobench.utils.config_resolvers import register_all_resolvers

# =============================================================================
# CONFIGURATION
# =============================================================================

STANDARD_GENERATION_PARAMETERS: dict[str, Any] = {
    "universe_parameters": {
        "K": 20,
        "feature_dim": 15,
        "center_variance": 0.2,
        "cluster_variance": 0.4,
        "edge_propensity_variance": 0.5,
        "seed": 42,
    },
    "family_parameters": {
        "n_graphs": 1000,
        "n_nodes_range": [50, 300],
        "n_communities_range": [5, 10],
        "homophily_range": [0.4, 0.6],
        "avg_degree_range": [1.0, 2.0],
        "degree_separation_range": [0.5, 1.0],
        "power_law_exponent_range": [1.5, 2.5],
        "seed": 42,
    },
}

HOMOPHILY_LEVELS: dict[str, list[float]] = {
    "h_lo": [0.0, 0.1],
    "h_mid": [0.4, 0.6],
    "h_hi": [0.9, 1.0],
}

AVG_DEGREE_LEVELS: dict[str, list[float]] = {
    "d_lo": [1.0, 2.5],
    "d_hi": [4.0, 5.0],
}

POWER_LAW_EXPONENT_LEVELS: dict[str, list[float]] = {
    "pl_lo": [1.5, 2.0],
    "pl_hi": [4.0, 5.0],
}

HOMOPHILY_AXIS_ORDER: tuple[str, ...] = ("h_lo", "h_mid", "h_hi")
AVG_DEGREE_AXIS_ORDER: tuple[str, ...] = ("d_lo", "d_hi")
POWER_LAW_AXIS_ORDER: tuple[str, ...] = ("pl_lo", "pl_hi")

CHALLENGE_TRAIN_SEEDS: tuple[int, ...] = (42, 43, 44)

DEFAULT_WANDB_PROJECT_CD = "challenge_community_detection"
DEFAULT_WANDB_PROJECT_TRI = "challenge_triangle_counting"

DEFAULT_EXPERIMENT_MODES: list[tuple[str, str, str, list[str]]] = [
    (
        "community_detection",
        DEFAULT_WANDB_PROJECT_CD,
        "dataset=graph/graphuniverse_inductive",
        [],
    ),
    (
        "triangle_counting",
        DEFAULT_WANDB_PROJECT_TRI,
        "dataset=graph/graphuniverse_inductive_triangle",
        [],
    ),
]

MAX_EPOCHS = 500

# Hydra overrides for `run_challenge_grid` only (repo `configs/` defaults stay unchanged).
CHECK_VAL_EVERY_N_EPOCH = 1
EARLY_STOPPING_PATIENCE = 10

CHALLENGE_GRID_HYDRA_OVERRIDES: list[str] = [
    f"trainer.check_val_every_n_epoch={CHECK_VAL_EVERY_N_EPOCH}",
    f"callbacks.early_stopping.patience={EARLY_STOPPING_PATIENCE}",
]

CHALLENGE_FEATURE_ENCODER_OUT_CHANNELS = 64

# =============================================================================
# GRID SETTINGS
# =============================================================================


@dataclass(frozen=True)
class GraphUniverseChallengeSetting:
    homophily_key: str
    avg_degree_key: str
    power_law_key: str
    generation_parameters: dict[str, Any]

    @property
    def run_slug(self) -> str:
        return f"{self.homophily_key}__{self.avg_degree_key}__{self.power_law_key}"


def build_generation_parameters(
    homophily_key: str,
    avg_degree_key: str,
    power_law_key: str,
) -> dict[str, Any]:
    if homophily_key not in HOMOPHILY_LEVELS:
        raise KeyError(homophily_key)
    if avg_degree_key not in AVG_DEGREE_LEVELS:
        raise KeyError(avg_degree_key)
    if power_law_key not in POWER_LAW_EXPONENT_LEVELS:
        raise KeyError(power_law_key)

    patch: dict[str, Any] = {
        "family_parameters": {
            "homophily_range": HOMOPHILY_LEVELS[homophily_key],
            "avg_degree_range": AVG_DEGREE_LEVELS[avg_degree_key],
            "power_law_exponent_range": POWER_LAW_EXPONENT_LEVELS[
                power_law_key
            ],
        },
    }
    return _deep_merge(STANDARD_GENERATION_PARAMETERS, patch)


def iter_challenge_settings() -> Iterator[GraphUniverseChallengeSetting]:
    for h_key, d_key, p_key in itertools.product(
        HOMOPHILY_LEVELS,
        AVG_DEGREE_LEVELS,
        POWER_LAW_EXPONENT_LEVELS,
    ):
        gp = build_generation_parameters(h_key, d_key, p_key)
        yield GraphUniverseChallengeSetting(h_key, d_key, p_key, gp)


# =============================================================================
# HYDRA UTILITIES
# =============================================================================


def generation_parameters_to_hydra_overrides(
    generation_parameters: dict[str, Any],
    *,
    prefix: str = "dataset.loader.parameters.generation_parameters",
) -> list[str]:
    lines: list[str] = []

    def _fmt(val: Any) -> str:
        if isinstance(val, list):
            return (
                "["
                + ",".join(
                    _fmt(x) if isinstance(x, list) else str(x) for x in val
                )
                + "]"
            )
        if isinstance(val, bool):
            return str(val).lower()
        return str(val)

    for k, v in generation_parameters["universe_parameters"].items():
        lines.append(f"{prefix}.universe_parameters.{k}={_fmt(v)}")
    for k, v in generation_parameters["family_parameters"].items():
        lines.append(f"{prefix}.family_parameters.{k}={_fmt(v)}")
    return lines


def challenge_setting_to_hydra_overrides(
    setting: GraphUniverseChallengeSetting,
) -> list[str]:
    return generation_parameters_to_hydra_overrides(
        setting.generation_parameters
    )


def apply_challenge_feature_encoder_out_channels(cfg: Any) -> None:
    """If ``model.feature_encoder.out_channels`` exists, force it for challenge runs (utils-only)."""
    if not OmegaConf.is_config(cfg):
        return
    model = cfg.get("model")
    if model is None or not OmegaConf.is_config(model):
        return
    fe = model.get("feature_encoder")
    if fe is None or not OmegaConf.is_config(fe):
        return
    if "out_channels" not in fe:
        return
    with open_dict(fe):
        fe.out_channels = CHALLENGE_FEATURE_ENCODER_OUT_CHANNELS


def resolve_project_root(here: Path | None = None) -> Path:
    _here = (here or Path.cwd()).resolve()
    root = _here if (_here / "configs" / "run.yaml").exists() else _here.parent
    if not (root / "configs" / "run.yaml").exists():
        raise FileNotFoundError(
            f"Could not find configs/run.yaml under {_here} or {_here.parent}. "
            "Run from the repo root or from challenge/."
        )
    return root


def ensure_repo_on_path(project_root: Path) -> None:
    s = str(project_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    os.environ["PROJECT_ROOT"] = s


# =============================================================================
# DATAMODULE CONSTRUCTION
# =============================================================================


def build_datamodule_for_setting(
    cfg: Any, setting: GraphUniverseChallengeSetting
) -> Any:
    cfg_eval = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    gp = setting.generation_parameters
    with open_dict(cfg_eval.dataset.loader.parameters.generation_parameters):
        cfg_eval.dataset.loader.parameters.generation_parameters.universe_parameters = OmegaConf.create(
            copy.deepcopy(gp["universe_parameters"])
        )
        cfg_eval.dataset.loader.parameters.generation_parameters.family_parameters = OmegaConf.create(
            copy.deepcopy(gp["family_parameters"])
        )
    dataset_loader = hydra.utils.instantiate(cfg_eval.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transform_config = cfg_eval.get("transforms", None)
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg_eval.dataset.split_params)
    )
    if cfg_eval.dataset.parameters.task_level not in ("node", "graph"):
        raise ValueError("Invalid task_level")
    return TBDataloader(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        **cfg_eval.dataset.get("dataloader_params", {}),
    )


# =============================================================================
# TRIANGLE COUNTING METRICS
# =============================================================================


def triangle_count_from_edge_index(data: Any) -> int:
    import networkx as nx

    ei = data.edge_index
    if ei is None or ei.numel() == 0:
        return 0
    u = ei[0].detach().cpu().numpy()
    v = ei[1].detach().cpu().numpy()
    if data.num_nodes is not None:
        n = int(data.num_nodes)
    else:
        n = int(ei.max().item()) + 1
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(u.shape[0]):
        a, b = int(u[i]), int(v[i])
        if a != b:
            g.add_edge(a, b)
    return int(sum(nx.triangles(g).values()) // 3)


def total_test_triangles_structural(datamodule: Any) -> int:
    ds = getattr(datamodule, "dataset_test", None)
    if ds is None or not hasattr(ds, "data_lst"):
        raise RuntimeError(
            "Expected ``datamodule.dataset_test`` with ``data_lst`` (DataloadDataset)."
        )
    return sum(triangle_count_from_edge_index(d) for d in ds.data_lst)


def compute_triangle_metrics(
    datamodule: Any, test_mse: float
) -> tuple[int, float | None]:
    total = total_test_triangles_structural(datamodule)
    mse_by_total: float | None = None
    if total > 0 and math.isfinite(test_mse):
        mse_by_total = float(test_mse) / float(total)
    return total, mse_by_total


# =============================================================================
# OOD EVALUATION
# =============================================================================


def _collect_ood_test_metrics(
    model: Any,
    cfg: Any,
    train_setting: GraphUniverseChallengeSetting,
    test_trainer: pl.Trainer,
    *,
    all_settings: list[GraphUniverseChallengeSetting],
    mode_name: str,
    quiet: bool,
) -> dict[str, dict[str, float]]:
    ood_out: dict[str, dict[str, float]] = {}
    for ood in all_settings:
        if ood.run_slug == train_setting.run_slug:
            continue
        ood_dm = build_datamodule_for_setting(cfg, ood)
        test_out = test_trainer.test(model, ood_dm)
        test_metrics = test_out[0] if test_out else {}
        acc = float(test_metrics.get("test/accuracy", float("nan")))
        mse = float(test_metrics.get("test/mse", float("nan")))
        row: dict[str, float] = {
            "test_best_rerun_accuracy": acc,
            "test_best_rerun_mse": mse,
        }
        if mode_name == "triangle_counting":
            tri_total, mse_by_tri = compute_triangle_metrics(ood_dm, mse)
            row["test_triangles_total_structural"] = float(tri_total)
            row["test_mse_by_total_triangles"] = (
                float(mse_by_tri) if mse_by_tri is not None else float("nan")
            )
        ood_out[ood.run_slug] = row
        if not quiet:
            print(
                f"    [OOD] train={train_setting.run_slug} eval={ood.run_slug} "
                f"acc={acc:.4g} mse={mse:.4g}",
                flush=True,
            )
    return ood_out


# =============================================================================
# TRAINING PIPELINE
# =============================================================================


@contextmanager
def _challenge_quiet(quiet: bool) -> Iterator[None]:
    if not quiet:
        yield
        return

    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    wandb_prev = os.environ.get("WANDB_SILENT")
    os.environ["WANDB_SILENT"] = "true"

    loggers: list[tuple[logging.Logger, int]] = []
    for name in (
        "",
        "lightning",
        "lightning.pytorch",
        "pytorch_lightning",
        "topobench",
        "wandb",
        "urllib3",
    ):
        lg = logging.getLogger(name)
        loggers.append((lg, lg.level))
        lg.setLevel(logging.ERROR)

    exc: BaseException | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sys.stdout, sys.stderr = buf_out, buf_err
                yield
            except BaseException as e:
                exc = e
                raise
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                if exc is not None:
                    o, e = buf_out.getvalue(), buf_err.getvalue()
                    if o.strip():
                        print(
                            "\n--- captured stdout (tail) ---\n",
                            o[-14_000:],
                            flush=True,
                        )
                    if e.strip():
                        print(
                            "\n--- captured stderr (tail) ---\n",
                            e[-140_000:],
                            file=sys.stderr,
                            flush=True,
                        )
    finally:
        for lg, lvl in loggers:
            lg.setLevel(lvl)
        if wandb_prev is None:
            os.environ.pop("WANDB_SILENT", None)
        else:
            os.environ["WANDB_SILENT"] = wandb_prev


def run_challenge_grid(
    *,
    project_root: Path | None = None,
    model_config: str = "graph/gin",
    wandb_project_cd: str = DEFAULT_WANDB_PROJECT_CD,
    wandb_project_tri: str = DEFAULT_WANDB_PROJECT_TRI,
    limit_runs: int | None = None,
    extra_overrides: list[str] | None = None,
    experiment_modes: list[tuple[str, str, str, list[str]]] | None = None,
    study_id: str | None = None,
    work_dir: Path | None = None,
    register_resolvers: bool = True,
    quiet: bool = False,
    train_seeds: tuple[int, ...] = CHALLENGE_TRAIN_SEEDS,
) -> tuple[list[dict[str, Any]], str]:
    """Run the full challenge grid; prepends ``CHALLENGE_GRID_HYDRA_OVERRIDES`` to Hydra overrides."""
    root = project_root or resolve_project_root()
    ensure_repo_on_path(root)
    if register_resolvers:
        register_all_resolvers()

    modes = experiment_modes or [
        (
            "community_detection",
            wandb_project_cd,
            "dataset=graph/graphuniverse_inductive",
            [],
        ),
        (
            "triangle_counting",
            wandb_project_tri,
            "dataset=graph/graphuniverse_inductive_triangle",
            [],
        ),
    ]
    extra = list(CHALLENGE_GRID_HYDRA_OVERRIDES) + list(extra_overrides or [])
    _work = (work_dir or Path.cwd()).resolve()
    sid = study_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results: list[dict[str, Any]] = []
    all_settings = list(iter_challenge_settings())

    n_settings = sum(1 for _ in iter_challenge_settings())
    if limit_runs is not None:
        n_settings = min(n_settings, limit_runs)
    total_runs = n_settings * len(modes) * len(train_seeds)
    run_ix = 0

    for mode_name, wandb_project, dataset_group, mode_overrides in modes:
        for idx, setting in enumerate(iter_challenge_settings()):
            if limit_runs is not None and idx >= limit_runs:
                break

            run_slug = setting.run_slug
            for train_seed in train_seeds:
                run_dir = (
                    root
                    / "logs"
                    / "train"
                    / "runs"
                    / f"notebook_gu_grid_{sid}__{mode_name}__{idx:02d}__{run_slug}__s{train_seed}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                wandb_name = pretty_wandb_run_name(
                    setting, model_config, train_seed
                )

                overrides: list[str] = [
                    dataset_group,
                    f"model={model_config}",
                    f"logger.wandb.project={wandb_project}",
                    f"+logger.wandb.name={_hydra_override_string_value(wandb_name)}",
                    f"paths.output_dir={run_dir.as_posix()}",
                    f"paths.work_dir={_work.as_posix()}",
                    f"tags=[gu_challenge,{mode_name},{run_slug},s{train_seed}]",
                    f"seed={train_seed}",
                    f"trainer.max_epochs={MAX_EPOCHS}",
                ]
                overrides.extend(mode_overrides)
                overrides.extend(challenge_setting_to_hydra_overrides(setting))
                overrides.extend(extra)

                GlobalHydra.instance().clear()
                with initialize_config_dir(
                    version_base="1.3", config_dir=str(root / "configs")
                ):
                    cfg = compose(config_name="run.yaml", overrides=overrides)

                apply_challenge_feature_encoder_out_channels(cfg)

                if OmegaConf.is_config(cfg) and cfg.get("trainer") is not None:
                    with open_dict(cfg.trainer):
                        cfg.trainer.enable_progress_bar = not quiet

                pl.seed_everything(cfg.seed, workers=True)
                torch.manual_seed(cfg.seed)
                np.random.seed(cfg.seed)
                random.seed(cfg.seed)

                run_ix += 1
                if quiet:
                    print(
                        f"[{run_ix}/{total_runs}] {mode_name} | {wandb_name}",
                        flush=True,
                    )
                else:
                    print(
                        f"\n=== [{mode_name}] [{idx + 1}] {wandb_name} ({run_slug}) seed={train_seed} | {run_dir} ===",
                        flush=True,
                    )

                tri_total: int | None = None
                mse_by_tri: float | None = None
                ood_test: dict[str, dict[str, float]] = {}
                with _challenge_quiet(quiet):
                    _metric_dict, object_dict = run(cfg)

                    model = object_dict["model"]
                    datamodule = object_dict["datamodule"]
                    test_trainer = pl.Trainer(
                        logger=False,
                        enable_progress_bar=not quiet,
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                    )
                    test_out = test_trainer.test(model, datamodule)
                    test_metrics = test_out[0] if test_out else {}
                    test_mse = float(
                        test_metrics.get("test/mse", float("nan"))
                    )

                    if mode_name == "triangle_counting":
                        tri_total, mse_by_tri = compute_triangle_metrics(
                            datamodule, test_mse
                        )
                        if not quiet:
                            print(
                                f"  [triangle_counting] structural test triangles={tri_total}; "
                                f"mse / triangles={mse_by_tri}",
                                flush=True,
                            )

                    ood_test = _collect_ood_test_metrics(
                        model,
                        cfg,
                        setting,
                        test_trainer,
                        all_settings=all_settings,
                        mode_name=mode_name,
                        quiet=quiet,
                    )

                wandb_cfg_metrics = _read_wandb_run_metrics_from_config_yaml(
                    run_dir
                )
                row: dict[str, Any] = {
                    "experiment": mode_name,
                    "wandb_project": wandb_project,
                    "wandb_run_name": wandb_name,
                    "train_seed": int(train_seed),
                    "homophily": setting.homophily_key,
                    "avg_degree": setting.avg_degree_key,
                    "power_law": setting.power_law_key,
                    "run_slug": run_slug,
                    "test_loss": float(
                        test_metrics.get("test/loss", float("nan"))
                    ),
                    "test_best_rerun_accuracy": float(
                        test_metrics.get("test/accuracy", float("nan"))
                    ),
                    "test_best_rerun_mse": test_mse,
                    "test_triangles_total_structural": float(tri_total)
                    if tri_total is not None
                    else float("nan"),
                    "test_mse_by_total_triangles": float(mse_by_tri)
                    if mse_by_tri is not None
                    else float("nan"),
                    "ood_test": ood_test,
                    "output_dir": str(run_dir),
                }
                if wandb_cfg_metrics:
                    row["wandb_config"] = wandb_cfg_metrics
                results.append(row)

    print(f"\nFinished {len(results)} run(s).")
    return results, sid


def check_challenge_grid(
    *,
    project_root: Path | None = None,
    model_config: str = "graph/gin",
    extra_overrides: list[str] | None = None,
    quiet: bool = True,
) -> None:
    """Quickly check that all configurations run correctly (1 epoch, 1 seed, no wandb)."""
    root = project_root or resolve_project_root()
    ensure_repo_on_path(root)
    if not GlobalHydra.instance().is_initialized():
        register_all_resolvers()

    modes = [
        ("community_detection", "dataset=graph/graphuniverse_inductive", []),
        (
            "triangle_counting",
            "dataset=graph/graphuniverse_inductive_triangle",
            [],
        ),
    ]

    all_settings = list(iter_challenge_settings())
    total_checks = len(modes) * len(all_settings)
    check_ix = 0

    if not quiet:
        print(f"Starting sanity check for {total_checks} configurations...")

    for mode_name, dataset_group, mode_overrides in modes:
        for setting in all_settings:
            check_ix += 1
            run_slug = setting.run_slug

            # Temporary directory for check logs
            run_dir = root / "logs" / "check" / f"{mode_name}_{run_slug}"
            run_dir.mkdir(parents=True, exist_ok=True)

            overrides: list[str] = [
                dataset_group,
                f"model={model_config}",
                "logger=csv",
                f"paths.output_dir={run_dir.as_posix()}",
                f"paths.work_dir={run_dir.as_posix()}",
                "seed=42",
                "trainer.max_epochs=1",
                "trainer.check_val_every_n_epoch=1",
            ]
            overrides.extend(mode_overrides)
            overrides.extend(challenge_setting_to_hydra_overrides(setting))
            if extra_overrides:
                overrides.extend(extra_overrides)

            GlobalHydra.instance().clear()
            with initialize_config_dir(
                version_base="1.3", config_dir=str(root / "configs")
            ):
                cfg = compose(config_name="run.yaml", overrides=overrides)

            apply_challenge_feature_encoder_out_channels(cfg)

            if OmegaConf.is_config(cfg) and cfg.get("trainer") is not None:
                with open_dict(cfg.trainer):
                    cfg.trainer.enable_progress_bar = False

            pl.seed_everything(cfg.seed, workers=True)

            if not quiet:
                print(
                    f"[{check_ix}/{total_checks}] Checking {mode_name} | {run_slug} ...",
                    end=" ",
                    flush=True,
                )

            with _challenge_quiet(True):
                try:
                    run(cfg)
                    if not quiet:
                        print("OK")
                except Exception as e:
                    if not quiet:
                        print("FAILED")
                    print(f"\n❌ CHECK FAILED: {mode_name} | {run_slug}")
                    print(f"Error: {e}")
                    raise e

    if not quiet:
        print(
            f"\n✅ All {total_checks} configurations passed the sanity check."
        )


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================


def _format_range_pair(rng: list[float]) -> str:
    a, b = float(rng[0]), float(rng[1])
    return f"{a:g}-{b:g}"


def _short_axis_label(levels: dict[str, list[float]], key: str) -> str:
    return _format_range_pair(levels[key])


def _model_slug(model_cfg: str) -> str:
    return model_cfg.strip().split("/")[-1].replace("-", "_")


def _hydra_override_string_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def pretty_wandb_run_name(
    setting: GraphUniverseChallengeSetting,
    model_config: str,
    train_seed: int | None = None,
) -> str:
    m = _model_slug(model_config)
    h = _short_axis_label(HOMOPHILY_LEVELS, setting.homophily_key)
    d = _short_axis_label(AVG_DEGREE_LEVELS, setting.avg_degree_key)
    p = _short_axis_label(POWER_LAW_EXPONENT_LEVELS, setting.power_law_key)
    base = f"{m}_hom_{h}__deg_{d}__gamma_{p}"
    if train_seed is None:
        return base
    return f"{base}__s{train_seed}"


def apply_publication_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "font.family": "sans-serif",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# HEATMAP GENERATION
# =============================================================================


def _mean_std_matrix_for_power_law_panel(
    results: list[dict[str, Any]],
    *,
    experiment: str,
    value_key: str,
    power_law_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    row_i = {k: i for i, k in enumerate(AVG_DEGREE_AXIS_ORDER)}
    col_i = {k: i for i, k in enumerate(HOMOPHILY_AXIS_ORDER)}
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        if (
            r.get("experiment") != experiment
            or r.get("power_law") != power_law_key
        ):
            continue
        v = r.get(value_key)
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            continue
        try:
            buckets[(r["avg_degree"], r["homophily"])].append(float(v))
        except (KeyError, TypeError):
            continue
    mean_m = np.full(
        (len(AVG_DEGREE_AXIS_ORDER), len(HOMOPHILY_AXIS_ORDER)),
        np.nan,
        dtype=float,
    )
    std_m = np.full_like(mean_m, np.nan)
    for (ad, hk), vals in buckets.items():
        ii, jj = row_i[ad], col_i[hk]
        a = np.asarray(vals, dtype=float)
        mean_m[ii, jj] = float(np.nanmean(a))
        std_m[ii, jj] = float(np.nanstd(a)) if a.size > 1 else 0.0
    return mean_m, std_m


def _resolve_colormap(cmap: str | mcolors.Colormap) -> mcolors.Colormap:
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def _relative_luminance_srgb(rgb: tuple[float, float, float]) -> float:
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _annotate_heatmap_cell_mean_std(
    ax: Axes,
    i: int,
    j: int,
    mean: float,
    std: float,
    *,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    decimals_mean: int,
    decimals_std: int,
) -> None:
    rgb = mcolors.to_rgb(cmap(norm(mean)))
    lum = _relative_luminance_srgb(rgb)
    txt_color = "#f8f8f8" if lum < 0.5 else "#101010"
    halo = "#101010" if lum < 0.5 else "#f5f5f5"
    std_disp = std if math.isfinite(std) else float("nan")
    line2 = (
        f"±{std_disp:.{decimals_std}f}" if math.isfinite(std_disp) else "±nan"
    )
    txt = f"{mean:.{decimals_mean}f}\n{line2}"
    ax.text(
        j,
        i,
        txt,
        ha="center",
        va="center",
        color=txt_color,
        fontsize=10,
        fontweight="600",
        path_effects=[
            mpe.withStroke(linewidth=2.0, foreground=halo, alpha=0.85)
        ],
    )


def plot_challenge_heatmap_figure(
    results: list[dict[str, Any]],
    *,
    experiment: str,
    value_key: str,
    suptitle: str,
    cbar_label: str,
    cmap: str | mcolors.Colormap = "RdBu_r",
    annotate_decimals: int = 3,
) -> Figure:
    apply_publication_matplotlib_style()
    cmap_resolved = _resolve_colormap(cmap)

    mats_mean: list[np.ndarray] = []
    mats_std: list[np.ndarray] = []
    for pk in POWER_LAW_AXIS_ORDER:
        mu, sig = _mean_std_matrix_for_power_law_panel(
            results,
            experiment=experiment,
            value_key=value_key,
            power_law_key=pk,
        )
        mats_mean.append(mu)
        mats_std.append(sig)
    stacked = np.concatenate([m.ravel() for m in mats_mean])
    finite = stacked[np.isfinite(stacked)]
    if finite.size:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(12.0, 7.2), constrained_layout=False
    )
    fig.subplots_adjust(
        left=0.10, right=0.908, top=0.86, bottom=0.08, hspace=0.42
    )
    fig.suptitle(suptitle, fontsize=16, fontweight="600", y=0.993)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    dec_std = max(2, annotate_decimals - 1)

    for ax, pl_key, mat, std_mat in zip(
        axes, POWER_LAW_AXIS_ORDER, mats_mean, mats_std, strict=True
    ):
        masked = np.ma.masked_where(~np.isfinite(mat), mat)
        ax.imshow(
            masked,
            cmap=cmap_resolved,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(HOMOPHILY_AXIS_ORDER)))
        ax.set_yticks(np.arange(len(AVG_DEGREE_AXIS_ORDER)))
        ax.set_xticklabels(
            [
                f"Homophily\n{_short_axis_label(HOMOPHILY_LEVELS, k)}"
                for k in HOMOPHILY_AXIS_ORDER
            ],
            fontsize=11,
        )
        ax.set_yticklabels(
            [
                f"Avg degree\n{_short_axis_label(AVG_DEGREE_LEVELS, k)}"
                for k in AVG_DEGREE_AXIS_ORDER
            ],
            fontsize=11,
        )
        pl_human = _short_axis_label(POWER_LAW_EXPONENT_LEVELS, pl_key)
        ax.set_title(
            rf"Power-law exponent $\gamma$: {pl_human}", fontsize=13, pad=10
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    sd = (
                        float(std_mat[i, j])
                        if np.isfinite(std_mat[i, j])
                        else float("nan")
                    )
                    _annotate_heatmap_cell_mean_std(
                        ax,
                        i,
                        j,
                        float(val),
                        sd,
                        cmap=cmap_resolved,
                        norm=norm,
                        decimals_mean=annotate_decimals,
                        decimals_std=dec_std,
                    )

    cbar_ax = fig.add_axes([0.915, 0.15, 0.026, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    return fig


# =============================================================================
# OOD VISUALIZATION
# =============================================================================

_OOD_HOMOPHILY_FILE_TAG: dict[str, str] = {
    "h_lo": "low",
    "h_mid": "mid",
    "h_hi": "high",
}


def _ood_group_by_train_slug(
    results: list[dict[str, Any]], experiment: str
) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.get("experiment") != experiment:
            continue
        slug = r.get("run_slug")
        if isinstance(slug, str):
            g[slug].append(r)
    for rows in g.values():
        rows.sort(key=lambda x: int(x.get("train_seed", 0)))
    return dict(g)


def _ood_train_slugs_for_homophily(homophily_key: str) -> list[str]:
    return [
        f"{homophily_key}__{dk}__{pk}"
        for dk in AVG_DEGREE_AXIS_ORDER
        for pk in POWER_LAW_AXIS_ORDER
    ]


def _ood_mean_std_ood_minus_in_matrix(
    rows_by_train_slug: dict[str, list[dict[str, Any]]],
    train_slug: str,
    *,
    power_law_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    row_i = {k: i for i, k in enumerate(AVG_DEGREE_AXIS_ORDER)}
    col_i = {k: i for i, k in enumerate(HOMOPHILY_AXIS_ORDER)}
    mean_m = np.full(
        (len(AVG_DEGREE_AXIS_ORDER), len(HOMOPHILY_AXIS_ORDER)),
        np.nan,
        dtype=float,
    )
    std_m = np.full_like(mean_m, np.nan)
    rows = rows_by_train_slug.get(train_slug, [])

    def _finite_float(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            return v if math.isfinite(v) else None
        return None

    for hk in HOMOPHILY_AXIS_ORDER:
        for ad in AVG_DEGREE_AXIS_ORDER:
            eval_slug = f"{hk}__{ad}__{power_law_key}"
            per_seed: list[float] = []
            for r in rows:
                in_v = _finite_float(r.get(metric_key))
                if in_v is None:
                    continue
                if eval_slug == train_slug:
                    per_seed.append(0.0)
                    continue
                ood_block = (r.get("ood_test") or {}).get(eval_slug)
                if not isinstance(ood_block, dict):
                    continue
                ood_v = _finite_float(ood_block.get(metric_key))
                if ood_v is None:
                    continue
                per_seed.append(ood_v - in_v)
            if not per_seed:
                continue
            ii, jj = row_i[ad], col_i[hk]
            a = np.asarray(per_seed, dtype=float)
            mean_m[ii, jj] = float(np.mean(a))
            std_m[ii, jj] = float(np.std(a)) if a.size > 1 else 0.0
    return mean_m, std_m


def _ood_identity_row_col_for_panel(
    train_slug: str, pl_key: str
) -> tuple[int, int] | None:
    parts = train_slug.split("__")
    if len(parts) != 3:
        return None
    h_t, d_t, p_t = parts[0], parts[1], parts[2]
    if pl_key != p_t:
        return None
    row_i = {k: i for i, k in enumerate(AVG_DEGREE_AXIS_ORDER)}
    col_i = {k: i for i, k in enumerate(HOMOPHILY_AXIS_ORDER)}
    if d_t not in row_i or h_t not in col_i:
        return None
    return row_i[d_t], col_i[h_t]


def _ood_mark_baseline_distribution_cell(ax: Axes, i: int, j: int) -> None:
    ax.add_patch(
        Rectangle(
            (j - 0.5, i - 0.5),
            1.0,
            1.0,
            facecolor="black",
            edgecolor="black",
            linewidth=1.5,
            zorder=25,
        )
    )
    pad = 0.42
    kw: dict[str, Any] = dict(
        color="white", linewidth=4.5, zorder=26, solid_capstyle="round"
    )
    ax.plot([j - pad, j + pad], [i - pad, i + pad], **kw)
    ax.plot([j - pad, j + pad], [i + pad, i - pad], **kw)


def _ood_annotate_cell_mean_std(
    ax: Axes,
    i: int,
    j: int,
    mean: float,
    std: float,
    *,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    decimals_mean: int,
    decimals_std: int,
    fontsize: int,
    halo_lw: float = 1.2,
) -> None:
    rgb = mcolors.to_rgb(cmap(norm(mean)))
    lum = _relative_luminance_srgb(rgb)
    txt_color = "#f8f8f8" if lum < 0.5 else "#101010"
    halo = "#101010" if lum < 0.5 else "#f5f5f5"
    std_disp = std if math.isfinite(std) else float("nan")
    line2 = (
        f"±{std_disp:.{decimals_std}f}" if math.isfinite(std_disp) else "±nan"
    )
    txt = f"{mean:.{decimals_mean}f}\n{line2}"
    ax.text(
        j,
        i,
        txt,
        ha="center",
        va="center",
        color=txt_color,
        fontsize=fontsize,
        fontweight="600",
        path_effects=[
            mpe.withStroke(linewidth=halo_lw, foreground=halo, alpha=0.85)
        ],
    )


def _ood_add_training_vs_ood_legend_strip(
    fig: Figure, *, text_fontsize: int, detail_fontsize: int
) -> None:
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    box_w, box_h = 0.84, 0.076
    x0 = (1.0 - box_w) / 2.0 + 0.058
    y0 = 0.818

    s_w = 0.038
    s_h = s_w * (fw / fh)
    if s_h > box_h * 0.92:
        s_h = box_h * 0.92
        s_w = s_h * (fh / fw)
    x_icon = x0 + 0.02
    y_icon = y0 + max(0.0, (box_h - s_h) / 2.0)
    icon_ax = fig.add_axes([x_icon, y_icon, s_w, s_h])
    icon_ax.set_xlim(0, 1)
    icon_ax.set_ylim(0, 1)
    icon_ax.set_aspect("equal", adjustable="box")
    icon_ax.axis("off")
    pad = 0.12
    icon_ax.add_patch(
        Rectangle(
            (pad, pad),
            1 - 2 * pad,
            1 - 2 * pad,
            facecolor="black",
            edgecolor="white",
            linewidth=2.0,
            clip_on=False,
        )
    )
    inset = 0.22
    cross_kw: dict[str, Any] = dict(
        color="white",
        linewidth=4.2,
        solid_capstyle="projecting",
        clip_on=False,
        zorder=5,
    )
    icon_ax.plot([inset, 1 - inset], [inset, 1 - inset], **cross_kw)
    icon_ax.plot([inset, 1 - inset], [1 - inset, inset], **cross_kw)

    tx0 = x_icon + s_w + 0.018
    tw = x0 + box_w - tx0 - 0.02
    text_ax = fig.add_axes([tx0, y0, tw, box_h])
    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(0, 1)
    text_ax.axis("off")
    cy1, cy2 = 0.62, 0.22
    text_ax.text(
        0.0,
        cy1,
        "Training distribution",
        fontsize=text_fontsize,
        fontweight="700",
        va="center",
        ha="left",
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.31,
        cy1,
        "·",
        fontsize=text_fontsize * 1.35,
        va="center",
        ha="center",
        color="#555",
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.35,
        cy1,
        "All other cells: out-of-distribution",
        fontsize=detail_fontsize + 2,
        fontweight="600",
        va="center",
        ha="left",
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.0,
        cy2,
        "Δ = OOD test − in-distribution test; cell text = mean ± std over training seeds",
        fontsize=detail_fontsize,
        va="center",
        ha="left",
        transform=text_ax.transAxes,
    )


def plot_ood_delta_by_homophily_figure(
    results: list[dict[str, Any]],
    homophily_key: str,
    *,
    experiment: str,
    metric_key: str,
    slug: str,
    cbar_label: str,
    cmap: str | mcolors.Colormap = "RdBu_r",
    annotate_decimals_mean: int = 3,
    annotate_decimals_std: int = 2,
    cell_fontsize: int = 17,
    tick_fontsize: int = 22,
    axis_title_fontsize: int = 24,
    gamma_title_fontsize: int = 26,
    suptitle_fontsize: int = 23,
    cbar_label_fontsize: int = 25,
    cbar_tick_fontsize: int = 21,
    legend_title_fontsize: int = 24,
    legend_detail_fontsize: int = 20,
) -> Figure:
    apply_publication_matplotlib_style()
    cmap_resolved = _resolve_colormap(cmap)
    rows_by = _ood_group_by_train_slug(results, experiment)
    train_slugs = _ood_train_slugs_for_homophily(homophily_key)

    precomputed: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    all_means: list[float] = []
    for train_slug in train_slugs:
        panels: list[tuple[str, np.ndarray, np.ndarray]] = []
        for pk in POWER_LAW_AXIS_ORDER:
            mu, sig = _ood_mean_std_ood_minus_in_matrix(
                rows_by, train_slug, power_law_key=pk, metric_key=metric_key
            )
            panels.append((pk, mu, sig))
            id_ij = _ood_identity_row_col_for_panel(train_slug, pk)
            for i in range(mu.shape[0]):
                for j in range(mu.shape[1]):
                    if id_ij is not None and (i, j) == id_ij:
                        continue
                    v = mu[i, j]
                    if np.isfinite(v):
                        all_means.append(float(v))
        precomputed[train_slug] = panels

    if all_means:
        lim = max(abs(x) for x in all_means)
        lim = lim if lim > 1e-12 else 1e-6
    else:
        lim = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    tag = _OOD_HOMOPHILY_FILE_TAG.get(homophily_key, homophily_key)
    h_label = _short_axis_label(HOMOPHILY_LEVELS, homophily_key)
    fig = plt.figure(figsize=(24, 22))
    fig.subplots_adjust(
        left=0.084,
        right=0.898,
        top=0.778,
        bottom=0.065,
        hspace=0.26,
        wspace=0.0,
    )
    exp_title = experiment.replace("_", " ").title()
    fig.suptitle(
        f"{tag.capitalize()} homophily ({h_label}) — {exp_title}: OOD − in-distribution — {slug}",
        fontsize=suptitle_fontsize,
        fontweight="600",
        x=0.53,
        y=0.906,
        ha="center",
    )
    _ood_add_training_vs_ood_legend_strip(
        fig,
        text_fontsize=legend_title_fontsize,
        detail_fontsize=legend_detail_fontsize,
    )

    outer = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.0)

    for idx, train_slug in enumerate(train_slugs):
        rr, cc = divmod(idx, 2)
        inner = outer[rr, cc].subgridspec(2, 1, hspace=0.12)
        for pi, (pl_key, mat, std_mat) in enumerate(precomputed[train_slug]):
            ax = fig.add_subplot(inner[pi, 0])
            disp = np.array(mat, dtype=float, copy=True)
            id_ij = _ood_identity_row_col_for_panel(train_slug, pl_key)
            if id_ij is not None:
                ii, jj = id_ij
                disp[ii, jj] = float("nan")

            masked = np.ma.masked_where(~np.isfinite(disp), disp)
            ax.imshow(
                masked,
                cmap=cmap_resolved,
                norm=norm,
                aspect="equal",
                interpolation="nearest",
            )
            if id_ij is not None:
                _ood_mark_baseline_distribution_cell(ax, id_ij[0], id_ij[1])

            ax.set_xticks(np.arange(len(HOMOPHILY_AXIS_ORDER)))
            ax.set_yticks(np.arange(len(AVG_DEGREE_AXIS_ORDER)))
            show_x = pi == 1
            show_y = cc == 0
            if show_x:
                ax.set_xticklabels(
                    [
                        _short_axis_label(HOMOPHILY_LEVELS, k)
                        for k in HOMOPHILY_AXIS_ORDER
                    ],
                    fontsize=tick_fontsize,
                    rotation=25,
                    ha="right",
                )
                ax.tick_params(
                    axis="x",
                    which="major",
                    labelsize=tick_fontsize,
                    length=7,
                    width=1.3,
                )
            else:
                ax.set_xticklabels([])
            if show_y:
                ax.set_yticklabels(
                    [
                        _short_axis_label(AVG_DEGREE_LEVELS, k)
                        for k in AVG_DEGREE_AXIS_ORDER
                    ],
                    fontsize=tick_fontsize,
                )
                ax.tick_params(
                    axis="y",
                    which="major",
                    labelsize=tick_fontsize,
                    length=7,
                    width=1.3,
                )
            else:
                ax.set_yticklabels([])

            if show_x and rr == 1:
                ax.set_xlabel(
                    "Homophily",
                    fontsize=axis_title_fontsize,
                    fontweight="700",
                    labelpad=14,
                )
            if show_y and pi == 1:
                ax.set_ylabel(
                    "Avg degree",
                    fontsize=axis_title_fontsize,
                    fontweight="700",
                    labelpad=10,
                )
                ax.yaxis.set_label_coords(-0.16, 1.05, transform=ax.transAxes)

            pl_human = _short_axis_label(POWER_LAW_EXPONENT_LEVELS, pl_key)
            gpad = 8 if pi == 0 else 6
            ax.set_title(
                rf"Power-law $\gamma$: {pl_human}",
                fontsize=gamma_title_fontsize,
                fontweight="700",
                pad=gpad,
            )

            dec_std = max(2, annotate_decimals_mean - 1)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if id_ij is not None and (i, j) == id_ij:
                        continue
                    val = mat[i, j]
                    if np.isfinite(val):
                        sd = (
                            float(std_mat[i, j])
                            if np.isfinite(std_mat[i, j])
                            else float("nan")
                        )
                        _ood_annotate_cell_mean_std(
                            ax,
                            i,
                            j,
                            float(val),
                            sd,
                            cmap=cmap_resolved,
                            norm=norm,
                            decimals_mean=annotate_decimals_mean,
                            decimals_std=dec_std,
                            fontsize=cell_fontsize,
                            halo_lw=4.0,
                        )

    cbar_ax = fig.add_axes([0.912, 0.072, 0.044, 0.69])
    sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=cbar_label_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize, width=1.6, length=9)
    return fig


# =============================================================================
# RESULTS EXPORT
# =============================================================================

_WANDB_RUN_CONFIG_KEYS: tuple[str, ...] = (
    "AvgTime/train_epoch_mean",
    "AvgTime/train_epoch_std",
    "model/params/total",
    "model/params/trainable",
    "model/params/non_trainable",
)


def _find_wandb_run_config_yaml(output_dir: Path) -> Path | None:
    wandb_dir = output_dir / "wandb"
    if not wandb_dir.is_dir():
        return None
    candidates = list(wandb_dir.glob("run-*/files/config.yaml"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_wandb_run_metrics_from_config_yaml(
    output_dir: Path,
) -> dict[str, Any]:
    """Read selected flat keys from wandb's exported ``config.yaml`` under ``output_dir``."""
    path = _find_wandb_run_config_yaml(output_dir)
    if path is None:
        return {}
    try:
        cfg = OmegaConf.load(str(path))
        data = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _WANDB_RUN_CONFIG_KEYS:
        node = data.get(key)
        if node is None:
            continue
        if isinstance(node, dict) and "value" in node:
            out[key] = node["value"]
        else:
            out[key] = node
    return out


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _has_finite_metric(
    results: list[dict[str, Any]], *, experiment: str, value_key: str
) -> bool:
    for r in results:
        if r.get("experiment") != experiment:
            continue
        v = r.get(value_key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return True
    return False


def _has_ood_eval_data(
    results: list[dict[str, Any]], *, experiment: str, value_key: str
) -> bool:
    if not _has_finite_metric(
        results, experiment=experiment, value_key=value_key
    ):
        return False
    for r in results:
        if r.get("experiment") != experiment:
            continue
        ot = r.get("ood_test")
        if isinstance(ot, dict) and len(ot) > 0:
            return True
    return False


def write_challenge_figure_outputs(
    results: list[dict[str, Any]],
    *,
    out_dir: Path,
    model_config: str = "graph/gin",
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(model_config)
    paths: dict[str, Any] = {}
    apply_publication_matplotlib_style()

    if _has_finite_metric(
        results,
        experiment="community_detection",
        value_key="test_best_rerun_accuracy",
    ):
        fig_cd = plot_challenge_heatmap_figure(
            results,
            experiment="community_detection",
            value_key="test_best_rerun_accuracy",
            suptitle=f"Community detection — test accuracy mean ± std over seeds (best ckpt)\n{slug}",
            cbar_label="Accuracy",
            cmap="RdBu_r",
            annotate_decimals=3,
        )
        p_cd = out_dir / "heatmap_community_detection_accuracy.png"
        fig_cd.savefig(p_cd, bbox_inches="tight")
        plt.close(fig_cd)
        paths["cd_png"] = p_cd

    if _has_finite_metric(
        results,
        experiment="triangle_counting",
        value_key="test_mse_by_total_triangles",
    ):
        fig_tri = plot_challenge_heatmap_figure(
            results,
            experiment="triangle_counting",
            value_key="test_mse_by_total_triangles",
            suptitle=f"Triangle counting — MSE/triangles mean ± std over seeds (best ckpt)\n{slug}",
            cbar_label=r"MSE / $\sum$ triangles (test)",
            cmap="RdBu_r",
            annotate_decimals=4,
        )
        p_tri = out_dir / "heatmap_triangle_mse_over_triangles.png"
        fig_tri.savefig(p_tri, bbox_inches="tight")
        plt.close(fig_tri)
        paths["tri_png"] = p_tri

    ood_dir = out_dir / "OOD"
    if _has_ood_eval_data(
        results,
        experiment="community_detection",
        value_key="test_best_rerun_accuracy",
    ):
        ood_dir.mkdir(parents=True, exist_ok=True)
        for hk in HOMOPHILY_AXIS_ORDER:
            fig = plot_ood_delta_by_homophily_figure(
                results,
                hk,
                experiment="community_detection",
                metric_key="test_best_rerun_accuracy",
                slug=slug,
                cbar_label=r"$\Delta$ accuracy (OOD − ID)",
                cmap="RdBu_r",
                annotate_decimals_mean=3,
                annotate_decimals_std=2,
            )
            tag = _OOD_HOMOPHILY_FILE_TAG[hk]
            stem = f"OOD_{tag}_homophily__community_detection"
            dest = ood_dir / f"{stem}.png"
            fig.savefig(dest, bbox_inches="tight")
            plt.close(fig)
        paths["ood_community_dir"] = ood_dir

    if _has_ood_eval_data(
        results,
        experiment="triangle_counting",
        value_key="test_mse_by_total_triangles",
    ):
        ood_dir.mkdir(parents=True, exist_ok=True)
        for hk in HOMOPHILY_AXIS_ORDER:
            fig = plot_ood_delta_by_homophily_figure(
                results,
                hk,
                experiment="triangle_counting",
                metric_key="test_mse_by_total_triangles",
                slug=slug,
                cbar_label=r"$\Delta$ MSE / triangles (OOD − ID)",
                cmap="RdBu_r",
                annotate_decimals_mean=4,
                annotate_decimals_std=3,
                cell_fontsize=15,
                tick_fontsize=20,
                axis_title_fontsize=22,
                gamma_title_fontsize=24,
                suptitle_fontsize=21,
                cbar_label_fontsize=23,
                cbar_tick_fontsize=19,
                legend_title_fontsize=22,
                legend_detail_fontsize=18,
            )
            tag = _OOD_HOMOPHILY_FILE_TAG[hk]
            stem = f"OOD_{tag}_homophily__triangle_counting"
            dest = ood_dir / f"{stem}.png"
            fig.savefig(dest, bbox_inches="tight")
            plt.close(fig)
        paths["ood_triangle_dir"] = ood_dir

    return paths


def save_challenge_artifacts(
    results: list[dict[str, Any]],
    *,
    out_dir: Path | None = None,
    model_config: str = "graph/gin",
    study_id: str | None = None,
) -> dict[str, Any]:
    sid = study_id or datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    base = out_dir or (Path(__file__).resolve().parent / "outputs" / sid)
    base.mkdir(parents=True, exist_ok=True)

    seeds_seen = sorted(
        {
            int(r["train_seed"])
            for r in results
            if r.get("train_seed") is not None
        }
    )
    meta = {
        "study_id": sid,
        "model_config": model_config,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_runs": len(results),
        "train_seeds": seeds_seen or list(CHALLENGE_TRAIN_SEEDS),
        "heatmap_note": "Cells show mean ± std over train_seeds (in-distribution test).",
    }
    payload = {"metadata": meta, "results": results}
    json_path = base / "results.json"
    json_path.write_text(
        json.dumps(_json_sanitize(payload), indent=2), encoding="utf-8"
    )

    out_paths: dict[str, Any] = {"dir": base, "json": json_path}
    fig_paths = write_challenge_figure_outputs(
        results, out_dir=base, model_config=model_config
    )
    out_paths.update(fig_paths)

    print(f"Saved JSON and figures under: {base}")
    return out_paths


# =============================================================================
# HELPER UTILITIES
# =============================================================================


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out
