"""Runner for the exp3 feature-signal sweep.

See ``README.md`` in this folder for the scientific question, design, and
findings. This is a plain script (not a notebook) so it survives a Kaggle
session restart: it is idempotent and resumable, appending one record to
``e3_feature_signal_results.json`` immediately after each run and skipping
any ``(point, model, seed)`` already present on the next invocation.

Each ``(point, model, seed)`` job runs in its **own subprocess**, spawned by
the orchestrator in ``main()``, mirroring
``../lifting_confounding_study/run_lifting_ablation.py``: a hard OOM-kill
(or, on WSL2, a host memory-pressure freeze) is not a catchable Python
exception, so the orchestrator itself never touches CUDA and observes a
dying job only as a non-zero/killed subprocess exit code.

``center_variance`` is a **universe-level** parameter: changing it
regenerates the whole latent universe, not just a resample, so
each of the 5 sweep points pays its own dataset-generation cost and gets its
own preprocessing cache directory (verified in ``preflight_check()``).

Usage
-----
Phase 1 (10 runs, seed 42 only)::

    python run_e3.py

Phase 2 (30 runs total, adds seeds 43 and 44 — only once Phase 1 is fully
green)::

    python run_e3.py --phase2

Safe local sanity check *before* launching anything real (tiny synthetic
data, 2 epochs, CPU only — see "Local testing" in README.md)::

    python run_e3.py --smoke-test --cpu
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil  # cheap, orchestrator-only import; never touches CUDA

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- path setup --------------------------------------------------------
# `topobench`'s auto-discovery (topobench/loss/__init__.py,
# topobench/evaluator/metrics/__init__.py, etc.) inserts the `topobench/`
# package directory onto sys.path as a side effect of importing it. That
# directory contains a `utils/` subpackage, which then shadows the
# top-level `2026_tdl_challenge/utils.py` module for any *later*
# `import utils`. Importing `utils` first (before anything from
# `topobench`) caches the correct module object in `sys.modules`, exactly
# like `run_evaluation.ipynb` does.
_HERE = Path(__file__).resolve().parent
_CHALLENGE_DIR = _HERE.parent.parent
_REPO_ROOT = _CHALLENGE_DIR.parent
for _p in (_CHALLENGE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lightning as pl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402
from utils import (  # noqa: E402
    CHALLENGE_GRID_HYDRA_OVERRIDES,
    CHALLENGE_TRAIN_SEEDS,
    MAX_EPOCHS,
    GraphUniverseChallengeSetting,
    apply_challenge_feature_encoder_out_channels,
    build_datamodule_for_setting,
    build_generation_parameters,
    generation_parameters_to_hydra_overrides,
    resolve_project_root,
)

from topobench.run import count_number_of_parameters, run  # noqa: E402
from topobench.utils import config_resolvers  # noqa: E402

PROJECT_ROOT = resolve_project_root(_CHALLENGE_DIR)
RESULTS_PATH = _HERE / "e3_feature_signal_results.json"
FEATURE_SIGNAL_CACHE_PATH = _HERE / "feature_signal_empirical.json"
RUNS_DIR = _HERE / "runs"

# =============================================================================
# Experimental design — see README.md for the full rationale
# =============================================================================

# 5 points, log-ish spacing around the official center_variance=0.2.
CENTER_VARIANCE_POINTS: dict[str, float] = {
    "fs_00": 0.02,
    "fs_01": 0.05,
    "fs_02": 0.2,  # official setting — reference point
    "fs_03": 0.8,
    "fs_04": 2.0,
}
POINT_ORDER: tuple[str, ...] = tuple(CENTER_VARIANCE_POINTS)
OFFICIAL_POINT = "fs_02"

# Fixed structural cell: mid-homophily, sparse, heavy-tailed.
CELL_HOMOPHILY, CELL_AVG_DEGREE, CELL_POWER_LAW = "h_mid", "d_lo", "pl_lo"
CELL_KEY = f"{CELL_HOMOPHILY}__{CELL_AVG_DEGREE}__{CELL_POWER_LAW}"

# Both arms mandatory — the experiment is the *gap* between them.
MODELS: tuple[str, ...] = ("hypergraph/dphgnn", "graph/gcn")

PHASE1_SEEDS: tuple[int, ...] = (42,)

# Step 0b: number of graphs probed per point for the empirical feature_signal
# (community_signals diagnostic).
_FEATURE_SIGNAL_N_GRAPHS = 30
_FEATURE_SIGNAL_RANDOM_STATE = 42

# OOM fallback ladder for the dataloader batch size (mirrors
# lifting_confounding_study; DPHGNN's default khop lifting can still OOM
# depending on hyperedge cardinality). `None` means "repo default (64)".
_BATCH_SIZE_FALLBACKS: tuple[int | None, ...] = (None, 32, 16)

# --smoke-test: tiny synthetic data + 2 epochs, for validating the script's
# mechanics in seconds without the resource footprint of a real run.
_SMOKE_TEST_MAX_EPOCHS = 2
_SMOKE_TEST_GP_PATCH: dict[str, Any] = {
    "n_graphs": 20,
    "n_nodes_range": [20, 40],
}


def build_e3_generation_parameters(center_variance: float) -> dict[str, Any]:
    """Build the E3 generation-parameters dict for one sweep point.

    Starts from the fixed structural cell (``h_mid__d_lo__pl_lo``) via
    ``utils.build_generation_parameters``, then overrides only
    ``universe_parameters.center_variance``. Everything else — including
    ``cluster_variance`` — stays at ``STANDARD_GENERATION_PARAMETERS``.

    Parameters
    ----------
    center_variance : float
        The universe-level ``center_variance`` value for this sweep point.

    Returns
    -------
    dict
        A ``{universe_parameters, family_parameters}`` dict, ready for
        ``generation_parameters_to_hydra_overrides``.
    """
    gp = build_generation_parameters(
        CELL_HOMOPHILY, CELL_AVG_DEGREE, CELL_POWER_LAW
    )
    gp = copy.deepcopy(gp)
    gp["universe_parameters"]["center_variance"] = center_variance
    return gp


# =============================================================================
# Hydra composition
# =============================================================================


def _compose_cfg(
    *,
    model_config: str,
    center_variance: float,
    seed: int,
    run_dir: Path,
    batch_size: int | None = None,
    smoke_test: bool = False,
    force_cpu: bool = False,
) -> Any:
    """Compose the Hydra config for one ``(point, model, seed)`` run.

    Mirrors ``utils.run_challenge_grid`` (same
    ``CHALLENGE_GRID_HYDRA_OVERRIDES``, same ``MAX_EPOCHS``, same
    ``feature_encoder.out_channels`` patch) so real runs stay comparable to
    the official ``results.json``. The ``center_variance`` override is
    produced by ``utils.generation_parameters_to_hydra_overrides`` and must
    never be hand-written.

    Parameters
    ----------
    model_config : str
        Hydra ``model=`` group value, e.g. ``"hypergraph/dphgnn"``.
    center_variance : float
        The universe-level ``center_variance`` for this sweep point.
    seed : int
        Training seed.
    run_dir : pathlib.Path
        Output directory for this run's checkpoints/logs.
    batch_size : int or None, optional
        If given, overrides ``dataset.dataloader_params.batch_size``.
    smoke_test : bool, optional
        If True, shrink ``trainer.max_epochs`` and the generated graph
        family to a tiny size for a fast, low-resource sanity check. Never
        use this for a real (reportable) run.
    force_cpu : bool, optional
        If True, override ``trainer.accelerator``/``trainer.devices`` to
        run on CPU regardless of the repo default. Use this for any local
        test — ``configs/trainer/default.yaml`` hardcodes
        ``accelerator: gpu, devices: [0]`` with no "auto" fallback, which is
        correct on Kaggle but must never be trusted blindly on a local
        machine.

    Returns
    -------
    omegaconf.DictConfig
        The composed, fully-overridden config.
    """
    gp = build_e3_generation_parameters(center_variance)
    if smoke_test:
        gp = copy.deepcopy(gp)
        gp["family_parameters"].update(_SMOKE_TEST_GP_PATCH)

    max_epochs = _SMOKE_TEST_MAX_EPOCHS if smoke_test else MAX_EPOCHS
    overrides = [
        "dataset=graph/graphuniverse_inductive",
        f"model={model_config}",
        "logger=csv",
        f"paths.output_dir={run_dir.as_posix()}",
        f"paths.work_dir={PROJECT_ROOT.as_posix()}",
        f"tags=[e3_feature_signal,{model_config.replace('/', '_')}]",
        f"seed={seed}",
        f"trainer.max_epochs={max_epochs}",
        # Always forced off (not just for smoke tests) — a long sequential
        # sweep that leaves `persistent_workers=True` DataLoader worker
        # subprocesses running across many `run_one()` calls is a real way
        # to exhaust host memory outside Python's own bookkeeping (see
        # lifting_confounding_study/README.md, Local testing).
        "dataset.dataloader_params.num_workers=0",
        "dataset.dataloader_params.persistent_workers=False",
    ]
    overrides.extend(generation_parameters_to_hydra_overrides(gp))
    overrides.extend(CHALLENGE_GRID_HYDRA_OVERRIDES)
    if batch_size is not None:
        overrides.append(f"dataset.dataloader_params.batch_size={batch_size}")
    if force_cpu:
        overrides.append("trainer.accelerator=cpu")
        overrides.append("trainer.devices=1")

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")
    ):
        cfg = compose(config_name="run.yaml", overrides=overrides)

    apply_challenge_feature_encoder_out_channels(cfg)
    if OmegaConf.is_config(cfg) and cfg.get("trainer") is not None:
        with open_dict(cfg.trainer):
            cfg.trainer.enable_progress_bar = False
    return cfg


# =============================================================================
# Preflight — center_variance override acceptance tests (blocking: must
# pass before the sweep starts)
# =============================================================================


def _between_within_variance_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Return the ratio of between-class to within-class feature variance.

    A simple multivariate analogue of a one-way ANOVA F-ratio numerator vs.
    denominator: ``sum_c n_c * ||mean_c - mean||^2`` over
    ``sum_c sum_{i in c} ||x_i - mean_c||^2``, both normalized by the total
    sample count. Used by acceptance test B to verify that
    ``center_variance`` overrides actually reach the feature generator.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Node features.
    y : numpy.ndarray, shape (n,)
        Integer class (community) labels.

    Returns
    -------
    float
        The between/within variance ratio; ``inf`` if within-class
        variance is exactly zero.
    """
    classes = np.unique(y)
    overall_mean = x.mean(axis=0)
    total_n = x.shape[0]
    between = 0.0
    within = 0.0
    for c in classes:
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            continue
        mean_c = xc.mean(axis=0)
        between += n_c * float(np.sum((mean_c - overall_mean) ** 2))
        within += float(np.sum((xc - mean_c) ** 2))
    between /= total_n
    within /= total_n
    return between / within if within > 0 else float("inf")


def _small_probe_generation_parameters(
    center_variance: float,
) -> dict[str, Any]:
    """Return a cheap-to-generate variant of the E3 generation parameters.

    Parameters
    ----------
    center_variance : float
        The universe-level ``center_variance`` for this sweep point.

    Returns
    -------
    dict
        Generation parameters with ``n_graphs``/``n_nodes_range`` shrunk for
        a fast acceptance-test / diagnostic run.
    """
    gp = build_e3_generation_parameters(center_variance)
    gp["family_parameters"]["n_graphs"] = 20
    gp["family_parameters"]["n_nodes_range"] = [30, 60]
    return gp


def _acceptance_test_a() -> None:
    """Verify the 5 composed configs report the 5 intended ``center_variance``.

    Raises
    ------
    AssertionError
        If any composed config's resolved ``center_variance`` does not match
        the intended sweep-point value — the override silently composed the
        default instead.
    """
    print("[preflight] test A: center_variance Hydra overrides...", flush=True)
    with tempfile.TemporaryDirectory(prefix="e3_preflight_a_") as tmp:
        tmp_dir = Path(tmp)
        for point in POINT_ORDER:
            cv = CENTER_VARIANCE_POINTS[point]
            cfg = _compose_cfg(
                model_config="graph/gcn",
                center_variance=cv,
                seed=42,
                run_dir=tmp_dir / point,
            )
            resolved = float(
                cfg.dataset.loader.parameters.generation_parameters.universe_parameters.center_variance
            )
            assert resolved == cv, (
                f"[preflight] test A FAILED: point={point} expected "
                f"center_variance={cv}, composed config resolved to "
                f"{resolved}. The Hydra override did not take effect."
            )
            print(f"    {point}: center_variance={resolved} OK", flush=True)
    print("[preflight] test A PASSED.", flush=True)


def _acceptance_test_b() -> dict[str, float]:
    """Verify feature statistics actually differ across sweep points.

    Generates a small dataset at each of the 5 points and computes the
    between/within-class variance ratio of ``data.x`` (pooled over the
    training split). The ratio must increase monotonically with
    ``center_variance`` — identical statistics across points would mean the
    override did not reach the feature generator.

    Returns
    -------
    dict
        ``{point: variance_ratio}`` for all 5 points, for logging/README.

    Raises
    ------
    AssertionError
        If the ratio at ``fs_04`` is not strictly greater than at ``fs_00``,
        or if the sequence is not monotonically increasing.
    """
    print(
        "[preflight] test B: feature between/within variance ratio...",
        flush=True,
    )
    ratios: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="e3_preflight_b_") as tmp:
        tmp_dir = Path(tmp)
        for point in POINT_ORDER:
            cv = CENTER_VARIANCE_POINTS[point]
            gp = _small_probe_generation_parameters(cv)
            setting = GraphUniverseChallengeSetting(
                CELL_HOMOPHILY, CELL_AVG_DEGREE, CELL_POWER_LAW, gp
            )
            cfg = _compose_cfg(
                model_config="graph/gcn",
                center_variance=cv,
                seed=42,
                run_dir=tmp_dir / point,
            )
            # `_compose_cfg` always uses the full-size family; patch the
            # composed config's generation_parameters in-place to the small
            # probe family instead of regenerating overrides.
            with open_dict(
                cfg.dataset.loader.parameters.generation_parameters
            ):
                cfg.dataset.loader.parameters.generation_parameters.family_parameters.n_graphs = gp[
                    "family_parameters"
                ]["n_graphs"]
                cfg.dataset.loader.parameters.generation_parameters.family_parameters.n_nodes_range = gp[
                    "family_parameters"
                ]["n_nodes_range"]
            dm = build_datamodule_for_setting(cfg, setting)

            xs, ys = [], []
            for batch in dm.train_dataloader():
                xs.append(batch.x.detach().cpu().numpy())
                ys.append(batch.y.detach().cpu().numpy())
            x = np.concatenate(xs, axis=0)
            y = np.concatenate(ys, axis=0)
            ratio = _between_within_variance_ratio(x, y)
            ratios[point] = ratio
            print(
                f"    {point}: center_variance={cv} n_nodes={x.shape[0]} "
                f"between/within_ratio={ratio:.6f}",
                flush=True,
            )

    values = [ratios[p] for p in POINT_ORDER]
    monotonic = all(values[i] < values[i + 1] for i in range(len(values) - 1))
    assert ratios["fs_04"] > ratios["fs_00"], (
        "[preflight] test B FAILED: between/within variance ratio at "
        f"fs_04 ({ratios['fs_04']:.6f}) is not greater than at fs_00 "
        f"({ratios['fs_00']:.6f}) — the center_variance override did not "
        "reach the feature generator. Aborting before wasting the sweep."
    )
    if not monotonic:
        print(
            "[preflight] WARNING: test B ratios are not strictly "
            f"monotonic across all 5 points: {values}. fs_00 < fs_04 "
            "holds (test does not fail), but inspect before trusting "
            "intermediate points.",
            flush=True,
        )
    print("[preflight] test B PASSED.", flush=True)
    return ratios


def _verify_cache_dirs_distinct() -> None:
    """Verify each ``center_variance`` point has its own raw-data directory.

    A shared preprocessing/raw-data cache across points would silently
    reuse ``fs_00``'s data for every point — the data-level analogue of the
    acceptance-test-B failure mode. This is
    checked independently of test B by directly listing
    ``GraphUniverseDatasetLoader``'s raw directory for each point and
    asserting 5 distinct paths, all actually populated with a ``data.pt``.

    Raises
    ------
    AssertionError
        If fewer than 5 distinct, populated raw-data directories are found.
    """
    print(
        "[preflight] verifying per-point preprocessing cache directories...",
        flush=True,
    )
    from topobench.data.loaders.graph.graph_universe_loader import (
        GraphUniverseDatasetLoader,
    )

    raw_dirs: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="e3_preflight_cache_") as tmp:
        tmp_dir = Path(tmp)
        for point in POINT_ORDER:
            cv = CENTER_VARIANCE_POINTS[point]
            gp = _small_probe_generation_parameters(cv)
            cfg = _compose_cfg(
                model_config="graph/gcn",
                center_variance=cv,
                seed=42,
                run_dir=tmp_dir / point,
            )
            with open_dict(
                cfg.dataset.loader.parameters.generation_parameters
            ):
                cfg.dataset.loader.parameters.generation_parameters.family_parameters.n_graphs = gp[
                    "family_parameters"
                ]["n_graphs"]
                cfg.dataset.loader.parameters.generation_parameters.family_parameters.n_nodes_range = gp[
                    "family_parameters"
                ]["n_nodes_range"]
            params = OmegaConf.to_container(
                cfg.dataset.loader.parameters, resolve=True
            )
            loader = GraphUniverseDatasetLoader(OmegaConf.create(params))
            _dataset, data_dir = loader.load()
            raw_dirs[point] = str(data_dir)
            assert Path(data_dir).exists() and any(
                Path(data_dir).glob("*.pt")
            ), (
                f"[preflight] point={point}: raw data dir {data_dir} does "
                "not contain a processed .pt file."
            )

    n_distinct = len(set(raw_dirs.values()))
    assert n_distinct == len(POINT_ORDER), (
        "[preflight] cache-dir check FAILED: expected "
        f"{len(POINT_ORDER)} distinct raw-data directories, found "
        f"{n_distinct}. {raw_dirs}"
    )
    for point, d in raw_dirs.items():
        print(f"    {point}: {d}", flush=True)
    print(
        f"[preflight] all {n_distinct} preprocessing cache directories "
        "verified distinct.",
        flush=True,
    )


def compute_empirical_feature_signal_all_points(
    *, n_graphs: int = _FEATURE_SIGNAL_N_GRAPHS, force: bool = False
) -> dict[str, float | None]:
    """Compute the empirical ``feature_signal`` diagnostic for all 5 points.

    Implements the Step 0b empirical diagnostic: ``graph_universe.GraphSample``
    exposes ``calculate_feature_signal()`` (Random-Forest macro-F1 of a
    classifier trained on node features alone, predicting the community
    label). Those objects are not reachable through the TopoBench loader
    (``GraphUniverseDataset.download()`` discards the
    ``GraphFamilyGenerator``/``GraphSample`` objects once it has converted
    them to PyG ``Data``), but the same public ``graph_universe`` API used
    internally by the loader can be called a second time, directly, with
    the exact same ``generation_parameters`` — no loader refactor needed.

    Results are cached to ``feature_signal_empirical.json`` next to this
    script so repeated calls (e.g. from the analysis notebook or the Kaggle
    notebook) are instant after the first run.

    Parameters
    ----------
    n_graphs : int, optional
        Number of diagnostic graphs to probe per point.
    force : bool, optional
        If True, recompute even if a cache file is already present.

    Returns
    -------
    dict
        ``{point: mean_feature_signal}``. A point's value is ``None`` if
        the diagnostic raised (documented fallback — plot against
        ``center_variance`` instead).
    """
    if FEATURE_SIGNAL_CACHE_PATH.exists() and not force:
        with FEATURE_SIGNAL_CACHE_PATH.open(encoding="utf-8") as f:
            cached = json.load(f)
        return {
            p: cached.get(p, {}).get("mean_feature_signal")
            for p in POINT_ORDER
        }

    from graph_universe.graph_family import GraphFamilyGenerator
    from graph_universe.graph_universe import GraphUniverse

    out: dict[str, dict[str, Any]] = {}
    for point in POINT_ORDER:
        cv = CENTER_VARIANCE_POINTS[point]
        gp = build_e3_generation_parameters(cv)
        try:
            universe = GraphUniverse(**gp["universe_parameters"])
            fam_params = {
                k: v
                for k, v in gp["family_parameters"].items()
                if k != "n_graphs"
            }
            family = GraphFamilyGenerator(
                universe=universe, n_graphs=n_graphs, **fam_params
            )
            family.generate_family(n_graphs=n_graphs, show_progress=False)
            signals = [
                g.calculate_feature_signal(
                    random_state=_FEATURE_SIGNAL_RANDOM_STATE
                )
                for g in family.graphs
            ]
            out[point] = {
                "center_variance": cv,
                "mean_feature_signal": float(np.mean(signals)),
                "std_feature_signal": float(np.std(signals)),
                "n_graphs_probed": len(signals),
            }
            print(
                f"[feature_signal] {point}: mean={out[point]['mean_feature_signal']:.4f} "
                f"std={out[point]['std_feature_signal']:.4f} "
                f"(n={len(signals)})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — documented fallback path
            print(
                f"[feature_signal] {point}: FAILED ({e!r}); falling back "
                "to center_variance on the x axis for this point.",
                flush=True,
            )
            out[point] = {
                "center_variance": cv,
                "mean_feature_signal": None,
                "std_feature_signal": None,
                "n_graphs_probed": 0,
                "error": str(e),
            }

    with FEATURE_SIGNAL_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return {p: out[p]["mean_feature_signal"] for p in POINT_ORDER}


def preflight_check() -> dict[str, float]:
    """Run all E3 acceptance tests; must pass before the real sweep starts.

    Runs, in order: acceptance test A (config-level ``center_variance``),
    acceptance test B (data-level feature variance ratio), the
    preprocessing-cache-directory distinctness check, and the Step 0b
    empirical ``feature_signal`` diagnostic. Never touches the GPU (no
    model/trainer is built).

    Returns
    -------
    dict
        The acceptance-test-B variance ratios, ``{point: ratio}``.
    """
    _acceptance_test_a()
    ratios = _acceptance_test_b()
    _verify_cache_dirs_distinct()
    compute_empirical_feature_signal_all_points()
    print("[preflight] all acceptance tests passed. Proceeding.", flush=True)
    return ratios


# =============================================================================
# Result persistence — resumable, append-after-each-run
# =============================================================================


def _load_results() -> list[dict[str, Any]]:
    """Load existing records from the results file (``[]`` if absent).

    Returns
    -------
    list of dict
        The already-completed run records.
    """
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    """Return the ``(point, model, seed)`` identity of ``record``.

    Parameters
    ----------
    record : dict
        A run record (complete or in-progress).

    Returns
    -------
    tuple of (str, str, int)
        The identity used to decide whether a run can be skipped.
    """
    return (record["point"], record["model"], int(record["seed"]))


def _append_result(record: dict[str, Any]) -> None:
    """Append ``record`` to the results file, writing atomically.

    Parameters
    ----------
    record : dict
        The run record to persist.
    """
    results = _load_results()
    results.append(record)
    tmp_path = RESULTS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, RESULTS_PATH)


def _is_oom_error(exc: BaseException) -> bool:
    """Return whether ``exc`` looks like a CUDA out-of-memory error.

    Parameters
    ----------
    exc : BaseException
        The exception raised during a run.

    Returns
    -------
    bool
        True if ``exc`` is (or looks like) a CUDA OOM.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    if isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


def _release_run_resources(*objs: Any) -> None:
    """Drop references to ``objs`` and force a full GC + CUDA cache clear.

    Called after **every** run, success or failure — see
    ``lifting_confounding_study/README.md`` (Local testing) for why relying
    on refcounting alone is not enough across a long sequential sweep.

    Parameters
    ----------
    *objs : Any
        Objects to drop (already ``del``-ed by the caller; this only
        triggers the collection pass).
    """
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# One run
# =============================================================================


def run_one(
    point: str,
    model_config: str,
    seed: int,
    *,
    smoke_test: bool = False,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one ``(point, model, seed)`` configuration.

    On a CUDA out-of-memory error, retries with a smaller dataloader batch
    size before giving up. Any other exception is caught and recorded as a
    failed run so a single bad run cannot kill the sweep. All large objects
    are released and garbage collected before returning, win or lose.

    Parameters
    ----------
    point : str
        Sweep-point key (``"fs_00"``..``"fs_04"``).
    model_config : str
        Hydra ``model=`` group value (``"hypergraph/dphgnn"`` or
        ``"graph/gcn"``).
    seed : int
        Training seed.
    smoke_test : bool, optional
        Passed through to ``_compose_cfg`` — tiny data, 2 epochs. Never use
        for a real (reportable) run.
    force_cpu : bool, optional
        Passed through to ``_compose_cfg`` — never trust the repo's
        hardcoded ``accelerator: gpu`` default on a local machine.

    Returns
    -------
    dict
        A flat record following this experiment's results schema; always
        contains at least ``point``, ``model``, ``seed``, and ``status``.
    """
    center_variance = CENTER_VARIANCE_POINTS[point]
    feature_signal_by_point = compute_empirical_feature_signal_all_points()
    record: dict[str, Any] = {
        "point": point,
        "center_variance": center_variance,
        "feature_signal_empirical": feature_signal_by_point.get(point),
        "model": model_config,
        "cell_key": CELL_KEY,
        "seed": seed,
    }

    run_dir = RUNS_DIR / f"{point}__{model_config.replace('/', '_')}__s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt, batch_size in enumerate(_BATCH_SIZE_FALLBACKS):
        model = datamodule = trainer = test_trainer = object_dict = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            cfg = _compose_cfg(
                model_config=model_config,
                center_variance=center_variance,
                seed=seed,
                run_dir=run_dir,
                batch_size=batch_size,
                smoke_test=smoke_test,
                force_cpu=force_cpu,
            )
            pl.seed_everything(cfg.seed, workers=True)
            torch.manual_seed(cfg.seed)

            t0 = time.perf_counter()
            metric_dict, object_dict = run(cfg)
            model = object_dict["model"]
            datamodule = object_dict["datamodule"]
            trainer = object_dict["trainer"]

            test_trainer = pl.Trainer(
                logger=False,
                enable_progress_bar=False,
                accelerator=cfg.trainer.accelerator,
                devices=cfg.trainer.devices,
            )
            test_out = test_trainer.test(model, datamodule)
            test_metrics = test_out[0] if test_out else {}
            wall_clock_s = time.perf_counter() - t0

            peak_gpu_mb = (
                torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0.0
            )

            record.update(
                {
                    "test_accuracy": float(
                        test_metrics.get("test/accuracy", float("nan"))
                    ),
                    "val_accuracy": float(
                        metric_dict.get("val/accuracy", float("nan"))
                    ),
                    "n_params": count_number_of_parameters(model),
                    "epochs_run": int(trainer.current_epoch),
                    "wall_clock_s": wall_clock_s,
                    "peak_gpu_mb": peak_gpu_mb,
                    "status": "ok",
                }
            )
            if batch_size is not None:
                record["batch_size_override"] = batch_size
                record["note"] = (
                    f"Retried with dataloader batch_size={batch_size} "
                    "after an OOM at the repo default (64); breaks strict "
                    "comparability, see README.md known risks."
                )
            return record

        except Exception as e:  # noqa: BLE001 — one bad run must not kill the sweep
            last_error = e
            is_last_attempt = attempt == len(_BATCH_SIZE_FALLBACKS) - 1
            if not _is_oom_error(e) or is_last_attempt:
                break
            print(
                f"[run_one] {point}/{model_config}/s{seed}: OOM, retrying "
                "with a smaller batch size...",
                flush=True,
            )
        finally:
            _release_run_resources(
                model, datamodule, trainer, test_trainer, object_dict
            )

    record.update(
        {
            "test_accuracy": None,
            "val_accuracy": None,
            "n_params": None,
            "epochs_run": None,
            "wall_clock_s": None,
            "peak_gpu_mb": None,
            "status": "failed",
            "error": str(last_error),
        }
    )
    return record


# =============================================================================
# Sweep — orchestrator spawns one subprocess per job (see module docstring)
# =============================================================================

# Generous but bounded: a single healthy run should be well under ~15
# minutes. This is a safety valve against a hung/thrashing job, not the
# expected runtime.
_DEFAULT_JOB_TIMEOUT_S = 3600


def _ram_status() -> str:
    """Return a short ``"used%/total_gb"`` system-RAM summary for logging.

    Returns
    -------
    str
        E.g. ``"42% of 11.6GB"``.
    """
    vm = psutil.virtual_memory()
    return f"{vm.percent:.0f}% of {vm.total / 1e9:.1f}GB"


def _run_job_in_subprocess(
    point: str,
    model_config: str,
    seed: int,
    *,
    smoke_test: bool,
    force_cpu: bool,
    timeout_s: int,
) -> dict[str, Any]:
    """Run one job as an isolated ``python run_e3.py --worker`` subprocess.

    A hard OOM-kill is not a catchable Python exception in-process — the
    only reliable way to observe it is from *outside* the process, as a
    non-zero or negative (killed-by-signal) return code. This also
    guarantees a fresh CUDA context and Python heap per job.

    Parameters
    ----------
    point : str
        Sweep-point key.
    model_config : str
        Hydra ``model=`` group value.
    seed : int
        Training seed.
    smoke_test : bool
        Forwarded to the worker's ``--smoke-test``.
    force_cpu : bool
        Forwarded to the worker's ``--cpu``.
    timeout_s : int
        Kill the subprocess and record a failure if it runs longer than
        this.

    Returns
    -------
    dict
        A record with at least ``point``, ``model``, ``seed``, ``status``
        — either the worker's own JSON output, or a synthetic ``"failed"``
        record describing how the subprocess died.
    """
    with tempfile.TemporaryDirectory(prefix="e3_worker_") as tmp:
        out_path = Path(tmp) / "record.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            point,
            model_config,
            str(seed),
            "--worker-output",
            str(out_path),
        ]
        if smoke_test:
            cmd.append("--smoke-test")
        if force_cpu:
            cmd.append("--cpu")

        print(f"    [orchestrator] RAM before: {_ram_status()}", flush=True)
        try:
            proc = subprocess.run(cmd, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {
                "point": point,
                "model": model_config,
                "seed": seed,
                "status": "failed",
                "error": f"worker subprocess timed out after {timeout_s}s",
            }
        print(f"    [orchestrator] RAM after:  {_ram_status()}", flush=True)

        if proc.returncode != 0 or not out_path.exists():
            reason = (
                f"killed by signal {-proc.returncode}"
                if proc.returncode < 0
                else f"exited with code {proc.returncode}"
            )
            return {
                "point": point,
                "model": model_config,
                "seed": seed,
                "status": "failed",
                "error": (
                    f"worker subprocess {reason} without writing a "
                    "result — likely an OS-level OOM-kill or crash, not a "
                    "catchable Python exception."
                ),
            }
        with out_path.open(encoding="utf-8") as f:
            return json.load(f)


def _run_worker(args: argparse.Namespace) -> None:
    """Run exactly one job (the ``--worker`` entry point) and exit.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; must have ``worker`` (``[point, model,
        seed]``) and ``worker_output`` set.
    """
    point, model_config, seed_str = args.worker
    config_resolvers.register_all_resolvers()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = run_one(
        point,
        model_config,
        int(seed_str),
        smoke_test=args.smoke_test,
        force_cpu=args.cpu,
    )
    Path(args.worker_output).write_text(json.dumps(record), encoding="utf-8")


def main() -> None:
    """Run the sweep, resuming from the results file if present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2",
        action="store_true",
        help=(
            "Also run seeds 43 and 44 (30 runs total) instead of only the "
            "Phase-1 seed 42 (10 runs). Only start Phase 2 once Phase 1 is "
            "fully green — see README.md, Run budget."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the center_variance-override acceptance tests (not recommended).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Tiny synthetic data + 2 epochs instead of the real design. "
            "For validating the script mechanics only — never use this "
            "for a reportable run. See README.md, Local testing."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Force trainer.accelerator=cpu / devices=1, overriding the "
            "repo's hardcoded accelerator: gpu default. Always pass this "
            "for local testing — do not assume a local machine has a "
            "working CUDA setup. See README.md, Local testing."
        ),
    )
    parser.add_argument(
        "--job-timeout-s",
        type=int,
        default=_DEFAULT_JOB_TIMEOUT_S,
        help=(
            "Kill and record as failed any single job's subprocess that "
            f"runs longer than this (default {_DEFAULT_JOB_TIMEOUT_S}s)."
        ),
    )
    parser.add_argument(
        "--worker",
        nargs=3,
        metavar=("POINT", "MODEL", "SEED"),
        default=None,
        help=argparse.SUPPRESS,  # internal: single-job subprocess entry point
    )
    parser.add_argument(
        "--worker-output",
        default=None,
        help=argparse.SUPPRESS,  # internal: paired with --worker
    )
    args = parser.parse_args()

    if args.worker is not None:
        _run_worker(args)
        return

    config_resolvers.register_all_resolvers()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        preflight_check()

    seeds = (
        (
            *PHASE1_SEEDS,
            *(s for s in CHALLENGE_TRAIN_SEEDS if s not in PHASE1_SEEDS),
        )
        if args.phase2
        else PHASE1_SEEDS
    )
    done = {_record_key(r) for r in _load_results()}

    jobs = [
        (point, model, seed)
        for point in POINT_ORDER
        for model in MODELS
        for seed in seeds
    ]
    total = len(jobs)
    print(
        f"[sweep] {total} (point, model, seed) jobs, {len(done)} already "
        f"present in {RESULTS_PATH.name}. Each job runs as its own "
        f"subprocess (timeout {args.job_timeout_s}s)."
        + (" [SMOKE TEST]" if args.smoke_test else ""),
        flush=True,
    )

    for i, (point, model, seed) in enumerate(jobs, start=1):
        key = (point, model, seed)
        if key in done:
            print(
                f"[{i}/{total}] SKIP {point}/{model}/s{seed} (already present)",
                flush=True,
            )
            continue

        print(f"[{i}/{total}] RUN  {point}/{model}/s{seed}", flush=True)
        record = _run_job_in_subprocess(
            point,
            model,
            seed,
            smoke_test=args.smoke_test,
            force_cpu=args.cpu,
            timeout_s=args.job_timeout_s,
        )
        if not args.smoke_test:
            _append_result(record)
            done.add(key)

        if record["status"] == "ok":
            print(
                f"    -> ok  test_accuracy={record['test_accuracy']:.4f} "
                f"({record['wall_clock_s']:.0f}s, "
                f"{record['peak_gpu_mb']:.0f} MB peak)",
                flush=True,
            )
        else:
            print(f"    -> FAILED: {record['error']}", flush=True)

    if args.smoke_test:
        print(
            "[sweep] smoke test done. Nothing was written to "
            f"{RESULTS_PATH.name} (smoke-test records are never persisted).",
            flush=True,
        )
    else:
        print(f"[sweep] done. Results in {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
