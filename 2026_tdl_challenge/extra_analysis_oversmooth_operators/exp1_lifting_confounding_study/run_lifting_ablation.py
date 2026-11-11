"""Runner for the lifting confounding study.

See ``README.md`` in this folder for the scientific question, design, and
findings. This is a plain script (not a notebook) so it survives a Kaggle
session restart: it is idempotent and resumable, appending one record to
``lifting_ablation_results.json`` immediately after each run and skipping
any ``(arm, cell_key, seed)`` already present on the next invocation.

Each ``(arm, cell, seed)`` job runs in its **own subprocess**, spawned by
the orchestrator in ``main()``. A hard OOM-kill (or, on WSL2, a host
memory-pressure freeze) is not a catchable Python exception — the process,
or the whole machine, just dies with nothing logged. That already happened
once locally. Running each job in a disposable subprocess means: (a) the
orchestrator itself never touches CUDA and survives any single job dying,
observing the crash as an ordinary non-zero/killed exit code, and (b) every
job gets a completely fresh CUDA context and Python heap, so nothing can
accumulate across jobs no matter how the in-process cleanup behaves.

Usage
-----
Phase 1 (16 runs, seed 42 only)::

    python run_lifting_ablation.py

Phase 2 (48 runs total, adds seeds 43 and 44 — only once Phase 1 is fully
green)::

    python run_lifting_ablation.py --phase2

Safe local sanity check *before* launching anything real (tiny synthetic
data, 2 epochs, CPU only — see "Local testing" in README.md)::

    python run_lifting_ablation.py --smoke-test --cpu
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
RESULTS_PATH = _HERE / "lifting_ablation_results.json"
RUNS_DIR = _HERE / "runs"

# =============================================================================
# Experimental design — see README.md for the full rationale
# =============================================================================

_LIFTING_GROUP = (
    "transforms/liftings/graph2hypergraph@transforms.graph2hypergraph_lifting"
)

ARMS: dict[str, dict[str, Any]] = {
    "khop1": {
        "model_config": "hypergraph/dphgnn",
        "lifting_name": "HypergraphKHopLifting",
        "lifting_k": 1,
        "overrides": [f"{_LIFTING_GROUP}=khop"],
    },
    "khop2": {
        "model_config": "hypergraph/dphgnn",
        "lifting_name": "HypergraphKHopLifting",
        "lifting_k": 2,
        "overrides": [
            f"{_LIFTING_GROUP}=khop",
            "transforms.graph2hypergraph_lifting.k_value=2",
        ],
    },
    "knn3": {
        "model_config": "hypergraph/dphgnn",
        "lifting_name": "HypergraphKNNLifting",
        "lifting_k": 3,
        "overrides": [f"{_LIFTING_GROUP}=knn"],
    },
    "gcn": {
        "model_config": "graph/gcn",
        "lifting_name": None,
        "lifting_k": None,
        "overrides": [],
    },
}

# Four official grid cells (via `build_generation_parameters`), avg_degree
# held at d_lo throughout — keeps graphs sparse, caps runtime, and prevents
# khop2 hyperedges from exploding (README.md, Data settings).
CELL_KEYS: list[tuple[str, str, str]] = [
    ("h_lo", "d_lo", "pl_lo"),
    ("h_lo", "d_lo", "pl_hi"),
    ("h_hi", "d_lo", "pl_lo"),
    ("h_hi", "d_lo", "pl_hi"),
]

PHASE1_SEEDS: tuple[int, ...] = (42,)

# OOM fallback ladder for the dataloader batch size (README.md, Known
# risks: khop2 OOM). `None` means "use the repo default (64)".
_BATCH_SIZE_FALLBACKS: tuple[int | None, ...] = (None, 32, 16)

# --smoke-test: tiny synthetic data + 2 epochs, for validating the script's
# mechanics in seconds without the resource footprint of a real run (see
# README.md, Local testing).
_SMOKE_TEST_MAX_EPOCHS = 2
_SMOKE_TEST_GP_PATCH: dict[str, Any] = {
    "n_graphs": 20,
    "n_nodes_range": [20, 40],
}


def cell_key_str(homophily: str, avg_degree: str, power_law: str) -> str:
    """Return the canonical grid-cell slug.

    Parameters
    ----------
    homophily : str
        Homophily level key, e.g. ``"h_lo"``.
    avg_degree : str
        Average-degree level key, e.g. ``"d_lo"``.
    power_law : str
        Power-law exponent level key, e.g. ``"pl_lo"``.

    Returns
    -------
    str
        The ``f"{homophily}__{avg_degree}__{power_law}"`` slug used
        throughout ``lifting_ablation_results.json``.
    """
    return f"{homophily}__{avg_degree}__{power_law}"


# =============================================================================
# Hydra composition
# =============================================================================


def _compose_cfg(
    *,
    model_config: str,
    homophily: str,
    avg_degree: str,
    power_law: str,
    seed: int,
    lifting_overrides: list[str],
    run_dir: Path,
    batch_size: int | None = None,
    smoke_test: bool = False,
    force_cpu: bool = False,
) -> Any:
    """Compose the Hydra config for one ``(arm, cell, seed)`` run.

    Mirrors ``utils.run_challenge_grid`` (same
    ``CHALLENGE_GRID_HYDRA_OVERRIDES``, same ``MAX_EPOCHS``, same
    ``feature_encoder.out_channels`` patch) so real runs stay comparable to
    the official ``results.json``. Divergences (always on, not just under
    ``smoke_test``/``force_cpu``) are logged in README.md, Comparability:
    dataloader workers are always forced to 0 (see the note above
    ``dataset.dataloader_params.num_workers`` below).

    Parameters
    ----------
    model_config : str
        Hydra ``model=`` group value, e.g. ``"hypergraph/dphgnn"``.
    homophily : str
        Homophily level key.
    avg_degree : str
        Average-degree level key.
    power_law : str
        Power-law exponent level key.
    seed : int
        Training seed.
    lifting_overrides : list of str
        Extra Hydra overrides selecting/parameterizing the lifting (empty
        for the GCN baseline, which needs none).
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
        test — the repo's ``configs/trainer/default.yaml`` hardcodes
        ``accelerator: gpu, devices: [0]`` with no "auto" fallback, which
        is correct on Kaggle but must never be trusted blindly on a local
        machine (see README.md, Local testing).

    Returns
    -------
    omegaconf.DictConfig
        The composed, fully-overridden config.
    """
    gp = build_generation_parameters(homophily, avg_degree, power_law)
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
        f"tags=[lifting_confounding_study,{model_config.replace('/', '_')}]",
        f"seed={seed}",
        f"trainer.max_epochs={max_epochs}",
        # Always forced off (not just for smoke tests): a long sequential
        # sweep that leaves `persistent_workers=True` DataLoader worker
        # subprocesses running across many `run_one()` calls is a real way
        # to exhaust host memory outside Python's own bookkeeping — this
        # is what wedged a local machine once already. Worth the small
        # dataloading throughput cost for a bounded ablation study.
        "dataset.dataloader_params.num_workers=0",
        "dataset.dataloader_params.persistent_workers=False",
    ]
    overrides.extend(lifting_overrides)
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
# Preflight — lifting-override acceptance tests (must pass before the
# sweep starts)
# =============================================================================


def preflight_check() -> None:
    """Verify the lifting overrides actually take effect before the sweep.

    Two assertions per hypergraph arm (``khop1``, ``khop2``, ``knn3``):

    1. The composed config's ``transforms.graph2hypergraph_lifting`` block
       has the intended ``transform_name`` / ``k_value``.
    2. On the same tiny synthetic graph family, the three arms produce
       pairwise-*different* ``incidence_hyperedges``. A silent override
       failure would make all three identical, which would invalidate the
       whole experiment without erroring.

    This never touches the GPU (it only builds a datamodule, never a
    model/trainer), so it is always safe to run.

    Raises
    ------
    AssertionError
        If any override did not take effect as intended.
    """
    print("[preflight] checking lifting overrides...", flush=True)
    incidences: dict[str, torch.Tensor] = {}

    with tempfile.TemporaryDirectory(
        prefix="lifting_ablation_preflight_"
    ) as tmp:
        tmp_dir = Path(tmp)
        gp = copy.deepcopy(
            build_generation_parameters("h_lo", "d_lo", "pl_lo")
        )
        gp["family_parameters"]["n_graphs"] = 6
        gp["family_parameters"]["n_nodes_range"] = [20, 30]
        setting = GraphUniverseChallengeSetting("h_lo", "d_lo", "pl_lo", gp)

        for arm, spec in ARMS.items():
            if spec["lifting_name"] is None:
                continue  # GCN baseline: no lifting to check.

            run_dir = tmp_dir / arm
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = _compose_cfg(
                model_config=spec["model_config"],
                homophily="h_lo",
                avg_degree="d_lo",
                power_law="pl_lo",
                seed=42,
                lifting_overrides=spec["overrides"],
                run_dir=run_dir,
            )
            block = cfg.transforms.graph2hypergraph_lifting
            got_name = block.get("transform_name")
            got_k = block.get("k_value", None)
            assert got_name == spec["lifting_name"], (
                f"[preflight] arm={arm}: expected transform_name="
                f"{spec['lifting_name']!r}, got {got_name!r}. The Hydra "
                "override did not select the intended lifting."
            )
            if spec["lifting_k"] is not None:
                assert got_k == spec["lifting_k"], (
                    f"[preflight] arm={arm}: expected k_value="
                    f"{spec['lifting_k']!r}, got {got_k!r}."
                )

            dm = build_datamodule_for_setting(cfg, setting)
            batch = next(iter(dm.train_dataloader()))
            incidences[arm] = batch.incidence_hyperedges.to_dense()
            print(
                f"[preflight] {arm}: transform_name={got_name} "
                f"k_value={got_k} OK",
                flush=True,
            )

    names = list(incidences)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = incidences[names[i]], incidences[names[j]]
            identical = a.shape == b.shape and torch.equal(a, b)
            assert not identical, (
                f"[preflight] {names[i]} and {names[j]} produced IDENTICAL "
                "incidence matrices on the same input graphs — the lifting "
                "override silently failed to take effect. Aborting before "
                "wasting the sweep."
            )
    print(
        "[preflight] all lifting overrides verified distinct. Proceeding.",
        flush=True,
    )


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
    """Return the ``(arm, cell_key, seed)`` identity of ``record``.

    Parameters
    ----------
    record : dict
        A run record (complete or in-progress).

    Returns
    -------
    tuple of (str, str, int)
        The identity used to decide whether a run can be skipped.
    """
    return (record["arm"], record["cell_key"], int(record["seed"]))


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

    Lightning's ``Trainer``/``DataLoader``/callback graph holds reference
    cycles, so relying on refcounting alone leaves them (and, with them,
    any DataLoader worker subprocesses or CUDA allocations) alive until
    the *next* cyclic-GC pass — which may never come during a long
    sequential sweep. Called after **every** run, success or failure, not
    just on the OOM path.

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
    arm: str,
    cell_key: str,
    seed: int,
    *,
    smoke_test: bool = False,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one ``(arm, cell, seed)`` configuration.

    On a CUDA out-of-memory error, retries with a smaller dataloader batch
    size (README.md, Known risks: khop2 OOM) before giving up. Any other
    exception is caught and recorded as a failed run so a single bad run
    cannot kill the sweep. All large objects are released and garbage
    collected before returning, win or lose.

    Parameters
    ----------
    arm : str
        Key into ``ARMS`` (``"khop1"``, ``"khop2"``, ``"knn3"``, or
        ``"gcn"``).
    cell_key : str
        ``"{homophily}__{avg_degree}__{power_law}"`` slug identifying the
        GraphUniverse grid cell.
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
        A flat record following the schema in README.md; always contains
        at least ``arm``, ``cell_key``, ``seed``, and ``status``.
    """
    spec = ARMS[arm]
    homophily, avg_degree, power_law = cell_key.split("__")
    record: dict[str, Any] = {
        "arm": arm,
        "lifting_name": spec["lifting_name"],
        "lifting_k": spec["lifting_k"],
        "model": spec["model_config"],
        "cell_key": cell_key,
        "homophily": homophily,
        "avg_degree": avg_degree,
        "power_law": power_law,
        "seed": seed,
    }

    run_dir = RUNS_DIR / f"{arm}__{cell_key}__s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt, batch_size in enumerate(_BATCH_SIZE_FALLBACKS):
        model = datamodule = trainer = test_trainer = object_dict = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            cfg = _compose_cfg(
                model_config=spec["model_config"],
                homophily=homophily,
                avg_degree=avg_degree,
                power_law=power_law,
                seed=seed,
                lifting_overrides=spec["overrides"],
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
                f"[run_one] {arm}/{cell_key}/s{seed}: OOM, retrying with "
                f"a smaller batch size...",
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

# Generous but bounded: README.md says a single healthy run should be well
# under ~15 minutes. This is a safety valve against a hung/thrashing job,
# not the expected runtime.
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
    arm: str,
    cell: str,
    seed: int,
    *,
    smoke_test: bool,
    force_cpu: bool,
    timeout_s: int,
) -> dict[str, Any]:
    """Run one job as an isolated ``python run_lifting_ablation.py --worker`` subprocess.

    A hard OOM-kill is not a catchable Python exception in-process — the
    only reliable way to observe it is from *outside* the process, as a
    non-zero or negative (killed-by-signal) return code. This also
    guarantees a fresh CUDA context and Python heap per job, so nothing
    can accumulate across the sweep regardless of in-process cleanup.

    Parameters
    ----------
    arm : str
        Key into ``ARMS``.
    cell : str
        Grid-cell slug.
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
        A record with at least ``arm``, ``cell_key``, ``seed``, ``status``
        — either the worker's own JSON output, or a synthetic ``"failed"``
        record describing how the subprocess died.
    """
    with tempfile.TemporaryDirectory(prefix="lifting_ablation_worker_") as tmp:
        out_path = Path(tmp) / "record.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            arm,
            cell,
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
                "arm": arm,
                "cell_key": cell,
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
                "arm": arm,
                "cell_key": cell,
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
        Parsed CLI arguments; must have ``worker`` (``[arm, cell, seed]``)
        and ``worker_output`` set.
    """
    arm, cell, seed_str = args.worker
    config_resolvers.register_all_resolvers()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = run_one(
        arm,
        cell,
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
            "Also run seeds 43 and 44 (48 runs total) instead of only the "
            "Phase-1 seed 42 (16 runs). Only start Phase 2 once Phase 1 is "
            "fully green — see README.md, Run budget."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the lifting-override acceptance tests (not recommended).",
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
        metavar=("ARM", "CELL", "SEED"),
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

    seeds = CHALLENGE_TRAIN_SEEDS if args.phase2 else PHASE1_SEEDS
    done = {_record_key(r) for r in _load_results()}

    jobs = [
        (arm, cell_key_str(*cell), seed)
        for arm in ARMS
        for cell in CELL_KEYS
        for seed in seeds
    ]
    total = len(jobs)
    print(
        f"[sweep] {total} (arm, cell, seed) jobs, {len(done)} already "
        f"present in {RESULTS_PATH.name}. Each job runs as its own "
        f"subprocess (timeout {args.job_timeout_s}s)."
        + (" [SMOKE TEST]" if args.smoke_test else ""),
        flush=True,
    )

    for i, (arm, cell, seed) in enumerate(jobs, start=1):
        key = (arm, cell, seed)
        if key in done:
            print(
                f"[{i}/{total}] SKIP {arm}/{cell}/s{seed} (already present)",
                flush=True,
            )
            continue

        print(f"[{i}/{total}] RUN  {arm}/{cell}/s{seed}", flush=True)
        record = _run_job_in_subprocess(
            arm,
            cell,
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
