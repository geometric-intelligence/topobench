"""Single-cell experiment runner for SheafTSP under the official protocol.

Usage:
    python exp_sheaf.py <study_id> <slugs-csv> <task> [hydra overrides...]

Example:
    python exp_sheaf.py deep_filter h_mid__d_lo__pl_lo both \
        model.backbone.filter_order=8 model.backbone.n_layers=4

Runs the official harness (500 epochs, early stopping, encoder forced
to 64) on the given cells with extra hydra overrides, 1 seed.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2026_tdl_challenge"))

import utils as U

study_id = sys.argv[1]
slugs = set(sys.argv[2].split(","))
task = sys.argv[3]  # community_detection | triangle_counting | both
overrides = sys.argv[4:]

_orig_iter = U.iter_challenge_settings


def _filtered():
    for s in _orig_iter():
        if s.run_slug in slugs:
            yield s


U.iter_challenge_settings = _filtered

modes = [
    m
    for m in U.DEFAULT_EXPERIMENT_MODES
    if task in ("both", m[0])
]

results, sid = U.run_challenge_grid(
    project_root=ROOT,
    model_config="cell/sheaf_tsp",
    extra_overrides=[
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        *overrides,
    ],
    experiment_modes=modes,
    train_seeds=tuple(
        int(s) for s in os.environ.get("SEEDS", "42").split(",")
    ),
    study_id=study_id,
    quiet=True,
)

summary = [
    {
        k: r.get(k)
        for k in (
            "experiment",
            "run_slug",
            "train_seed",
            "test_best_rerun_accuracy",
            "test_best_rerun_mse",
            "test_mse_by_total_triangles",
        )
    }
    for r in results
]
print("EXP_SUMMARY_JSON", study_id)
print(json.dumps(summary, indent=1))
