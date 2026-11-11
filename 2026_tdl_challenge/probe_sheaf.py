"""Probe: SheafTSP under the real 500-epoch protocol on 3 representative cells.

Runs the official harness (utils.run_challenge_grid) restricted to
3 grid cells x 2 tasks x 1 seed to measure the true gap vs the
leaderboard before committing to the full 72-run grid.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2026_tdl_challenge"))

import utils as U

# hard heterophily / old reference cell / easy homophilic
SLUGS = {"h_lo__d_lo__pl_lo", "h_mid__d_lo__pl_lo", "h_hi__d_hi__pl_lo"}

_orig_iter = U.iter_challenge_settings


def _filtered():
    for s in _orig_iter():
        if s.run_slug in SLUGS:
            yield s


U.iter_challenge_settings = _filtered

results, sid = U.run_challenge_grid(
    project_root=ROOT,
    model_config="cell/sheaf_tsp",
    extra_overrides=[
        "trainer.accelerator=cpu",
        "trainer.devices=1",
    ],
    train_seeds=(42,),
    study_id="probe_sheaf_upstream",
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
print("PROBE_SUMMARY_JSON")
print(json.dumps(summary, indent=1))

out = U.save_challenge_artifacts(
    results, model_config="cell/sheaf_tsp", study_id=sid
)
print("saved:", out.get("json"))
