"""Official 72-run challenge grid for SheafTSP, partitioned for
parallel execution.

Usage:
    python grid_runner.py <partition>   # partition in {h_lo, h_mid, h_hi}

Runs the official harness (utils.run_challenge_grid) on the 4 grid
cells of one homophily band, both tasks, seeds 42/43/44 — 24 runs per
partition, no model overrides (frozen sheaf_tsp defaults). Partial
results are saved as JSON for merging with merge_grid.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2026_tdl_challenge"))

import utils as U

partition = sys.argv[1]
assert partition in ("h_lo", "h_mid", "h_hi")

_orig_iter = U.iter_challenge_settings


def _filtered():
    for s in _orig_iter():
        if s.homophily_key == partition:
            yield s


U.iter_challenge_settings = _filtered

results, sid = U.run_challenge_grid(
    project_root=ROOT,
    model_config="cell/sheaf_tsp",
    extra_overrides=[
        "trainer.accelerator=cpu",
        "trainer.devices=1",
    ],
    study_id=f"sheaf_tsp_grid_{partition}",
    quiet=True,
)

out = ROOT / "2026_tdl_challenge" / f"grid_partial_{partition}.json"
with open(out, "w") as f:
    json.dump(results, f)
print("PARTITION_DONE", partition, len(results), "runs ->", out)
