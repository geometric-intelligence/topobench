"""Merge grid partitions into the official results.json artifacts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2026_tdl_challenge"))

import utils as U

results = []
for part in ("h_lo", "h_mid", "h_hi"):
    p = ROOT / "2026_tdl_challenge" / f"grid_partial_{part}.json"
    with open(p) as f:
        chunk = json.load(f)
    print(f"{part}: {len(chunk)} runs")
    results.extend(chunk)

assert len(results) == 72, f"expected 72 runs, got {len(results)}"
out = U.save_challenge_artifacts(
    results, model_config="cell/sheaf_tsp", study_id="sheaf_tsp_full_grid"
)
print("results.json:", out["json"])
