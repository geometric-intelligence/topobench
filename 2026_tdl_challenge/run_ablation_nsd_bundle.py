"""Run the full GraphUniverse challenge grid for upstream NSD-bundle.

Apples-to-apples ablation against Conn-NSD: same harness, same grid, same
trainer budget, same hidden_dim — but with the *learned* bundle maps of
Bodnar et al. (NSDEncoder, sheaf_type="bundle") instead of the
deterministic Algorithm-1 maps of Conn-NSD.

Hyperparameter overrides force NSD-bundle to use stalk dim ``d=4`` and
``dropout=0.0`` to match the Conn-NSD config; everything else is the
upstream default (``num_layers=2``, ``hidden_dim=64``, etc.).

Output mirrors the conn_nsd_full layout::

    2026_tdl_challenge/outputs/nsd_bundle_full/
        results.json
        heatmap_community_detection_accuracy.png
        heatmap_triangle_mse_over_triangles.png
        OOD/
            OOD_{low,mid,high}_homophily__{community_detection,triangle_counting}.png

Run with::

    cd 2026_tdl_challenge
    WANDB_MODE=offline python run_ablation_nsd_bundle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utils import (  # noqa: E402  — sys.path setup must precede this import
    resolve_project_root,
    run_challenge_grid,
    save_challenge_artifacts,
)

MODEL_CONFIG = "graph/nsd"
# Override NSD-bundle to match the Conn-NSD config (stalk_dim=4, dropout=0).
# Anything else left at upstream defaults — same num_layers, same hidden_dim.
EXTRA_OVERRIDES = [
    "model.backbone.d=4",
    "model.backbone.dropout=0.0",
    "model.backbone.sheaf_type=bundle",
]
OUTPUT_DIR_NAME = "nsd_bundle_full"


def main() -> None:
    """Run the 72-cell grid for NSD-bundle and save artefacts."""
    project_root = resolve_project_root(_REPO)
    print(f"Project root: {project_root}", flush=True)
    print(f"Model: {MODEL_CONFIG}", flush=True)
    print(f"Overrides: {EXTRA_OVERRIDES}", flush=True)

    results, study_id = run_challenge_grid(
        project_root=project_root,
        model_config=MODEL_CONFIG,
        extra_overrides=EXTRA_OVERRIDES,
        quiet=True,
    )
    print(
        f"\nGrid finished: {len(results)} runs; study_id={study_id}",
        flush=True,
    )

    out_dir = _HERE / "outputs" / OUTPUT_DIR_NAME
    out = save_challenge_artifacts(
        results,
        out_dir=out_dir,
        model_config=MODEL_CONFIG,
        study_id=study_id,
    )
    print("Artefacts:")
    for k, v in out.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
