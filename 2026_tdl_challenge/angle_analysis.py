"""Transport-angle diagnostic: are the learned transports non-trivial?

Loads a trained checkpoint, rebuilds the datamodule for its grid cell,
hooks the RestrictionMapLearner, and reports the distribution of learned
SO(2) rotation angles over test batches. Identity transports (angles at 0)
would mean the sheaf machinery collapsed to a weighted GCN; a spread
correlated with heterophily is direct evidence for the
consistency-over-similarity hypothesis.

Usage: python angle_analysis.py <ckpt_glob> <h_key> <d_key> <pl_key>
"""

import glob
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2026_tdl_challenge"))

import utils as U
from topobench.utils.config_resolvers import register_all_resolvers

ckpt_glob, h_key, d_key, pl_key = sys.argv[1:5]
ckpt_path = sorted(glob.glob(ckpt_glob))[0]
print("checkpoint:", ckpt_path)

register_all_resolvers()
with initialize_config_dir(
    config_dir=str(ROOT / "configs"), version_base="1.3"
):
    cfg = compose(
        config_name="run.yaml",
        overrides=[
            "model=cell/sheaf_tsp",
            "dataset=graph/graphuniverse_inductive",
            "trainer.accelerator=cpu",
            "trainer.devices=1",
            "paths.output_dir=/tmp/angle_out",
            "paths.work_dir=/tmp/angle_out",
        ],
    )
U.apply_challenge_feature_encoder_out_channels(cfg)

setting = next(
    s
    for s in U.iter_challenge_settings()
    if (s.homophily_key, s.avg_degree_key, s.power_law_key)
    == (h_key, d_key, pl_key)
)
dm = U.build_datamodule_for_setting(cfg, setting)

import hydra as _hydra  # noqa: E402

model = _hydra.utils.instantiate(
    cfg.model,
    evaluator=cfg.evaluator,
    optimizer=cfg.optimizer,
    loss=cfg.loss,
)
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing keys:", len(missing), "| unexpected:", len(unexpected))
model.eval()

angles_per_layer = {}


def make_hook(name):
    def hook(module, args, output):
        R = output.detach()
        th = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        angles_per_layer.setdefault(name, []).append(th)

    return hook


backbone = model.backbone.backbone
for i, layer in enumerate(backbone.layers):
    layer.map_learner.register_forward_hook(make_hook(f"layer{i}"))

with torch.no_grad():
    for b, batch in enumerate(dm.test_dataloader()):
        model(batch)
        if b >= 4:
            break

print(f"\ncell {h_key}/{d_key}/{pl_key} — learned SO(2) transport angles:")
for name, chunks in angles_per_layer.items():
    th = torch.cat(chunks).numpy()
    deg = np.degrees(th)
    frac_nontrivial = float(np.mean(np.abs(deg) > 5.0))
    print(
        f"  {name}: n={len(deg)}  mean|θ|={np.abs(deg).mean():.1f}°  "
        f"std={deg.std():.1f}°  max|θ|={np.abs(deg).max():.1f}°  "
        f"frac(|θ|>5°)={frac_nontrivial:.2f}"
    )
