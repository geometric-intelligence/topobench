"""Modal script to run the DirSNN TDL Challenge 2026 evaluation grid.

Supports hyperparameter override sweeps.

Each call to run_evaluation() can take a list of Hydra-style override
strings (e.g. ["model.backbone.dropout=0.65"]), passed straight through to
run_challenge_grid's own extra_overrides parameter -- no changes to
utils.py or the competition notebook are made or needed.

Usage (single run, default hyperparameters):
    modal deploy modal_eval.py
    python3 -c "
    import modal
    f = modal.Function.from_name('dirsnn-tdl-evaluation', 'run_evaluation')
    call = f.spawn()
    print(call.object_id)
    "

Usage (parallel sweep, multiple independent hyperparameter variants):
    modal deploy modal_eval.py
    python3 sweep_launch.py
"""

import modal

app = modal.App("dirsnn-tdl-evaluation")

dataset_volume = modal.Volume.from_name(
    "dirsnn-datasets", create_if_missing=True
)

results_volume = modal.Volume.from_name(
    "dirsnn-results", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.3.0",
        "numpy<2.0",
        "scipy",
        "networkx",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "requests",
        "decorator",
        "trimesh",
        "spharapy",
        "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0",
        "hydra-optuna-sweeper==1.2.0",
        "yacs==0.1.8",
        "wandb",
        "tensorboard",
        "einops==0.7.0",
        "tabulate",
        "rich",
        "ogb",
        "rootutils",
        "lightning==2.4.0",
        "hypernetx<2.0.0",
        "PyTDC==1.1.15",
        "setuptools>=69,<82",
    )
    .pip_install(
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    )
    .pip_install("torch-geometric")
    .pip_install(
        "git+https://github.com/pyt-team/TopoModelX.git",
        "git+https://github.com/pyt-team/TopoNetX.git@c378925",
        "graph-universe==0.1.2",
    )
    .run_commands(
        "git clone --branch dirsnn-implementation "
        "https://github.com/Grolds-Code/topobench.git /root/topobench"
    )
    .workdir("/root/topobench")
    .run_commands("pip install -e .")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={
        "/root/topobench/datasets": dataset_volume,
        "/root/results": results_volume,
    },
    secrets=[modal.Secret.from_dict({"WANDB_MODE": "offline"})],
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def run_evaluation(
    extra_overrides: list[str] | None = None,
    tag: str = "default",
    experiment_modes: list[tuple[str, str, str, list[str]]] | None = None,
):
    """Run the sanity check and full DirSNN evaluation grid.

    Parameters
    ----------
    extra_overrides : list[str] | None
        Hydra-style override strings applied globally to every run,
        e.g. ["model.backbone.dropout=0.65"]. Passed straight through to
        run_challenge_grid's own parameter. None runs with the committed
        dirsnn.yaml defaults, unmodified.
    tag : str
        Human-readable label for this variant, used only to name the
        persisted results file so multiple parallel runs don't collide
        and are easy to identify afterward.
    experiment_modes : list[tuple] | None
        Optional per-task override structure, matching run_challenge_grid's
        own (mode_name, wandb_project, dataset_group, mode_overrides)
        tuple format. Lets different tasks (community_detection,
        triangle_counting) use different Hydra overrides -- e.g. a
        different num_layers per task -- via the framework's own
        documented mechanism, not a workaround.
        None uses run_challenge_grid's own default modes.

    Returns
    -------
    str
        The results.json contents as text.
    """
    import sys

    sys.path.insert(0, "/root/topobench")
    sys.path.insert(0, "/root/topobench/2026_tdl_challenge")

    from pathlib import Path

    from utils import (
        check_challenge_grid,
        resolve_project_root,
        run_challenge_grid,
        save_challenge_artifacts,
    )

    PROJECT_ROOT = resolve_project_root(Path("/root/topobench").resolve())
    MODEL_CONFIG = "simplicial/dirsnn"

    print(f"[{tag}] Running sanity check with overrides: {extra_overrides}")
    check_challenge_grid(
        project_root=PROJECT_ROOT,
        model_config=MODEL_CONFIG,
        extra_overrides=extra_overrides,
        quiet=True,
    )
    print(f"[{tag}] Sanity check PASSED. Starting full evaluation grid...")

    kwargs = dict(
        project_root=PROJECT_ROOT,
        model_config=MODEL_CONFIG,
        extra_overrides=extra_overrides,
    )
    if experiment_modes is not None:
        kwargs["experiment_modes"] = experiment_modes

    results, study_id = run_challenge_grid(**kwargs)

    output_paths = save_challenge_artifacts(
        results,
        model_config=MODEL_CONFIG,
        study_id=study_id,
    )

    import shutil

    dest = f"/root/results/results_{tag}_{study_id}.json"
    shutil.copy(output_paths["json"], dest)
    results_volume.commit()

    print(f"[{tag}] DONE. Persisted copy at: {dest}")

    with open(output_paths["json"]) as f:
        results_json_text = f.read()

    return results_json_text
