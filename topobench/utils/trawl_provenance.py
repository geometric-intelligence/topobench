"""Portable run manifests for auditing ablations and reproduction attempts."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
import torch_geometric
from omegaconf import OmegaConf


def tensor_digest(tensors):
    """Hash ordered named tensors, including shape and dtype metadata.

    Parameters
    ----------
    tensors : iterable of tuple
        ``(name, value)`` pairs; values that are not tensors are skipped.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for name, value in tensors:
        if not isinstance(value, torch.Tensor):
            continue
        value = value.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_dense()
        digest.update(
            str((name, tuple(value.shape), str(value.dtype))).encode()
        )
        digest.update(
            value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def write_run_manifest(config, model, datamodule):
    """Save resolved configuration, input-order hashes and numerical environment.

    Parameters
    ----------
    config : DictConfig
        Run configuration; the manifest is written to ``paths.output_dir``.
    model : torch.nn.Module
        Model whose initial ``state_dict`` is hashed.
    datamodule : TBDataloader
        Data module whose train, validation and test datasets are hashed.

    Returns
    -------
    pathlib.Path
        Path of the written ``trawl_manifest.json``.
    """
    splits = {}
    for name in ("train", "val", "test"):
        dataset = getattr(datamodule, f"dataset_{name}")
        if dataset is not None:
            tensors = (
                (f"{i}/{key}", data[key])
                for i, data in enumerate(dataset.data_lst)
                for key in sorted(data.keys())
                if key != "model_state"
            )
            splits[name] = {
                "count": len(dataset),
                "sha256": tensor_digest(tensors),
            }
    versions = {}
    for package in (
        "torch",
        "torch-geometric",
        "pyg-nightly",
        "toponetx",
        "topomodelx",
        "scikit-learn",
        "lightning",
        "numpy",
        "scipy",
        "networkx",
        "mamba-ssm",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema_version": 1,
        "config": OmegaConf.to_container(config, resolve=True),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_geometric_runtime_version": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "splits": splits,
        "supervised_initial_state_sha256": tensor_digest(
            model.state_dict().items()
        ),
    }
    package = Path(__file__).resolve().parents[1]
    sources = set(package.rglob("trawl*.py"))
    sources.update((package / "data/utils/trawl").glob("*.py"))
    sources.update(
        package / name
        for name in (
            "evaluator/checkpoint.py",
            "evaluator/evaluator.py",
            "callbacks/model_checkpoint.py",
            "run.py",
            "model/model.py",
            "dataloader/samplers.py",
        )
    )
    manifest["implementation_sha256"] = {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(sources)
    }
    destination = Path(config.paths.output_dir) / "trawl_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return destination
