"""TopoBench: A library for benchmarking of topological models."""

# torch >= 2.6 defaults to weights_only=True in torch.load, but OGB and
# older PyG code serialize these classes. Register them as safe so that
# torch.load works without weights_only=False everywhere.
import torch

if hasattr(torch.serialization, "add_safe_globals"):
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import (
        EdgeStorage,
        GlobalStorage,
        NodeStorage,
    )

    torch.serialization.add_safe_globals(
        [
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage,
            NodeStorage,
            EdgeStorage,
        ]
    )

# Import submodules
from . import (
    data,
    dataloader,
    evaluator,
    loss,
    model,
    nn,
    transforms,
    utils,
)
from .run import initialize_hydra

__all__ = [
    "data",
    "dataloader",
    "evaluator",
    "initialize_hydra",
    "loss",
    "model",
    "nn",
    "transforms",
    "utils",
]


__version__ = "0.0.1"
