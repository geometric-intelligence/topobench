"""Explicit checkpoint averaging and prediction ensembles for recipe studies."""

import copy

import torch
from torch import nn


def average_checkpoints(paths):
    """Average floating parameters/buffers; keep integer buffers from best.

    Paths must be ordered best-first by the validation checkpoint callback.
    Keys, dtypes and shapes must agree. This never averages optimizer states.

    Parameters
    ----------
    paths : list of str or pathlib.Path
        Lightning checkpoint files, ordered best-first.

    Returns
    -------
    dict
        Averaged model ``state_dict``; non-floating tensors are taken from
        the first (best) checkpoint.
    """
    if not paths:
        raise ValueError("No checkpoints available to average")
    result = None
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        if result is None:
            result = {key: value.clone() for key, value in state.items()}
            continue
        if state.keys() != result.keys():
            raise ValueError("Checkpoint keys differ")
        for key, value in state.items():
            if (
                value.shape != result[key].shape
                or value.dtype != result[key].dtype
            ):
                raise ValueError(f"Checkpoint tensor mismatch: {key}")
            if value.is_floating_point() or value.is_complex():
                result[key].add_(value)
    for value in result.values():
        if value.is_floating_point() or value.is_complex():
            value.div_(len(paths))
    return result


class PredictionEnsemble(nn.Module):
    """Run complete model pipelines and average logits before loss/metrics.

    Parameters
    ----------
    model : TBModel
        Model whose feature encoder, backbone and readout are copied for
        each ensemble member.
    paths : list of str or pathlib.Path
        Lightning checkpoint files, one per ensemble member.
    """

    def __init__(self, model, paths):
        super().__init__()
        members = []
        for path in paths:
            # Copy only neural components, not Lightning trainer/logger state.
            pipeline = nn.ModuleDict(
                {
                    "encoder": copy.deepcopy(model.feature_encoder),
                    "backbone": copy.deepcopy(model.backbone),
                    "readout": copy.deepcopy(model.readout),
                }
            )
            state = torch.load(path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
            for destination, source in [
                ("encoder", "feature_encoder"),
                ("backbone", "backbone"),
                ("readout", "readout"),
            ]:
                prefix = source + "."
                pipeline[destination].load_state_dict(
                    {
                        key[len(prefix) :]: value
                        for key, value in state.items()
                        if key.startswith(prefix)
                    }
                )
            members.append(pipeline)
        self.members = nn.ModuleList(members)

    def forward(self, batch):
        """Run every member pipeline and average their logits.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Input batch; each member receives its own clone.

        Returns
        -------
        dict
            Output of the last member with ``logits`` replaced by the mean
            logits across members.
        """
        predictions = []
        for pipeline in self.members:
            member_batch = batch.clone()
            output = pipeline["backbone"](pipeline["encoder"](member_batch))
            output = pipeline["readout"](output, member_batch)
            predictions.append(output["logits"])
        output["logits"] = torch.stack(predictions).mean(dim=0)
        return output


class PassThroughReadout(nn.Module):
    """Keep the TBModel task interface when logits were produced by an ensemble.

    Parameters
    ----------
    task_level : str
        Task level exposed to ``TBModel`` (e.g. ``"graph"`` or ``"node"``).
    """

    def __init__(self, task_level):
        super().__init__()
        self.task_level = task_level

    def forward(self, model_out, batch):
        """Return the model output unchanged.

        Parameters
        ----------
        model_out : dict
            Output of the ensemble, already containing ``logits``.
        batch : torch_geometric.data.Data
            Input batch (unused).

        Returns
        -------
        dict
            The unchanged ``model_out``.
        """
        return model_out
