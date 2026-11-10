"""Task-independent TRAWL heads compatible with TBModel's loss pipeline."""

import torch
from torch import nn

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class TRAWLReadout(AbstractZeroCellReadOut):
    """Predict from graph embeddings, mean walk logits, or contextual nodes.

    Parameters
    ----------
    hidden_dim : int
        Width of node embeddings, and of graph embeddings if ``graph_dim`` is
        None.
    out_channels : int
        Number of output channels.
    task_level : str, optional
        ``"node"``, ``"node_inductive"`` or ``"graph"`` (default: "graph").
    graph_dim : int, optional
        Width of graph and walk embeddings (default: None).
    head_hidden : int, optional
        Hidden width of an MLP head; a linear head is used if None
        (default: None).
    dropout : float, optional
        Dropout probability inside the head (default: 0.0).
    aggregation : str, optional
        Graph aggregation: ``"embedding"`` or ``"walk_logits"``
        (default: "embedding").
    input_dropout : float, optional
        Dropout probability on head inputs; defaults to ``dropout`` if None
        (default: None).
    **kwargs : dict
        Additional arguments of ``AbstractZeroCellReadOut`` (e.g.
        ``readout_name``).
    """

    def __init__(
        self,
        hidden_dim,
        out_channels,
        task_level="graph",
        graph_dim=None,
        head_hidden=None,
        dropout=0.0,
        aggregation="embedding",
        input_dropout=None,
        **kwargs,
    ):
        if aggregation not in {"embedding", "walk_logits"}:
            raise ValueError("Unknown readout aggregation")
        node_level = task_level in {"node", "node_inductive"}
        width = hidden_dim if node_level else (graph_dim or hidden_dim)
        # ``self.linear`` of the base class is replaced by the head below;
        # fork the RNG so building it does not shift the head's initialization.
        with torch.random.fork_rng(devices=[]):
            super().__init__(
                hidden_dim=width,
                out_channels=out_channels,
                task_level=task_level,
                pooling_type="mean",
                **kwargs,
            )
        self.aggregation = aggregation
        input_dropout = dropout if input_dropout is None else input_dropout
        if head_hidden:
            self.linear = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(width, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_channels),
            )
        else:
            # A linear head has a single dropout, applied to its input.
            self.linear = nn.Sequential(
                nn.Dropout(input_dropout), nn.Linear(width, out_channels)
            )

    def forward(self, model_out, batch):
        """Compute logits from the backbone output.

        Parameters
        ----------
        model_out : dict
            Backbone output dictionary.
        batch : torch_geometric.data.Batch
            Input batch (unused).

        Returns
        -------
        dict
            ``model_out`` with ``logits`` added.
        """
        if self.task_level != "graph":
            logits = self.linear(model_out["x_0"])
        else:
            if self.aggregation != "embedding" and not model_out.get(
                "walk_readout_compatible", True
            ):
                raise ValueError(
                    "Use embedding aggregation with separate neighborhoods or cell readout to preserve their fusion"
                )
            if self.aggregation == "walk_logits" and len(
                model_out["walk_batch"]
            ):
                predictions = self.linear(model_out["walk_embedding"])
                shape = (
                    len(model_out["graph_embedding"]),
                    predictions.shape[-1],
                )
                sums = predictions.new_zeros(shape).index_add(
                    0, model_out["walk_batch"], predictions
                )
                count = predictions.new_zeros(shape[0], 1).index_add(
                    0,
                    model_out["walk_batch"],
                    predictions.new_ones(len(predictions), 1),
                )
                logits = sums / count.clamp_min(1)
                missing = count.squeeze(-1) == 0
                if missing.any():
                    logits[missing] = self.linear(
                        model_out["graph_embedding"][missing]
                    )
            else:
                logits = self.linear(model_out["graph_embedding"])
        model_out["logits"] = logits
        return model_out
