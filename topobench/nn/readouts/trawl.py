"""Task-independent TRAWL heads compatible with TBModel's loss pipeline."""

from torch import nn


class TRAWLReadout(nn.Module):
    """Predict from graph embeddings, mean walk logits, or contextual nodes.

    Parameters
    ----------
    hidden_dim : int
        Width of node embeddings, and of graph embeddings if ``graph_dim`` is
        None.
    out_channels : int
        Number of output channels.
    task_level : str, optional
        ``"node"`` or ``"graph"`` (default: "graph").
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
    ):
        super().__init__()
        if task_level not in {"node", "graph"}:
            raise ValueError("TRAWL supports TopoBench's node and graph tasks")
        if aggregation not in {"embedding", "walk_logits"}:
            raise ValueError("Unknown readout aggregation")
        self.task_level, self.aggregation = task_level, aggregation
        width = (
            hidden_dim if task_level == "node" else (graph_dim or hidden_dim)
        )
        input_dropout = dropout if input_dropout is None else input_dropout
        if head_hidden:
            self.head = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(width, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_channels),
            )
        else:
            # A linear head has a single dropout, applied to its input.
            self.head = nn.Sequential(
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
        if self.task_level == "node":
            logits = self.head(model_out["x_0"])
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
                predictions = self.head(model_out["walk_embedding"])
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
                    logits[missing] = self.head(
                        model_out["graph_embedding"][missing]
                    )
            else:
                logits = self.head(model_out["graph_embedding"])
        model_out["logits"] = logits
        return model_out
