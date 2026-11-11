"""Task-agnostic readout that exposes all cell ranks to graph tasks."""

from __future__ import annotations

import torch
from torch_geometric.utils import scatter

from topobench.nn.readouts.propagate_signal_down import PropagateSignalDown


class AllRankReadout(PropagateSignalDown):
    r"""Readout for shared node-level and graph-level higher-order models.

    Node-level tasks use the standard top-down propagation path and predict
    from rank-0 features. Graph-level tasks first run the same propagation, then
    pool every available rank separately and predict from the concatenated graph
    representation. This keeps one readout class across tasks while avoiding a
    graph-level bottleneck where all higher-order signal must be compressed into
    nodes before pooling.

    Parameters
    ----------
    graph_hidden_dim : int, optional
        Hidden dimension of the graph-level MLP. Defaults to the
        propagation hidden dimension when not given.
    graph_dropout : float, optional
        Dropout applied inside the graph-level MLP (default: 0.0).
    **kwargs : dict
        Additional keyword arguments forwarded to
        ``PropagateSignalDown``, from which this class also reads
        ``num_cell_dimensions`` and ``out_channels``.
    """

    def __init__(
        self,
        graph_hidden_dim: int | None = None,
        graph_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_cell_dimensions = kwargs["num_cell_dimensions"]
        self.graph_hidden_dim = graph_hidden_dim or self.hidden_dim
        self.graph_dropout = graph_dropout

        if self.task_level == "graph":
            self.graph_mlp = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hidden_dim * self.num_cell_dimensions,
                    self.graph_hidden_dim,
                ),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.graph_dropout),
                torch.nn.Linear(self.graph_hidden_dim, kwargs["out_channels"]),
            )

    def __call__(self, model_out: dict, batch) -> dict:
        """Run propagation and compute logits for the configured task level.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Updated ``model_out``, with ``logits`` set and, for
            graph-level tasks, ``graph_features`` and
            ``readout_rank_pool_norms`` added.
        """
        model_out = self.forward(model_out, batch)
        if self.task_level != "graph":
            self.last_rank_pool_norms = None
            if model_out.get("logits", None) is None:
                model_out["logits"] = self.compute_logits(
                    model_out["x_0"], batch["batch_0"]
                )
            return model_out

        rank_pools = []
        rank_pool_norms = {}
        num_graphs = self._num_graphs(batch)
        for rank in range(self.num_cell_dimensions):
            features = model_out.get(f"x_{rank}")
            batch_index = batch.get(f"batch_{rank}", None)
            if features is None or batch_index is None:
                continue
            pooled = scatter(
                features,
                batch_index,
                dim=0,
                dim_size=num_graphs,
                reduce=self.pooling_type,
            )
            rank_pools.append(pooled)
            rank_pool_norms[str(rank)] = float(
                pooled.detach().norm(dim=-1).mean().cpu()
            )

        if len(rank_pools) != self.num_cell_dimensions:
            available = ", ".join(sorted(model_out))
            msg = (
                "AllRankReadout expected features and batch indices for "
                f"{self.num_cell_dimensions} ranks; found {len(rank_pools)}. "
                f"Available model_out keys: {available}"
            )
            raise ValueError(msg)

        graph_features = torch.cat(rank_pools, dim=-1)
        self.last_rank_pool_norms = rank_pool_norms
        model_out["graph_features"] = graph_features
        model_out["readout_rank_pool_norms"] = rank_pool_norms
        model_out["logits"] = self.graph_mlp(graph_features)
        return model_out

    @staticmethod
    def _num_graphs(batch) -> int:
        """Infer batch size from rank-0 membership.

        Higher ranks may be absent for some graphs, for example when an
        explicit triangle lift sees a triangle-free graph. Rank 0 is the
        stable source of graph membership for graph-level readout pooling.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        int
            Number of graphs represented in the batch.
        """
        batch_0 = batch.get("batch_0", None)
        if batch_0 is not None and batch_0.numel():
            return int(batch_0.max().item()) + 1
        num_graphs = getattr(batch, "num_graphs", None)
        if isinstance(num_graphs, int) and num_graphs > 0:
            return num_graphs
        labels = batch.get("y", None)
        if labels is not None and labels.ndim > 0:
            return int(labels.size(0))
        return 1
