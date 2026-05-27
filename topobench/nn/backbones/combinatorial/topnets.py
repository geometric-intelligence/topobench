"""Track-2 TopNets backbone for combinatorial-complex inputs.

This module adapts the TopNets learned-filtration dynamics to TopoBench's
combinatorial domain.  A graph-to-combinatorial lifting supplies Hasse-style
neighborhoods between cells of different ranks; each selected neighborhood is
processed with a TopNets graph-filtration route operator, and route outputs are
aggregated back to rank-wise cell embeddings.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods
from topobench.nn.backbones.graph.topnets import _TopNetsRouteOperator


class CombinatorialTopNetsBackbone(nn.Module):
    """TopNets dynamics on combinatorial-complex neighborhoods.

    Parameters
    ----------
    in_channels : int
        Input feature dimension after ``AllCellFeatureEncoder``.
    hidden_channels : int, optional
        Hidden cell embedding dimension. Defaults to ``in_channels``.
    neighborhoods : list[str], optional
        Neighborhood names provided by the combinatorial lifting.
    ranks : tuple[int, ...] | list[int], optional
        Rank-wise fallback API. If ``neighborhoods`` is omitted, each rank is
        processed through its ``up_adjacency-{rank}`` neighborhood.
    num_layers : int, optional
        Number of rank-wise neighborhood aggregation layers.
    num_steps : int, optional
        Number of fixed TopNets dynamics steps inside each route operator.
    gnn_type : str, optional
        Route message-passing layer type, either ``"gcn"`` or ``"gin"``.
    num_filtrations : int, optional
        Number of learned filtration functions per route operator.
    filtration_hidden : int, optional
        Hidden dimension of each route filtration MLP.
    coord_fun_count : int, optional
        Number of coordinates per TOGL coordinate function.
    dropout : float, optional
        Dropout in route graph-convolution blocks.
    use_dim1 : bool, optional
        Whether route operators include cycle-style topology summaries.
    sigmoid_filtrations : bool, optional
        Whether learned filtration values are sigmoid-constrained.
    activation : str, optional
        Rank aggregation activation. Supports ``relu``, ``elu``, ``tanh``,
        ``sigmoid``, and ``id``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        neighborhoods: list[str] | None = None,
        ranks: tuple[int, ...] | list[int] | None = None,
        num_layers: int = 2,
        num_steps: int = 2,
        gnn_type: str = "gcn",
        num_filtrations: int = 6,
        filtration_hidden: int = 16,
        coord_fun_count: int = 2,
        dropout: float = 0.0,
        use_dim1: bool = True,
        sigmoid_filtrations: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            msg = "num_layers must be positive"
            raise ValueError(msg)
        if neighborhoods:
            self.neighborhoods = list(neighborhoods)
            self.routes = [
                tuple(route)
                for route in get_routes_from_neighborhoods(self.neighborhoods)
            ]
        elif ranks:
            self.neighborhoods = [f"up_adjacency-{rank}" for rank in ranks]
            self.routes = [(int(rank), int(rank)) for rank in ranks]
        else:
            msg = "neighborhoods or ranks must contain at least one route"
            raise ValueError(msg)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels or in_channels
        self.out_channels = self.hidden_channels
        self.max_rank = max(max(route) for route in self.routes)
        self.num_layers = num_layers
        self.activation = _make_activation(activation)

        route_layers = []
        for layer_idx in range(num_layers):
            route_in_channels = in_channels if layer_idx == 0 else self.hidden_channels
            route_layers.append(
                nn.ModuleList(
                    [
                        _TopNetsRouteOperator(
                            in_channels=route_in_channels,
                            hidden_channels=self.hidden_channels,
                            num_steps=num_steps,
                            gnn_type=gnn_type,
                            num_filtrations=num_filtrations,
                            filtration_hidden=filtration_hidden,
                            coord_fun_count=coord_fun_count,
                            dropout=dropout,
                            use_dim1=use_dim1,
                            sigmoid_filtrations=sigmoid_filtrations,
                        )
                        for _ in self.routes
                    ]
                )
            )
        self.route_layers = nn.ModuleList(route_layers)

    def forward(self, batch: Data) -> dict[int, torch.Tensor]:
        """Return updated cell embeddings keyed by rank."""
        x_by_rank = {}
        rank = 0
        while hasattr(batch, f"x_{rank}"):
            x_by_rank[rank] = getattr(batch, f"x_{rank}")
            rank += 1
        membership = {
            rank: self._membership(batch, rank, x.device)
            for rank, x in x_by_rank.items()
        }

        for layer_routes in self.route_layers:
            route_outputs: dict[int, list[torch.Tensor]] = {}
            for route_idx, (src_rank, dst_rank) in enumerate(self.routes):
                if src_rank not in x_by_rank or dst_rank not in x_by_rank:
                    continue
                neighborhood = self.neighborhoods[route_idx]
                route_model = layer_routes[route_idx]
                if src_rank == dst_rank:
                    x_out = self._intrarank_forward(
                        batch=batch,
                        route_model=route_model,
                        neighborhood=neighborhood,
                        rank=src_rank,
                        x=x_by_rank[src_rank],
                        batch_vector=membership[src_rank],
                    )
                else:
                    x_out = self._interrank_forward(
                        batch=batch,
                        route_model=route_model,
                        neighborhood=neighborhood,
                        src_x=x_by_rank[src_rank],
                        dst_x=x_by_rank[dst_rank],
                        src_batch=membership[src_rank],
                        dst_batch=membership[dst_rank],
                    )
                route_outputs.setdefault(dst_rank, []).append(x_out)

            for rank, outputs in route_outputs.items():
                x_by_rank[rank] = self.activation(torch.stack(outputs).sum(dim=0))

        return {
            rank: x_by_rank[rank]
            for rank in range(self.max_rank + 1)
            if rank in x_by_rank
        }

    def _intrarank_forward(
        self,
        *,
        batch: Data,
        route_model: _TopNetsRouteOperator,
        neighborhood: str,
        rank: int,
        x: torch.Tensor,
        batch_vector: torch.Tensor,
    ) -> torch.Tensor:
        matrix = _rank_neighborhood_matrix(batch, neighborhood, rank)
        edge_index = matrix.indices().to(device=x.device, dtype=torch.long)
        edge_weight = matrix.values().to(device=x.device, dtype=x.dtype)
        if edge_weight.dim() != 1:
            edge_weight = None
        return route_model(
            x=x,
            edge_index=edge_index,
            batch=batch_vector,
            edge_weight=edge_weight,
        )

    def _interrank_forward(
        self,
        *,
        batch: Data,
        route_model: _TopNetsRouteOperator,
        neighborhood: str,
        src_x: torch.Tensor,
        dst_x: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
    ) -> torch.Tensor:
        matrix = batch[neighborhood].coalesce()
        edge_index = _interrank_edge_index(
            matrix=matrix,
            num_dst=dst_x.shape[0],
            device=dst_x.device,
        )
        if edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        expanded_x = torch.cat([torch.zeros_like(dst_x), src_x], dim=0)
        expanded_batch = torch.cat([dst_batch, src_batch], dim=0)
        expanded_out = route_model(
            x=expanded_x,
            edge_index=edge_index,
            batch=expanded_batch,
        )
        return expanded_out[: dst_x.shape[0]]

    @staticmethod
    def _membership(batch: Data, rank: int, device: torch.device) -> torch.Tensor:
        batch_key = f"batch_{rank}"
        if hasattr(batch, batch_key):
            return getattr(batch, batch_key).to(device=device, dtype=torch.long)

        x = getattr(batch, f"x_{rank}")
        if not hasattr(batch, "cell_statistics"):
            return torch.zeros(x.shape[0], dtype=torch.long, device=device)

        stats = batch.cell_statistics.to(device=device)
        if rank >= stats.shape[1]:
            return torch.zeros(x.shape[0], dtype=torch.long, device=device)

        graph_ids = torch.arange(stats.shape[0], device=device, dtype=torch.long)
        return torch.repeat_interleave(graph_ids, stats[:, rank].to(torch.long))


def _rank_neighborhood_matrix(
    batch: Data,
    neighborhood: str,
    rank: int,
) -> torch.Tensor:
    if neighborhood in batch:
        return batch[neighborhood].coalesce()
    if rank == 0 and "incidence_1" in batch:
        return _node_adjacency_from_incidence(batch["incidence_1"].coalesce())
    msg = f"Missing combinatorial neighborhood: {neighborhood}"
    raise KeyError(msg)


def _node_adjacency_from_incidence(incidence: torch.Tensor) -> torch.Tensor:
    indices = incidence.indices().to(dtype=torch.long)
    num_nodes = incidence.shape[0]
    if indices.numel() == 0:
        return torch.sparse_coo_tensor(
            indices.new_empty((2, 0)),
            incidence.values().new_empty((0,)),
            (num_nodes, num_nodes),
            device=incidence.device,
        ).coalesce()

    rows: list[int] = []
    cols: list[int] = []
    node_ids, edge_ids = indices[0], indices[1]
    for edge_id in edge_ids.unique(sorted=True).tolist():
        nodes = node_ids[edge_ids == edge_id].unique(sorted=True)
        if nodes.numel() < 2:
            continue
        src = nodes.repeat_interleave(nodes.numel())
        dst = nodes.repeat(nodes.numel())
        mask = src != dst
        rows.extend(src[mask].tolist())
        cols.extend(dst[mask].tolist())

    if not rows:
        edge_index = indices.new_empty((2, 0))
        values = incidence.values().new_empty((0,))
    else:
        edge_index = torch.tensor(
            [rows, cols],
            dtype=torch.long,
            device=incidence.device,
        )
        values = incidence.values().new_ones(edge_index.shape[1])
    return torch.sparse_coo_tensor(
        edge_index,
        values,
        (num_nodes, num_nodes),
        device=incidence.device,
    ).coalesce()


def _interrank_edge_index(
    matrix: torch.Tensor,
    num_dst: int,
    device: torch.device,
) -> torch.Tensor:
    indices = matrix.indices().to(device=device, dtype=torch.long)
    if indices.numel() == 0:
        return indices.new_empty((2, 0))
    dst_nodes = indices[0]
    src_nodes = indices[1] + num_dst
    return torch.stack([src_nodes, dst_nodes], dim=0)


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "id":
        return nn.Identity()
    msg = f"Unsupported activation: {name}"
    raise ValueError(msg)
