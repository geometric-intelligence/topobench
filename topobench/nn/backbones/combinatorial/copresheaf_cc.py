"""Higher-order copresheaf message passing on combinatorial complexes.

Implements Copresheaf Topological Neural Networks (CTNNs) [1].

[1] Hajij et al. "Copresheaf Topological Neural Networks: A Generalized
Deep Learning Framework." Advances in Neural Information Processing
Systems (NeurIPS), 2025.
https://proceedings.neurips.cc/paper_files/paper/2025/file/dc62cd4ec77f3bbec8e6245d0bd91d08-Paper-Conference.pdf
"""

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from topobench.nn.layers.copresheaf import (
    CopresheafRoute,
    HigherOrderCopresheafLayer,
)


class CopresheafCC(nn.Module):
    r"""A rank-aware Copresheaf Topological Neural Network.

    The configured TopoBench neighborhoods induce directed graphs on cells.
    Every route gets its own learned maps :math:`\rho_{y\to x}` and message
    function, while messages arriving at the same rank are combined before a
    rank-specific update. This realizes the message-passing form in
    Definitions 9--10 of Copresheaf Topological Neural Networks [1]
    (NeurIPS 2025).

    Parameters
    ----------
    in_channels : int
        Feature dimension shared by all input ranks.
    hidden_channels : int
        Hidden stalk feature dimension.
    out_channels : int
        Output feature dimension for every represented rank.
    neighborhoods : sequence of str
        TopoBench neighborhood names, for example ``up_incidence-0``.
    num_layers : int, optional
        Number of stacked ``HigherOrderCopresheafLayer`` blocks. Must be
        at least one (default: 3).
    heads : int, optional
        Number of independent stalk bundles per route. Mutually
        exclusive with ``num_heads``; leave at the default when passing
        ``num_heads`` instead (default: 1).
    stalk_dimension : int, optional
        Dimension of one per-head stalk. By default,
        ``hidden_channels // heads``.
    map_type : str or mapping
        One map family for all routes or a neighborhood-to-family mapping.
    aggr : str, optional
        Per-edge normalization used by every route's transport:
        ``"sum"``, ``"mean"``, or ``"symmetric"`` (default: "mean").
    neighborhood_aggr : str, optional
        How messages from multiple routes landing on the same rank are
        combined: ``"sum"``, ``"mean"``, or ``"gated"`` for a learned
        softmax over routes (default: "sum").
    message_type : str, optional
        Message function applied per route: ``"identity"`` or ``"mlp"``
        over the concatenated target and transported features
        (default: "identity").
    update_type : str, optional
        Per-rank update function: ``"residual"`` for a gated residual
        update, or ``"mlp"`` over the concatenated feature and message
        (default: "residual").
    dropout : float, optional
        Dropout applied inside the transport maps, message functions,
        and update functions (default: 0.0).
    map_kwargs : mapping, optional
        Extra constructor arguments forwarded to each route's transport
        map (default: None).
    num_heads : int, optional
        Alternative spelling of ``heads``. Passing both ``num_heads``
        and a non-default ``heads`` raises ``ValueError``
        (default: None).
    message_gate_init : float, optional
        Initial value for a learned scalar gate applied to each
        update's message term, passed through ``sigmoid``. By default,
        no gate is used and the message term has weight 1.
    route_self_bias : float, optional
        Initial ``neighborhood_aggr="gated"`` logit bias favoring
        routes whose source and target ranks match (default: 1.0).
    message_schedule : sequence of str, optional
        If provided, apply neighborhood updates sequentially in this order
        inside each layer instead of aggregating all incoming routes
        simultaneously.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        neighborhoods: Sequence[str],
        num_layers: int = 3,
        heads: int = 1,
        stalk_dimension: int | None = None,
        map_type: str | Mapping[str, str] = "shared_local",
        aggr: str = "mean",
        neighborhood_aggr: str = "sum",
        message_type: str = "identity",
        update_type: str = "residual",
        dropout: float = 0.0,
        map_kwargs: Mapping | None = None,
        num_heads: int | None = None,
        message_gate_init: float | None = None,
        route_self_bias: float = 1.0,
        message_schedule: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if num_heads is not None:
            if heads != 1:
                raise ValueError("pass heads or num_heads, not both")
            heads = num_heads

        self.neighborhoods = list(neighborhoods)
        self.message_schedule = (
            tuple(message_schedule) if message_schedule is not None else None
        )
        self.routes = [
            CopresheafRoute.from_neighborhood(name)
            for name in self.neighborhoods
        ]
        self.ranks = sorted(
            {route.source_rank for route in self.routes}
            | {route.target_rank for route in self.routes}
        )
        self.input_projections = nn.ModuleDict(
            {
                str(rank): nn.Linear(in_channels, hidden_channels)
                for rank in self.ranks
            }
        )
        self.layers = nn.ModuleList(
            [
                HigherOrderCopresheafLayer(
                    channels=hidden_channels,
                    neighborhoods=self.neighborhoods,
                    heads=heads,
                    stalk_dimension=stalk_dimension,
                    map_type=map_type,
                    aggr=aggr,
                    neighborhood_aggr=neighborhood_aggr,
                    message_type=message_type,
                    update_type=update_type,
                    dropout=dropout,
                    map_kwargs=map_kwargs,
                    message_gate_init=message_gate_init,
                    route_self_bias=route_self_bias,
                    message_schedule=message_schedule,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projections = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_channels, out_channels)
                for rank in self.ranks
            }
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: Mapping[int, torch.Tensor],
        connectivities: Mapping[str, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Encode features on all configured cell ranks.

        Parameters
        ----------
        features : mapping of int to torch.Tensor
            Input features keyed by cell rank.
        connectivities : mapping of str to torch.Tensor
            Batched connectivity tensors keyed by neighborhood name.

        Returns
        -------
        dict of int to torch.Tensor
            Output features keyed by cell rank.
        """
        missing_ranks = set(self.ranks) - set(features)
        if missing_ranks:
            raise KeyError(
                f"missing features for ranks {sorted(missing_ranks)}"
            )

        hidden = {
            rank: self.dropout(
                self.activation(
                    self.input_projections[str(rank)](features[rank])
                )
            )
            for rank in self.ranks
        }
        for layer in self.layers:
            hidden = layer(hidden, connectivities)
            hidden = {
                rank: self.dropout(self.activation(value))
                for rank, value in hidden.items()
            }
        return {
            rank: self.output_projections[str(rank)](value)
            for rank, value in hidden.items()
        }
