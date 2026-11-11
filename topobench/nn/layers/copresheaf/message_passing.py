"""Message passing with learned copresheaf transport maps."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .maps import create_copresheaf_map
from .routes import CopresheafRoute


class CopresheafMessagePassing(nn.Module):
    r"""Transport and aggregate messages along one directed neighborhood.

    This layer implements the inner aggregation in Definition 9 of the CTNN
    paper. The learned map :math:`\rho_{y\to x}` transports a source feature
    into the target stalk before the optional message function ``alpha`` and
    a permutation-invariant aggregation are applied.

    Parameters
    ----------
    channels : int
        Dimension of the source and target cell features.
    heads : int
        Number of independent stalk bundles.
    stalk_dimension : int, optional
        Dimension of one stalk. By default, ``channels // heads``.
    map_type : str
        Name of the transport map passed to ``create_copresheaf_map``.
    aggr : str
        Per-edge normalization: ``"sum"``, ``"mean"``, or ``"symmetric"``.
    message_type : str
        Message function: ``"identity"`` or an ``"mlp"`` over the
        concatenated target and transported features.
    dropout : float
        Dropout applied inside the transport map and the ``mlp``
        message function.
    map_kwargs : Mapping, optional
        Extra constructor arguments forwarded to the transport map.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        stalk_dimension: int | None = None,
        map_type: str = "diagonal",
        aggr: str = "sum",
        message_type: str = "identity",
        dropout: float = 0.0,
        map_kwargs: Mapping | None = None,
    ) -> None:
        super().__init__()
        if aggr not in {"sum", "mean", "symmetric"}:
            raise ValueError("aggr must be one of: sum, mean, symmetric")
        if message_type not in {"identity", "mlp"}:
            raise ValueError("message_type must be 'identity' or 'mlp'")

        self.channels = channels
        self.heads = heads
        self.stalk_dimension = stalk_dimension or channels // heads
        self.aggr = aggr
        self.transport = create_copresheaf_map(
            map_type,
            channels=channels,
            heads=heads,
            stalk_dimension=self.stalk_dimension,
            dropout=dropout,
            **dict(map_kwargs or {}),
        )
        self.alpha = (
            nn.Identity()
            if message_type == "identity"
            else nn.Sequential(
                nn.Linear(2 * channels, channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, channels),
            )
        )

    def _normalization(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_source: int,
        num_target: int,
    ) -> torch.Tensor:
        """Rescale edge weights according to ``self.aggr``.

        Parameters
        ----------
        edge_index : torch.Tensor
            ``[2, num_edges]`` ``(source, target)`` edge index.
        edge_weight : torch.Tensor
            One weight per edge, before normalization.
        num_source : int
            Number of source cells, used to size the degree buffer.
        num_target : int
            Number of target cells, used to size the degree buffer.

        Returns
        -------
        torch.Tensor
            One normalized weight per edge.
        """
        source, target = edge_index
        if self.aggr == "sum":
            return edge_weight

        target_degree = edge_weight.new_zeros(num_target)
        target_degree.index_add_(0, target, edge_weight)
        if self.aggr == "mean":
            return edge_weight / target_degree[target].clamp_min(1.0)

        source_degree = edge_weight.new_zeros(num_source)
        source_degree.index_add_(0, source, edge_weight)
        denominator = (source_degree[source] * target_degree[target]).sqrt()
        return edge_weight / denominator.clamp_min(1.0)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one aggregated message for every target cell.

        Parameters
        ----------
        source_features : torch.Tensor
            ``[num_source, channels]`` features of the source cells.
        target_features : torch.Tensor
            ``[num_target, channels]`` features of the target cells.
        edge_index : torch.Tensor
            ``[2, num_edges]`` ``(source, target)`` edge index.
        edge_weight : torch.Tensor, optional
            One weight per edge. By default, every edge has weight 1.

        Returns
        -------
        torch.Tensor
            ``[num_target, channels]`` aggregated message per target
            cell.
        """
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if source_features.size(-1) != self.channels:
            raise ValueError("source feature width does not match channels")
        if target_features.size(-1) != self.channels:
            raise ValueError("target feature width does not match channels")

        edge_index = edge_index.to(source_features.device).long()
        source, target = edge_index
        if edge_weight is None:
            edge_weight = source_features.new_ones(source.numel())
        else:
            edge_weight = edge_weight.to(
                device=source_features.device, dtype=source_features.dtype
            ).reshape(-1)
        if edge_weight.numel() != source.numel():
            raise ValueError("edge_weight must contain one value per edge")

        # Some higher-order batches contain padded sparse connectivity blocks
        # for ranks that are absent in one or more examples. Such structural
        # padding is not a cell and must not generate a message.
        valid = (
            (source >= 0)
            & (source < source_features.size(0))
            & (target >= 0)
            & (target < target_features.size(0))
        )
        source, target = source[valid], target[valid]
        edge_weight = edge_weight[valid]
        edge_index = torch.stack((source, target), dim=0)

        source_on_edges = source_features[source]
        target_on_edges = target_features[target]
        transport = self.transport(source_on_edges, target_on_edges)
        values = source_on_edges.view(-1, self.heads, self.stalk_dimension)
        messages = torch.matmul(transport, values.unsqueeze(-1)).squeeze(-1)
        messages = messages.reshape(-1, self.channels)
        if not isinstance(self.alpha, nn.Identity):
            messages = self.alpha(torch.cat((target_on_edges, messages), -1))

        weights = self._normalization(
            edge_index,
            edge_weight,
            source_features.size(0),
            target_features.size(0),
        )
        messages = messages * weights.unsqueeze(-1)
        output = target_features.new_zeros(
            target_features.size(0), self.channels
        )
        output.index_add_(0, target, messages)
        return output


class CopresheafUpdate(nn.Module):
    r"""Apply the CTNN update function :math:`\beta(h_x,m_x)`.

    Parameters
    ----------
    channels : int
        Dimension of the cell features.
    update_type : str
        ``"residual"`` for a gated residual update, or ``"mlp"`` for an
        MLP over the concatenated feature and message.
    dropout : float
        Dropout applied inside the update MLP.
    trainable_epsilon : bool
        If ``True``, the residual mixing coefficient ``epsilon`` is a
        learned parameter; otherwise it is a fixed buffer at zero.
    message_gate_init : float, optional
        Initial value for a learned scalar gate applied to the message
        term, passed through ``sigmoid``. By default, no gate is used
        and the message term has weight 1.
    """

    def __init__(
        self,
        channels: int,
        update_type: str = "residual",
        dropout: float = 0.0,
        trainable_epsilon: bool = True,
        message_gate_init: float | None = None,
    ) -> None:
        super().__init__()
        if update_type not in {"residual", "mlp"}:
            raise ValueError("update_type must be 'residual' or 'mlp'")
        self.update_type = update_type
        epsilon = torch.zeros(1)
        if trainable_epsilon:
            self.epsilon = nn.Parameter(epsilon)
        else:
            self.register_buffer("epsilon", epsilon)
        if message_gate_init is None:
            self.register_parameter("message_gate", None)
        else:
            self.message_gate = nn.Parameter(
                torch.tensor(float(message_gate_init))
            )
        if update_type == "residual":
            self.update = nn.Sequential(
                nn.Linear(channels, channels), nn.Dropout(dropout)
            )
        else:
            self.update = nn.Sequential(
                nn.Linear(2 * channels, channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, channels),
            )
        self.norm = nn.LayerNorm(channels)

    def forward(self, features: torch.Tensor, message: torch.Tensor):
        """Combine the current stalk feature with its aggregate message.

        Parameters
        ----------
        features : torch.Tensor
            ``[num_cells, channels]`` current stalk features.
        message : torch.Tensor
            ``[num_cells, channels]`` aggregated incoming message.

        Returns
        -------
        torch.Tensor
            ``[num_cells, channels]`` updated and layer-normalized
            features.
        """
        gate = (
            1.0
            if self.message_gate is None
            else torch.sigmoid(self.message_gate)
        )
        if self.update_type == "residual":
            update = (1.0 + self.epsilon) * features
            update = update + gate * self.update(message)
        else:
            update = features + gate * self.update(
                torch.cat((features, message), -1)
            )
        return self.norm(update)


class HigherOrderCopresheafLayer(nn.Module):
    r"""Apply Definition 10 over several rank-aware neighborhoods.

    Each neighborhood owns a transport/message module. Messages that
    arrive at the same target rank are combined by
    ``neighborhood_aggr`` and passed to one rank-specific update
    function. ``"sum"`` and ``"mean"`` are permutation-invariant;
    ``"gated"`` instead weights each route with a softmax over a
    learned per-route logit, computed separately for the routes that
    feed each target rank, so unlike sum or mean the weighting is not
    symmetric across routes.

    Parameters
    ----------
    channels : int
        Dimension of the cell features at every rank.
    neighborhoods : Sequence of str
        TopoBench neighborhood names, one per directed route.
    heads : int
        Number of independent stalk bundles.
    stalk_dimension : int, optional
        Dimension of one stalk. By default, ``channels // heads``.
    map_type : str or Mapping of str to str
        Transport map name used by every route, or a mapping from
        neighborhood name to map name.
    aggr : str
        Per-edge normalization forwarded to each
        ``CopresheafMessagePassing``: ``"sum"``, ``"mean"``, or
        ``"symmetric"``.
    neighborhood_aggr : str
        How messages from different routes are combined at a shared
        target rank: ``"sum"``, ``"mean"``, or ``"gated"``.
    message_type : str
        Message function forwarded to each route: ``"identity"`` or
        ``"mlp"``.
    update_type : str
        Update function forwarded to each rank: ``"residual"`` or
        ``"mlp"``.
    dropout : float
        Dropout forwarded to the transport maps, message functions,
        and update functions.
    map_kwargs : Mapping, optional
        Extra constructor arguments forwarded to every transport map.
    message_gate_init : float, optional
        Initial value for each rank's learned message gate. By
        default, no gate is used.
    route_self_bias : float
        Initial ``route_logits`` value for self-routes (routes whose
        source and target rank match), used only when
        ``neighborhood_aggr`` is ``"gated"``.
    message_schedule : Sequence of str, optional
        Ordered neighborhood names for sequential, rather than
        simultaneous, route updates. By default, all routes update
        simultaneously.
    """

    def __init__(
        self,
        channels: int,
        neighborhoods: Sequence[str],
        heads: int = 1,
        stalk_dimension: int | None = None,
        map_type: str | Mapping[str, str] = "diagonal",
        aggr: str = "sum",
        neighborhood_aggr: str = "sum",
        message_type: str = "identity",
        update_type: str = "residual",
        dropout: float = 0.0,
        map_kwargs: Mapping | None = None,
        message_gate_init: float | None = None,
        route_self_bias: float = 1.0,
        message_schedule: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if not neighborhoods:
            raise ValueError("at least one neighborhood is required")
        if neighborhood_aggr not in {"sum", "mean", "gated"}:
            raise ValueError(
                "neighborhood_aggr must be 'sum', 'mean', or 'gated'"
            )

        self.routes = [
            CopresheafRoute.from_neighborhood(name) for name in neighborhoods
        ]
        if len({route.neighborhood for route in self.routes}) != len(
            self.routes
        ):
            raise ValueError("neighborhood names must be unique")
        self.neighborhood_aggr = neighborhood_aggr
        self.message_schedule = (
            tuple(message_schedule) if message_schedule is not None else None
        )
        route_lookup = {
            route.neighborhood: index
            for index, route in enumerate(self.routes)
        }
        if self.message_schedule is not None:
            unknown = sorted(set(self.message_schedule) - set(route_lookup))
            if unknown:
                raise ValueError(
                    "message_schedule contains unknown neighborhoods: "
                    f"{unknown}"
                )
        self._route_lookup = route_lookup
        initial_route_logits = torch.zeros(len(self.routes))
        for index, route in enumerate(self.routes):
            if route.source_rank == route.target_rank:
                initial_route_logits[index] = route_self_bias
        if neighborhood_aggr == "gated":
            self.route_logits = nn.Parameter(initial_route_logits)
        else:
            self.register_buffer(
                "route_logits", initial_route_logits, persistent=False
            )
        self.message_passing = nn.ModuleList()
        for route in self.routes:
            route_map_type = (
                map_type[route.neighborhood]
                if isinstance(map_type, Mapping)
                else map_type
            )
            self.message_passing.append(
                CopresheafMessagePassing(
                    channels=channels,
                    heads=heads,
                    stalk_dimension=stalk_dimension,
                    map_type=route_map_type,
                    aggr=aggr,
                    message_type=message_type,
                    dropout=dropout,
                    map_kwargs=map_kwargs,
                )
            )

        ranks = sorted(
            {route.source_rank for route in self.routes}
            | {route.target_rank for route in self.routes}
        )
        self.updates = nn.ModuleDict(
            {
                str(rank): CopresheafUpdate(
                    channels,
                    update_type=update_type,
                    dropout=dropout,
                    message_gate_init=message_gate_init,
                )
                for rank in ranks
            }
        )

    def forward(
        self,
        features: Mapping[int, torch.Tensor],
        connectivities: Mapping[str, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Update every represented cell rank simultaneously.

        Parameters
        ----------
        features : Mapping of int to torch.Tensor
            Cell features keyed by rank.
        connectivities : Mapping of str to torch.Tensor
            Neighborhood matrices keyed by TopoBench neighborhood
            name, one per entry of ``neighborhoods``.

        Returns
        -------
        dict of int to torch.Tensor
            Updated cell features keyed by rank.
        """
        if self.message_schedule is not None:
            return self._forward_scheduled(features, connectivities)

        messages = defaultdict(list)
        for route_index, (route, passing) in enumerate(
            zip(self.routes, self.message_passing, strict=True)
        ):
            if route.neighborhood not in connectivities:
                raise KeyError(f"missing connectivity {route.neighborhood!r}")
            try:
                source_features = features[route.source_rank]
                target_features = features[route.target_rank]
            except KeyError as error:
                raise KeyError(
                    f"missing features for rank {error.args[0]}"
                ) from error

            edge_index, edge_weight = route.edge_index(
                connectivities[route.neighborhood]
            )
            messages[route.target_rank].append(
                (
                    route_index,
                    passing(
                        source_features,
                        target_features,
                        edge_index,
                        edge_weight,
                    ),
                )
            )

        output = {}
        for rank, rank_features in features.items():
            if messages[rank]:
                route_indices, rank_messages = zip(
                    *messages[rank], strict=True
                )
                stacked = torch.stack(rank_messages, dim=0)
                if self.neighborhood_aggr == "gated":
                    weights = torch.softmax(
                        self.route_logits[list(route_indices)], dim=0
                    )
                    message = torch.einsum("r,rnc->nc", weights, stacked)
                else:
                    message = stacked.sum(dim=0)
                    if self.neighborhood_aggr == "mean":
                        message = message / len(rank_messages)
            else:
                message = torch.zeros_like(rank_features)
            output[rank] = self.updates[str(rank)](rank_features, message)
        return output

    def _forward_scheduled(
        self,
        features: Mapping[int, torch.Tensor],
        connectivities: Mapping[str, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Apply one directed neighborhood update at a time.

        The default copresheaf layer aggregates all incoming neighborhoods for
        a target rank before applying one simultaneous update.  A schedule
        instead gives an explicit sequence of directed maps, for example
        ``0->1, 1->2, 2->1, 1->0``.  Later steps see features already updated
        by earlier steps, making ``num_layers`` the number of repeated
        schedule sweeps.

        Parameters
        ----------
        features : Mapping of int to torch.Tensor
            Cell features keyed by rank.
        connectivities : Mapping of str to torch.Tensor
            Neighborhood matrices keyed by TopoBench neighborhood
            name, one per entry of ``message_schedule``.

        Returns
        -------
        dict of int to torch.Tensor
            Updated cell features keyed by rank.
        """
        output = dict(features)
        for neighborhood in self.message_schedule or ():
            route_index = self._route_lookup[neighborhood]
            route = self.routes[route_index]
            passing = self.message_passing[route_index]
            if route.neighborhood not in connectivities:
                raise KeyError(f"missing connectivity {route.neighborhood!r}")
            try:
                source_features = output[route.source_rank]
                target_features = output[route.target_rank]
            except KeyError as error:
                raise KeyError(
                    f"missing features for rank {error.args[0]}"
                ) from error

            edge_index, edge_weight = route.edge_index(
                connectivities[route.neighborhood]
            )
            message = passing(
                source_features,
                target_features,
                edge_index,
                edge_weight,
            )
            output[route.target_rank] = self.updates[str(route.target_rank)](
                target_features, message
            )
        return output

    def route_weights(self) -> dict[str, float]:
        """Return current per-target normalized route weights for logging.

        Returns
        -------
        dict of str to float
            Softmax-normalized ``route_logits`` weight for every
            neighborhood, or an empty dict when ``neighborhood_aggr``
            is not ``"gated"`` or a ``message_schedule`` is set.
        """
        if (
            self.message_schedule is not None
            or self.neighborhood_aggr != "gated"
        ):
            return {}
        weights = {}
        target_ranks = sorted({route.target_rank for route in self.routes})
        for rank in target_ranks:
            indices = [
                index
                for index, route in enumerate(self.routes)
                if route.target_rank == rank
            ]
            normalized = torch.softmax(self.route_logits[indices], dim=0)
            for index, weight in zip(indices, normalized, strict=True):
                weights[self.routes[index].neighborhood] = float(
                    weight.detach().cpu()
                )
        return weights
