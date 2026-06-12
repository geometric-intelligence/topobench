"""TopoBench-native ETNN backbone for combinatorial complexes.

This module implements a coordinate-free TopoBench adaptation of
E(n)-Equivariant Topological Neural Networks (ETNNs) from Battiloro et al.,
``E(n) Equivariant Topological Neural Networks``, arXiv:2405.15429, and the
official implementation at
``https://github.com/NSAPH-Projects/topological-equivariant-networks``.

The original ETNN layer combines two coupled updates:

1. A combinatorial-complex feature update over neighborhood functions.
2. An E(n)-equivariant coordinate update over geometric node features.

GraphUniverse datasets used in the TDL Challenge do not provide physical
coordinates. Therefore, this backbone implements the ETNN feature update over
TopoBench neighborhoods and intentionally omits the coordinate update. The
result is a combinatorial ETNN backbone for coordinate-free inputs. It should
not be read as a full E(n)-equivariant model unless coordinates and coordinate
updates are supplied by a future extension.

The feature update implemented here can be read as the following plain-text
version of the CCMPN/ETNN update from the paper. For a receiver cell c, a
configured neighborhood N, sender cells d in N(c), receiver rank r, and sender
rank s:

    m_{c,N} = sum_{d in N(c)} psi_{N,r,s}(h_c, h_d, a_{d,c,N})
    m_c     = concat_{N in CN(c)} m_{c,N}
    h'_c    = h_c + beta_r(h_c, m_c)

Here ``h_c`` and ``h_d`` are cell features, ``a_{d,c,N}`` is the scalar sparse
TopoBench neighborhood value, ``psi`` is a relation-specific message MLP,
``sum`` is the intra-neighborhood aggregation, ``concat`` is the
inter-neighborhood aggregation over relation types, and ``beta`` is the
rank-specific update MLP. This corresponds to the feature part of ETNN Eq. 6
and the CCMPN neighborhood aggregation described by Eq. 1--3. The coordinate
update from Eq. 7 is omitted because no Euclidean coordinates are available in
the challenge data.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import torch
from torch import nn

from topobench.data.utils import get_routes_from_neighborhoods


class ETNN(nn.Module):
    """Coordinate-free ETNN backbone over TopoBench neighborhoods.

    The configured TopoBench neighborhoods instantiate the collection of
    neighborhood functions ``CN`` used by the ETNN feature update. Each
    neighborhood is a typed relation from a source cell rank to a destination
    cell rank. The class applies the following computation over those
    relations:

        relation message:
            m_{c,N} = sum_{d in N(c)} psi_N(h_c, h_d, a_{d,c,N})

        rank update:
            h'_c = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

    This mirrors the paper's neighborhood-wise message functions ``psi`` and
    rank-wise feature update ``beta`` while adapting the geometric invariant
    input to the coordinate-free GraphUniverse setting. Since no physical
    coordinates are available, ``a_{d,c,N}`` is the scalar value stored in the
    sparse TopoBench neighborhood rather than an E(n)-invariant geometric
    quantity. The coordinate update from ETNN Eq. 7 is not applied.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for every visible cell rank.
    hidden_channels : int
        Hidden dimension used by ETNN layers.
    out_channels : int
        Output feature dimension for every visible cell rank.
    neighborhoods : list[str]
        TopoBench neighborhood names, e.g. ``"up_adjacency-0"`` or
        ``"down_incidence-1"``.
    num_layers : int, optional
        Number of ETNN message-passing layers.
    dropout : float, optional
        Dropout probability used inside message and update blocks.
    activation : str, optional
        Activation function name.
    use_batch_norm : bool, optional
        Whether to use batch normalization inside MLP blocks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        neighborhoods: list[str],
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(
                "ETNN requires at least one message-passing layer."
            )
        if len(neighborhoods) == 0:
            raise ValueError("ETNN requires at least one neighborhood.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # TopoBench names neighborhoods by the sparse relation they encode.
        # We keep those names as the public config surface, then derive the
        # corresponding source/destination ranks once at construction time.
        self.neighborhoods = list(neighborhoods)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        self.num_layers = num_layers
        self.max_rank = max(max(route) for route in self.routes)

        # This coordinate-free implementation uses one shared input dimension
        # for all selected ranks because AllCellFeatureEncoder projects every
        # rank to the same hidden size.
        self.input_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(in_channels, hidden_channels)
                for rank in range(self.max_rank + 1)
            }
        )

        # Each ETNN layer owns one message-passing block per topological
        # relation. This preserves ETNN's relation-specific parameterization.
        self.layers = nn.ModuleList(
            [
                _ETNNLayer(
                    neighborhoods=self.neighborhoods,
                    routes=self.routes,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Project every rank back to the feature size expected by TopoBench
        # wrappers/readouts.
        self.output_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_channels, out_channels)
                for rank in range(self.max_rank + 1)
            }
        )

    def forward(self, batch) -> dict[int, torch.Tensor]:
        """Run ETNN message passing over a lifted combinatorial batch.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing rank-wise features ``x_i`` and
            sparse neighborhood tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Rank-indexed output embeddings compatible with ``TuneWrapper``.
        """
        x = {}
        for rank in range(self.max_rank + 1):
            key = f"x_{rank}"

            # The lifting/feature encoder must provide one feature tensor per
            # rank touched by the configured neighborhoods.
            if not hasattr(batch, key):
                raise AttributeError(
                    f"ETNN expected rank-{rank} features at `{key}`."
                )

            # Convert rank-wise cell features to the common ETNN hidden space.
            x[rank] = self.input_projection[str(rank)](getattr(batch, key))

        # Apply relation-wise topological message passing repeatedly.
        for layer in self.layers:
            x = layer(x, batch)

        # Return a rank-indexed dictionary so the existing TuneWrapper can turn
        # it into TopoBench's standard model_out format.
        return {
            rank: self.output_projection[str(rank)](features)
            for rank, features in x.items()
        }


class _ETNNLayer(nn.Module):
    """One relation-wise ETNN message-passing layer.

    This layer is the coordinate-free counterpart of the ETNN feature update.
    It implements the paper's two aggregation levels in ordinary tensor
    operations:

    - Intra-neighborhood aggregation: for a fixed neighborhood relation N,
      sender messages are summed into receiver cells with ``torch.index_add_``.
    - Inter-neighborhood aggregation: messages arriving through different
      relation types are concatenated before the rank-specific update MLP.

    Thus each layer computes a residual update of the form:

        h'_c = h_c + beta_rank(c)(h_c, concat_N sum_{d in N(c)} message_N(d,c))

    Parameters
    ----------
    neighborhoods : list[str]
        TopoBench sparse neighborhood names used as ETNN relation types.
    routes : list[list[int]]
        Source and destination rank pairs inferred from ``neighborhoods``.
    hidden_channels : int
        Hidden feature dimension for every rank.
    dropout : float
        Dropout probability used in message and update MLPs.
    activation : str
        Activation name used in message and update MLPs.
    use_batch_norm : bool
        Whether to insert batch normalization in MLP blocks.
    """

    def __init__(
        self,
        neighborhoods: list[str],
        routes: list[list[int]],
        hidden_channels: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.neighborhoods = list(neighborhoods)
        self.routes = [tuple(route) for route in routes]
        self.hidden_channels = hidden_channels

        # ETNN uses separate message functions for separate relation types.
        # Here the relation types are the configured TopoBench neighborhoods.
        self.message_passing = nn.ModuleList(
            [
                _ETNNMessagePassing(
                    hidden_channels=hidden_channels,
                    edge_channels=1,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for _ in self.neighborhoods
            ]
        )

        # The update MLP input size depends on how many relations send messages
        # into each destination rank.
        incoming_counts = defaultdict(int)
        for _, dst_rank in self.routes:
            incoming_counts[dst_rank] += 1

        # Every rank gets its own update function, matching ETNN's rank-wise
        # state update after relation-specific aggregation.
        ranks = sorted({rank for route in self.routes for rank in route})
        self.update = nn.ModuleDict(
            {
                str(rank): _make_mlp(
                    in_channels=(1 + incoming_counts[rank]) * hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for rank in ranks
            }
        )

    def forward(
        self, x: dict[int, torch.Tensor], batch
    ) -> dict[int, torch.Tensor]:
        """Apply one ETNN layer.

        Parameters
        ----------
        x : dict[int, torch.Tensor]
            Rank-indexed hidden cell features.
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing sparse neighborhoods.

        Returns
        -------
        dict[int, torch.Tensor]
            Updated rank-indexed hidden cell features.
        """
        # Collect all incoming messages by destination rank. A rank may receive
        # messages from several neighborhoods in the same layer.
        messages_by_rank = defaultdict(list)

        for route_idx, (neighborhood, route) in enumerate(
            zip(self.neighborhoods, self.routes, strict=False)
        ):
            src_rank, dst_rank = route

            # Convert the sparse TopoBench relation into explicit sender and
            # receiver indices on the same device as the active features.
            edge_index, edge_attr = _neighborhood_to_edge_index(
                batch=batch,
                neighborhood=neighborhood,
                src_rank=src_rank,
                dst_rank=dst_rank,
                device=x[src_rank].device,
                dtype=x[src_rank].dtype,
                num_src_cells=x[src_rank].shape[0],
                num_dst_cells=x[dst_rank].shape[0],
            )

            # Apply the relation-specific ETNN message block and store the
            # aggregated message under the destination rank.
            message = self.message_passing[route_idx](
                x_src=x[src_rank],
                x_dst=x[dst_rank],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            messages_by_rank[dst_rank].append(message)

        out = {}
        for rank, features in x.items():
            # ETNN updates a rank by concatenating its current state with all
            # messages received by that rank, then applying a rank-specific MLP.
            update_input = torch.cat(
                [features, *messages_by_rank.get(rank, [])], dim=-1
            )

            # Residual update keeps the layer stable and mirrors the reference
            # ETNN implementation's rank-wise residual connection.
            out[rank] = features + self.update[str(rank)](update_input)
        return out


class _ETNNMessagePassing(nn.Module):
    """Relation-specific gated message passing.

    The message block plays the role of the neighborhood- and rank-dependent
    ``psi`` function in the ETNN feature update. Because this TopoBench
    implementation has no coordinates, the message input contains sender
    features, receiver features, and a scalar sparse-neighborhood value instead
    of the geometric invariant ``Inv`` used by the full ETNN formulation:

        message_N(d,c) = gate_N(z_{d,c}) * mlp_N(z_{d,c})
        z_{d,c} = concat(h_d, h_c, a_{d,c,N})

    The gate is a learned sigmoid scalar. It follows the same practical design
    idea as the official ETNN implementation: relation messages are learned by
    an MLP and can be softly suppressed or amplified per edge.

    Parameters
    ----------
    hidden_channels : int
        Hidden feature dimension for sender, receiver, and message states.
    edge_channels : int
        Number of scalar structural edge features per relation edge.
    dropout : float
        Dropout probability used inside the message MLP.
    activation : str
        Activation name used inside the message MLP.
    use_batch_norm : bool
        Whether to insert batch normalization inside the message MLP.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()

        # The message sees sender state, receiver state, and one structural
        # edge attribute from the sparse TopoBench neighborhood.
        self.message_mlp = _make_mlp(
            in_channels=2 * hidden_channels + edge_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        # A scalar gate lets the model suppress or emphasize individual
        # relation edges, following the reference ETNN message design.
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate gated messages from source cells to destination cells.

        Parameters
        ----------
        x_src : torch.Tensor
            Source-rank cell features.
        x_dst : torch.Tensor
            Destination-rank cell features.
        edge_index : torch.Tensor
            Sparse relation edges in ``[sender, receiver]`` format.
        edge_attr : torch.Tensor
            Scalar structural edge attributes aligned with ``edge_index``.

        Returns
        -------
        torch.Tensor
            Aggregated message tensor with the same shape as ``x_dst``.
        """
        # Allocate output on the receiver feature device for CPU/GPU safety.
        out = x_dst.new_zeros(x_dst.shape[0], x_dst.shape[1])

        # Some lifted mini-batches may have no cells of a requested rank. In
        # that case the relation is structurally unavailable for this batch, so
        # it contributes a zero message instead of indexing into empty tensors.
        if (
            edge_index.numel() == 0
            or x_src.shape[0] == 0
            or x_dst.shape[0] == 0
        ):
            return out

        sender, receiver = edge_index

        # Build the per-edge ETNN state: sender embedding, receiver embedding,
        # and structural relation attribute.
        state = torch.cat(
            [x_src[sender], x_dst[receiver], edge_attr.to(x_dst.dtype)], dim=-1
        )

        # Compute messages, gate them, and aggregate by receiver cell.
        messages = self.message_mlp(state)
        messages = messages * self.edge_gate(messages)
        out.index_add_(0, receiver, messages)
        return out


def _neighborhood_to_edge_index(
    batch,
    neighborhood: str,
    src_rank: int,
    dst_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    num_src_cells: int | None = None,
    num_dst_cells: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a sparse TopoBench neighborhood into ETNN message edges.

    TopoBench incidence-style neighborhoods are stored as sparse matrices with
    destination cells on rows and source cells on columns. ETNN message passing
    uses an explicit ``[sender, receiver]`` edge index, so we flip the sparse
    matrix indices.

    The conversion is deliberately centralized here because relation direction
    is part of the ETNN semantics: messages must flow from the cells in
    ``N(x)`` to the receiver cell ``x`` in the update equations.

    Only scalar sparse-neighborhood values are used as edge attributes. This is
    enough for TopoBench adjacency/incidence tensors, where a nonzero entry
    means that the sender cell participates in the receiver cell's
    neighborhood. Explicitly stored zero values are discarded before message
    passing because they represent absent or placeholder relations.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch containing the requested sparse neighborhood.
    neighborhood : str
        Name of the sparse neighborhood tensor on ``batch``.
    src_rank : int
        Rank of sender cells for this relation.
    dst_rank : int
        Rank of receiver cells for this relation.
    device : torch.device
        Device on which the returned tensors should live.
    dtype : torch.dtype
        Floating dtype for returned edge attributes.
    num_src_cells : int | None, optional
        Number of real sender cells in the current batch.
    num_dst_cells : int | None, optional
        Number of real receiver cells in the current batch.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge indices in ``[sender, receiver]`` format and scalar edge
        attributes.
    """
    if not hasattr(batch, neighborhood):
        raise AttributeError(f"Missing ETNN neighborhood `{neighborhood}`.")

    # Sparse tensors may be CPU or CUDA depending on the Lightning/TopoBench
    # batch transfer. We coalesce before reading indices and values.
    sparse_neighborhood = getattr(batch, neighborhood).coalesce()
    indices = sparse_neighborhood.indices().long()
    values = sparse_neighborhood.values()

    # Generated sparse neighborhoods can contain explicitly stored zeros for
    # empty or degenerate ranks. They should not become message-passing edges.
    nonzero_mask = values != 0
    indices = indices[:, nonzero_mask]
    values = values[nonzero_mask]
    if values.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=dtype, device=device)
        return edge_index, edge_attr

    # Some TopoBench liftings encode an empty rank with a single zero-valued
    # placeholder row/column. PyG batching preserves those sparse matrix slots,
    # while rank-wise feature tensors only concatenate real cells. Compact the
    # sparse row/column axes back to feature-row indices before message passing.
    dst_map = _sparse_axis_to_feature_index(
        batch=batch,
        rank=dst_rank,
        sparse_size=sparse_neighborhood.shape[0],
        num_cells=num_dst_cells,
        device=indices.device,
    )
    src_map = _sparse_axis_to_feature_index(
        batch=batch,
        rank=src_rank,
        sparse_size=sparse_neighborhood.shape[1],
        num_cells=num_src_cells,
        device=indices.device,
    )

    receiver = dst_map[indices[0]]
    sender = src_map[indices[1]]
    valid_edge_mask = (receiver >= 0) & (sender >= 0)
    receiver = receiver[valid_edge_mask]
    sender = sender[valid_edge_mask]
    values = values[valid_edge_mask]

    # TopoBench stores sparse matrix indices as [row, col]. For message
    # passing, we interpret columns as senders and rows as receivers.
    edge_index = torch.stack([sender, receiver], dim=0).to(device)

    # Use one scalar structural edge feature per sparse nonzero in the
    # coordinate-free model. ``view(-1, 1)`` also gives the correct [0, 1]
    # shape when all stored entries were filtered out above.
    edge_attr = values.view(-1, 1).to(device=device, dtype=dtype)
    return edge_index, edge_attr


def _sparse_axis_to_feature_index(
    batch,
    rank: int,
    sparse_size: int,
    num_cells: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Map a batched sparse axis to compact rank-wise feature rows.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch containing ``batch_<rank>`` assignments.
    rank : int
        Cell rank represented by this sparse axis.
    sparse_size : int
        Length of the sparse matrix axis before compaction.
    num_cells : int | None
        Number of real cells represented in the rank-wise feature tensor.
    device : torch.device
        Device on which the returned mapping should live.

    Returns
    -------
    torch.Tensor
        Long tensor mapping sparse-axis positions to feature-row positions, with
        ``-1`` for placeholder positions that should be dropped.
    """
    if num_cells is None:
        num_cells = sparse_size

    # If sparse rows/columns already align with feature rows, the identity map
    # is both correct and avoids unnecessary work for the common case.
    if sparse_size == num_cells:
        return torch.arange(sparse_size, device=device)

    batch_key = f"batch_{rank}"
    if not hasattr(batch, batch_key):
        raise ValueError(
            "Cannot compact ETNN sparse neighborhood axis for rank "
            f"{rank}: sparse axis has length {sparse_size}, but the "
            f"rank-{rank} feature tensor has {num_cells} rows and "
            f"`{batch_key}` is missing."
        )

    batch_vector = getattr(batch, batch_key).to(device)
    num_graphs = getattr(batch, "num_graphs", None)
    if num_graphs is None:
        num_graphs = (
            int(batch_vector.max().item()) + 1 if batch_vector.numel() else 1
        )

    counts = torch.bincount(batch_vector, minlength=num_graphs).tolist()
    expected_sparse_size = sum(max(1, int(count)) for count in counts)

    if expected_sparse_size != sparse_size:
        raise ValueError(
            "Cannot compact ETNN sparse neighborhood axis for rank "
            f"{rank}: sparse axis has length {sparse_size}, but "
            f"`{batch_key}` implies {expected_sparse_size} slots under "
            "TopoBench's empty-rank placeholder convention."
        )

    mapping = torch.full((sparse_size,), -1, dtype=torch.long, device=device)
    sparse_offset = 0
    feature_offset = 0
    for count in counts:
        count = int(count)
        if count > 0:
            mapping[sparse_offset : sparse_offset + count] = torch.arange(
                feature_offset,
                feature_offset + count,
                device=device,
            )
            feature_offset += count

        # Empty ranks still occupy one sparse slot; nonempty ranks occupy one
        # sparse slot per real cell.
        sparse_offset += max(1, count)

    return mapping


def _make_mlp(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
    activation: str,
    use_batch_norm: bool,
) -> nn.Sequential:
    """Build the small MLP blocks used by ETNN messages and updates.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden feature dimension.
    out_channels : int
        Output feature dimension.
    dropout : float
        Dropout probability.
    activation : str
        Activation name.
    use_batch_norm : bool
        Whether to insert batch normalization after the first linear layer.

    Returns
    -------
    nn.Sequential
        MLP block used by ETNN message and update functions.
    """
    act = _get_activation(activation)

    # Keep the block intentionally small so the coordinate-free backbone stays
    # lightweight inside standard TopoBench training pipelines.
    layers: list[nn.Module] = [nn.Linear(in_channels, hidden_channels)]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(hidden_channels))
    layers.extend([copy.deepcopy(act), nn.Dropout(dropout)])
    layers.append(nn.Linear(hidden_channels, out_channels))
    return nn.Sequential(*layers)


def _get_activation(name: str) -> nn.Module:
    """Resolve activation names used in ETNN configs.

    Parameters
    ----------
    name : str
        Activation name from the ETNN config.

    Returns
    -------
    nn.Module
        Instantiated PyTorch activation module.
    """
    # Match common TopoBench config names while keeping ETNN self-contained.
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "id":
        return nn.Identity()
    raise NotImplementedError(f"Activation `{name}` is not supported.")
