"""Coordinate-policy ETNN backbone for combinatorial complexes.

This module implements a consolidated TopoBench adaptation of
E(n)-Equivariant Topological Neural Networks (ETNNs) from Battiloro et al.,
``E(n) Equivariant Topological Neural Networks``, arXiv:2405.15429, and the
official NSAPH implementation at
``https://github.com/NSAPH-Projects/topological-equivariant-networks``.

The original ETNN architecture combines topological message passing over cells
with geometric invariant information derived from Euclidean coordinates.  In
TopoBench, different datasets expose different coordinate contracts:

1. GraphUniverse-style graph datasets may have no Euclidean coordinates.
2. Graph datasets can be equipped with structural pseudo-coordinates such as
   Laplacian positional encodings.
3. Molecular or physical datasets can provide true rank-0 positions through
   attributes such as ``data.pos``.

This backbone treats the coordinate choice as an explicit model policy rather
than silently inventing or ignoring geometry.  All policies share the same
rank-wise feature-message-passing scaffold:

    m_{c,N} = sum_{d in N(c)} psi_N(h_d, h_c, e_{d,c,N})
    h'_c    = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

Here ``c`` is a receiver cell, ``d`` is a sender cell in a configured
TopoBench neighborhood ``N(c)``, ``psi_N`` is a relation-specific message
function, and ``beta_rank(c)`` is a rank-specific update MLP.  This is the
feature-update part of ETNN/CCMPN.  The coordinate policy determines the edge
attribute ``e_{d,c,N}`` supplied to ``psi_N``:

``coordinate_policy="none"``
    Use only the sparse TopoBench relation value ``a_{d,c,N}``.  This is the
    conservative coordinate-free adaptation for datasets such as GraphUniverse
    where no physical Euclidean coordinates are present:

        e_{d,c,N} = [a_{d,c,N}]

``coordinate_policy="structural_lappe"``
    Use normalized graph Laplacian eigenvectors as structural
    pseudo-coordinates and append a squared structural distance to each
    relation edge:

        p_0(v) = LapPE(v)
        p_r(c) = mean_{d incident to c} p_{r-1}(d)
        e_{d,c,N} = [a_{d,c,N}, ||p_d - p_c||^2]

    These coordinates are graph-derived structural embeddings, not physical
    coordinates.  The distance channel is invariant to rigid transformations
    of the chosen structural coordinate frame, but this mode should not be
    interpreted as physical E(n)-equivariance of the original graph.

``coordinate_policy="physical"``
    Use rank-0 physical coordinates, usually ``data.pos``, to compute
    ETNN-style invariant cell geometry.  For each higher-rank cell, the
    implementation first reconstructs its incident rank-0 vertices from
    TopoBench incidence matrices, then computes centroid and diameter
    summaries from the physical coordinates:

    p_c       = centroid of the rank-0 vertices incident to cell c
    diam(c)   = max_{u,v incident to c} ||pos_u - pos_v||
    H(d,c)    = max_{u in d} min_{v in c} ||pos_u - pos_v||

    e_{d,c,N} = [||p_d - p_c||, diam(d), diam(c), H(d,c), H(c,d)]

The physical mode is the closest TopoBench-native step toward the original
ETNN/NSAPH setting because it uses actual Euclidean coordinates when a dataset
provides them.  When ``pos_update=True``, rank-0 coordinates are updated
internally by a learned radial displacement and physical invariants are
recomputed before the next layer.  The updated coordinates stay inside the
backbone; TopoBench wrappers and readouts still consume only rank-wise feature
embeddings.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import torch
from torch import nn

from topobench.data.utils import get_routes_from_neighborhoods

_SUPPORTED_COORDINATE_POLICIES = {
    "none",
    "structural_lappe",
    "physical",
}
_SUPPORTED_INVARIANT_NORMALIZATIONS = {
    "none",
    "batch_norm",
    "mean_abs",
}


def _get_activation(name: str) -> nn.Module:
    """Resolve an activation used by ETNN message and update MLPs.

    Parameters
    ----------
    name : str
        Activation name from the public model config.

    Returns
    -------
    nn.Module
        Instantiated activation module.

    Raises
    ------
    NotImplementedError
        If ``name`` is not supported by the ETNN config surface.
    """
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "id": nn.Identity,
    }
    if name not in activations:
        raise NotImplementedError(f"Activation `{name}` is not supported.")
    return activations[name]()


def _make_mlp(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
    activation: str,
    use_batch_norm: bool,
    final_activation: bool = False,
) -> nn.Sequential:
    """Build the MLP used by ETNN message and feature-update functions.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden feature dimension.
    out_channels : int
        Output feature dimension.
    dropout : float
        Dropout probability after the hidden activation.
    activation : str
        Activation name resolved by :func:`_get_activation`.
    use_batch_norm : bool
        Whether to normalize the hidden representation.
    final_activation : bool, optional
        Whether to apply the configured activation after the output linear
        layer. The standard coordinate-policy configs leave this disabled.
        The controlled QM9 parity wrapper enables it only for relation-message
        MLPs to match the pinned NSAPH ``lean=False`` implementation.

    Returns
    -------
    nn.Sequential
        Two-linear-layer ETNN MLP block.
    """
    activation_module = _get_activation(activation)
    layers: list[nn.Module] = [nn.Linear(in_channels, hidden_channels)]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(hidden_channels))
    layers.extend([copy.deepcopy(activation_module), nn.Dropout(dropout)])
    layers.append(nn.Linear(hidden_channels, out_channels))
    if final_activation:
        layers.append(copy.deepcopy(activation_module))
    return nn.Sequential(*layers)


class _ETNNMessagePassing(nn.Module):
    """Apply one relation-specific gated ETNN message function.

    For sender ``d``, receiver ``c``, and relation ``N``, this block computes

        m_tilde(d,c,N) = phi_N([h_d, h_c, e_d,c,N])
        gate(d,c,N) = sigmoid(W_N m_tilde(d,c,N) + b_N)
        m(c,N) = sum_{d in N(c)} gate(d,c,N) * m_tilde(d,c,N)

    The scalar gate follows the edge-inference mechanism in the official NSAPH
    ETNN implementation.  The coordinate policy determines the width and
    meaning of ``e_d,c,N``; message aggregation itself is shared by all modes.

    Parameters
    ----------
    hidden_channels : int
        Hidden feature dimension for sender, receiver, and message states.
    edge_channels : int
        Number of policy-dependent scalar relation attributes.
    dropout : float
        Dropout probability inside the message MLP.
    activation : str
        Message-MLP activation name.
    use_batch_norm : bool
        Whether to normalize the hidden message representation.
    final_activation : bool, optional
        Whether the relation-message MLP applies an activation after its
        output linear layer.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        self.message_mlp = _make_mlp(
            in_channels=2 * hidden_channels + edge_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
            final_activation=final_activation,
        )
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
        """Aggregate gated messages into receiver cells.

        Parameters
        ----------
        x_src : torch.Tensor
            Source-rank embeddings.
        x_dst : torch.Tensor
            Destination-rank embeddings.
        edge_index : torch.Tensor
            Relation edges in ``[sender, receiver]`` format.
        edge_attr : torch.Tensor
            Policy-dependent relation attributes aligned with ``edge_index``.

        Returns
        -------
        torch.Tensor
            Aggregated messages with one row per destination cell.
        """
        out = x_dst.new_zeros(x_dst.shape[0], x_dst.shape[1])
        if (
            edge_index.numel() == 0
            or x_src.shape[0] == 0
            or x_dst.shape[0] == 0
        ):
            return out

        sender, receiver = edge_index
        state = torch.cat(
            [x_src[sender], x_dst[receiver], edge_attr.to(x_dst.dtype)], dim=-1
        )
        messages = self.message_mlp(state)
        messages = messages * self.edge_gate(messages)
        out.index_add_(0, receiver, messages)
        return out


def _sparse_axis_to_feature_index(
    batch,
    rank: int,
    sparse_size: int,
    num_cells: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Map a placeholder-padded sparse axis to compact feature rows.

    TopoBench batching can reserve one sparse slot for a graph with no cells at
    a requested rank, while the corresponding ``x_<rank>`` tensor contains no
    row for that graph.  This helper reconstructs the sparse-slot to feature-row
    map from ``batch_<rank>`` and marks placeholder slots with ``-1``.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted batch containing rank-wise batch assignments.
    rank : int
        Cell rank represented by the sparse axis.
    sparse_size : int
        Sparse axis length before compaction.
    num_cells : int | None
        Number of real feature rows for this rank.
    device : torch.device
        Device for the returned index map.

    Returns
    -------
    torch.Tensor
        Long map from sparse slots to feature rows; placeholders map to ``-1``.

    Raises
    ------
    ValueError
        If feature rows and sparse slots differ but batch metadata cannot prove
        a safe compaction.
    """
    if num_cells is None:
        num_cells = sparse_size
    if sparse_size == num_cells:
        return torch.arange(sparse_size, device=device)

    batch_key = f"batch_{rank}"
    if not hasattr(batch, batch_key):
        raise ValueError(
            "Cannot compact ETNN sparse neighborhood axis for rank "
            f"{rank}: sparse axis has length {sparse_size}, but the rank-"
            f"{rank} feature tensor has {num_cells} rows and `{batch_key}` "
            "is missing."
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
            f"{rank}: sparse axis has length {sparse_size}, but `{batch_key}` "
            f"implies {expected_sparse_size} slots under TopoBench's empty-"
            "rank placeholder convention."
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
        sparse_offset += max(1, count)
    return mapping


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
    """Convert a sparse TopoBench relation into ETNN message edges.

    TopoBench relation matrices store receivers on rows and senders on columns,
    whereas message passing consumes ``[sender, receiver]`` edge indices.  The
    conversion also removes stored zeros and compacts empty-rank placeholder
    axes before any feature tensor is indexed.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted batch containing the requested sparse relation.
    neighborhood : str
        Sparse relation attribute name.
    src_rank : int
        Sender-cell rank.
    dst_rank : int
        Receiver-cell rank.
    device : torch.device
        Device for returned tensors.
    dtype : torch.dtype
        Floating dtype for sparse relation values.
    num_src_cells : int | None, optional
        Number of real sender feature rows.
    num_dst_cells : int | None, optional
        Number of real receiver feature rows.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``[sender, receiver]`` indices and scalar sparse relation values.

    Raises
    ------
    AttributeError
        If the configured relation is absent from the lifted batch.
    """
    if not hasattr(batch, neighborhood):
        raise AttributeError(f"Missing ETNN neighborhood `{neighborhood}`.")

    sparse_neighborhood = getattr(batch, neighborhood).coalesce()
    indices = sparse_neighborhood.indices().long()
    values = sparse_neighborhood.values()

    # Stored zeros are absent relations, including empty-rank placeholders.
    nonzero_mask = values != 0
    indices = indices[:, nonzero_mask]
    values = values[nonzero_mask]
    if values.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, 1), dtype=dtype, device=device),
        )

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

    edge_index = torch.stack([sender, receiver], dim=0).to(device)
    edge_attr = values.view(-1, 1).to(device=device, dtype=dtype)
    return edge_index, edge_attr


def _average_coordinates_through_incidence(
    lower_coordinates: torch.Tensor,
    incidence: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """Average lower-rank coordinates into incident higher-rank cells.

    Parameters
    ----------
    lower_coordinates : torch.Tensor
        Rank-``r-1`` coordinates with shape ``[num_lower_cells, coord_dim]``.
    incidence : torch.Tensor
        Sparse ``incidence_r`` with lower cells on rows and rank-``r`` cells
        on columns.
    num_cells : int
        Number of real rank-``r`` cells.

    Returns
    -------
    torch.Tensor
        Rank-``r`` coordinates with shape ``[num_cells, coord_dim]``.

    Raises
    ------
    ValueError
        If either incidence axis is incompatible with the rank-wise tensors.
    """
    if num_cells == 0:
        return lower_coordinates.new_empty((0, lower_coordinates.shape[1]))

    incidence = incidence.coalesce()
    if incidence.shape[0] != lower_coordinates.shape[0]:
        raise ValueError(
            "Cannot lift structural ETNN coordinates: incidence has "
            f"{incidence.shape[0]} source rows, but the lower rank has "
            f"{lower_coordinates.shape[0]} coordinate rows."
        )
    if incidence.shape[1] != num_cells:
        raise ValueError(
            "Cannot lift structural ETNN coordinates: incidence has "
            f"{incidence.shape[1]} columns, but the target rank has "
            f"{num_cells} cells."
        )

    indices = incidence.indices()
    values = incidence.values().abs().to(lower_coordinates.dtype)
    nonzero_mask = values != 0
    indices = indices[:, nonzero_mask]
    values = values[nonzero_mask]

    coordinates = lower_coordinates.new_zeros(
        (num_cells, lower_coordinates.shape[1])
    )
    weights = lower_coordinates.new_zeros((num_cells, 1))
    if values.numel() == 0:
        return coordinates

    lower_idx, cell_idx = indices
    coordinates.index_add_(
        0,
        cell_idx,
        lower_coordinates[lower_idx] * values.unsqueeze(-1),
    )
    weights.index_add_(0, cell_idx, values.unsqueeze(-1))

    # Degenerate target cells retain zero coordinates; nonzero columns receive
    # their absolute-incidence-weighted barycenter.
    weights = weights.clamp_min(torch.finfo(weights.dtype).eps)
    return coordinates / weights


def _build_lappe_cell_coordinates(
    batch,
    coordinate_attr: str,
    max_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Lift rank-0 LapPE coordinates recursively through incidences.

    Rank 0 reads the graph-derived coordinate matrix stored at
    ``coordinate_attr``.  For each higher rank, absolute ``incidence_r`` values
    average rank-``r-1`` coordinates into rank-``r`` cells.  This recursive
    policy matches the available TopoBench data contract and deliberately keeps
    structural coordinates separate from physical geometry.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted batch with rank-wise features and consecutive incidences.
    coordinate_attr : str
        Rank-0 structural-coordinate attribute, normally ``LapPE``.
    max_rank : int
        Highest visible rank requiring coordinates.
    device : torch.device
        Active model device.
    dtype : torch.dtype
        Active feature dtype.

    Returns
    -------
    dict[int, torch.Tensor]
        Structural coordinates indexed by cell rank.

    Raises
    ------
    AttributeError
        If coordinates, rank features, or required incidences are missing.
    ValueError
        If rank-0 coordinates are malformed or misaligned with ``x_0``.
    """
    if not hasattr(batch, coordinate_attr):
        raise AttributeError(
            "Structural-LapPE ETNN expected rank-0 coordinates at "
            f"`{coordinate_attr}`."
        )

    rank_0_coordinates = getattr(batch, coordinate_attr).to(
        device=device,
        dtype=dtype,
    )
    if rank_0_coordinates.ndim != 2:
        raise ValueError(
            "Structural-LapPE ETNN expected a rank-0 coordinate matrix at "
            f"`{coordinate_attr}`, but found shape "
            f"{tuple(rank_0_coordinates.shape)}."
        )
    if rank_0_coordinates.shape[0] != batch.x_0.shape[0]:
        raise ValueError(
            "Structural-LapPE ETNN expected one coordinate row per rank-0 "
            f"cell, but found {rank_0_coordinates.shape[0]} coordinates for "
            f"{batch.x_0.shape[0]} cells."
        )

    coordinates = {0: rank_0_coordinates}
    for rank in range(1, max_rank + 1):
        feature_key = f"x_{rank}"
        if not hasattr(batch, feature_key):
            raise AttributeError(
                "Structural-LapPE ETNN expected rank-"
                f"{rank} features at `{feature_key}`."
            )
        incidence_key = f"incidence_{rank}"
        if not hasattr(batch, incidence_key):
            raise AttributeError(
                "Structural-LapPE ETNN needs incidence matrices to lift "
                f"coordinates, but `{incidence_key}` is missing."
            )

        incidence = getattr(batch, incidence_key).coalesce().to(device)
        coordinates[rank] = _average_coordinates_through_incidence(
            lower_coordinates=coordinates[rank - 1],
            incidence=incidence,
            num_cells=getattr(batch, feature_key).shape[0],
        )
    return coordinates


def _squared_coordinate_distances(
    src_coordinates: torch.Tensor,
    dst_coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute E(n)-invariant squared distances for relation edges.

    Parameters
    ----------
    src_coordinates : torch.Tensor
        Sender-cell structural coordinates.
    dst_coordinates : torch.Tensor
        Receiver-cell structural coordinates.
    edge_index : torch.Tensor
        Relation edges in ``[sender, receiver]`` format.
    dtype : torch.dtype
        Floating dtype for the distance channel.

    Returns
    -------
    torch.Tensor
        Squared distances with shape ``[num_edges, 1]``.
    """
    if edge_index.numel() == 0:
        return src_coordinates.new_empty((0, 1), dtype=dtype)
    sender, receiver = edge_index
    delta = src_coordinates[sender] - dst_coordinates[receiver]
    return delta.pow(2).sum(dim=-1, keepdim=True).to(dtype)


class ETNNCoordinatePolicy(nn.Module):
    """ETNN feature backbone with explicit coordinate policies.

    The class exposes one ETNN model family whose behavior is controlled by an
    explicit coordinate policy.  This avoids ambiguous ``auto`` behavior: a
    reproducible experiment should state whether it is using no coordinates,
    structural pseudo-coordinates, or physical Euclidean coordinates.

    All policies share the same TopoBench lifting contract, rank-wise
    input/output projections, typed neighborhood routes, relation-specific
    message MLPs, and residual rank-wise feature updates.  They differ only in
    which invariant edge attributes are appended to relation messages.

    For a receiver cell ``c``, sender cell ``d``, and neighborhood relation
    ``N``, the shared feature update is:

        m_{c,N} = sum_{d in N(c)} psi_N(h_d, h_c, e_{d,c,N})
        h'_c    = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

    In implementation, the ``concat_N`` operation is rank-wise and follows the
    configured ``neighborhoods`` order.  Empty relations contribute zero
    messages through the shared message-passing helper, so the rank update MLP
    input dimension is fixed by the model config rather than by individual
    cells in a batch.

    The policy determines ``e_{d,c,N}`` and therefore the relation edge-channel
    width:

    - ``none``: one channel ``[a_{d,c,N}]``
    - ``structural_lappe``: two channels
      ``[a_{d,c,N}, ||p_d - p_c||^2]``
    - ``physical``: three or five channels depending on ``hausdorff_dists``.
      With the default Hausdorff setting, this is
      ``[||centroid_d-centroid_c||, diam(d), diam(c), H(d,c), H(c,d)]``.

    The ``physical`` policy is intentionally strict: it requires
    ``physical_coordinate_attr`` and raises if the coordinate tensor is missing
    or not aligned with rank-0 features.  This is important because many graph
    datasets use attributes named ``pos`` for layouts or pseudo-coordinates;
    the config should decide when those coordinates are meaningful physical
    inputs.

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
    coordinate_policy : str, optional
        Coordinate mode. Supported values are ``"none"``,
        ``"structural_lappe"``, and ``"physical"``.
    structural_coordinate_attr : str, optional
        Batch attribute containing rank-0 structural coordinates for
        ``coordinate_policy="structural_lappe"``.
    physical_coordinate_attr : str, optional
        Batch attribute containing rank-0 physical coordinates for
        ``coordinate_policy="physical"``.
    pos_update : bool, optional
        Whether physical mode should update rank-0 coordinates after each layer.
    coordinate_update_scale : float, optional
        Multiplicative factor applied to the learned radial coordinate update.
    coordinate_update_neighborhood : str, optional
        Rank-0 to rank-0 neighborhood used for coordinate updates.
    hausdorff_dists : bool, optional
        Whether physical mode should include the two directed Hausdorff-style
        distances used by the NSAPH implementation.
    normalize_invariants : bool, optional
        Convenience flag for enabling physical invariant normalization.  If
        ``True`` and ``invariant_normalization`` is left as ``"none"``, the
        physical layer uses ``"batch_norm"``.
    invariant_normalization : str, optional
        Physical invariant normalization mode.  Supported values are
        ``"none"``, ``"batch_norm"``, and ``"mean_abs"``.  ``"batch_norm"``
        matches NSAPH's per-adjacency ``BatchNorm1d(..., affine=False)`` pattern,
        while ``"mean_abs"`` is a lightweight TopoBench-safe alternative.
    invariant_normalization_eps : float, optional
        Numerical floor used by the ``"mean_abs"`` normalization mode.
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
        coordinate_policy: str = "none",
        structural_coordinate_attr: str = "LapPE",
        physical_coordinate_attr: str = "pos",
        pos_update: bool = False,
        coordinate_update_scale: float = 0.1,
        coordinate_update_neighborhood: str = "up_adjacency-0",
        hausdorff_dists: bool = True,
        normalize_invariants: bool = False,
        invariant_normalization: str = "none",
        invariant_normalization_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        # Fail early on malformed configs. The rest of the constructor assumes
        # there is at least one relation route and one message-passing layer.
        if num_layers < 1:
            raise ValueError(
                "ETNNCoordinatePolicy requires at least one "
                "message-passing layer."
            )
        if len(neighborhoods) == 0:
            raise ValueError(
                "ETNNCoordinatePolicy requires at least one neighborhood."
            )

        # Keep the policy string validated and stored once. This avoids
        # downstream branches silently accepting misspelled policy names.
        self.coordinate_policy = _validate_coordinate_policy(coordinate_policy)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.structural_coordinate_attr = structural_coordinate_attr
        self.physical_coordinate_attr = physical_coordinate_attr
        self.pos_update = bool(pos_update)
        self.coordinate_update_scale = coordinate_update_scale
        self.coordinate_update_neighborhood = coordinate_update_neighborhood
        self.hausdorff_dists = bool(hausdorff_dists)
        self.invariant_normalization = _resolve_invariant_normalization(
            normalize_invariants=normalize_invariants,
            invariant_normalization=invariant_normalization,
        )
        self.normalize_invariants = self.invariant_normalization != "none"
        self.invariant_normalization_eps = invariant_normalization_eps

        if (
            self.coordinate_policy != "physical"
            and self.invariant_normalization != "none"
        ):
            raise ValueError(
                "`invariant_normalization` is only used for "
                "coordinate_policy='physical'. Set "
                "`invariant_normalization='none'` for coordinate-free or "
                "structural-coordinate policies."
            )

        if self.pos_update and self.coordinate_policy != "physical":
            raise ValueError(
                "`pos_update=True` is only supported for "
                "coordinate_policy='physical'."
            )

        # Match the existing ETNN config surface: neighborhoods are public,
        # while routes are inferred once for deterministic relation ordering.
        self.neighborhoods = list(neighborhoods)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        self.num_layers = num_layers
        self.max_rank = max(max(route) for route in self.routes)

        if self.pos_update:
            if self.coordinate_update_neighborhood not in self.neighborhoods:
                raise ValueError(
                    "`pos_update=True` requires coordinate update "
                    f"neighborhood `{self.coordinate_update_neighborhood}` "
                    "to be present in `neighborhoods`."
                )
            route_idx = self.neighborhoods.index(
                self.coordinate_update_neighborhood
            )
            if tuple(self.routes[route_idx]) != (0, 0):
                raise ValueError(
                    "Physical coordinate updates require a rank-0 to rank-0 "
                    "neighborhood. "
                    f"`{self.coordinate_update_neighborhood}` has route "
                    f"{tuple(self.routes[route_idx])}."
                )

        # One rank-specific linear map is used because TopoBench represents
        # each cell rank with its own feature tensor x_0, x_1, ...
        self.input_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(in_channels, hidden_channels)
                for rank in range(self.max_rank + 1)
            }
        )

        # Every ETNN layer uses the same coordinate policy, so each layer has
        # the correct relation edge-channel width from construction time.
        self.layers = nn.ModuleList(
            [
                _ETNNCoordinatePolicyLayer(
                    neighborhoods=self.neighborhoods,
                    routes=self.routes,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    coordinate_policy=self.coordinate_policy,
                    max_rank=self.max_rank,
                    pos_update=self.pos_update,
                    coordinate_update_scale=self.coordinate_update_scale,
                    coordinate_update_neighborhood=(
                        self.coordinate_update_neighborhood
                    ),
                    hausdorff_dists=self.hausdorff_dists,
                    invariant_normalization=self.invariant_normalization,
                    invariant_normalization_eps=(
                        self.invariant_normalization_eps
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        # Project each updated rank back to the wrapper/readout feature size.
        self.output_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_channels, out_channels)
                for rank in range(self.max_rank + 1)
            }
        )

    def forward(self, batch) -> dict[int, torch.Tensor]:
        """Run ETNN message passing under the selected coordinate policy.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing rank-wise features and the
            coordinate attributes required by the configured policy.

        Returns
        -------
        dict[int, torch.Tensor]
            Rank-indexed output embeddings compatible with ``TuneWrapper``.
        """
        x = {}
        for rank in range(self.max_rank + 1):
            key = f"x_{rank}"
            features = getattr(batch, key, None)
            if features is None:
                raise AttributeError(
                    "ETNNCoordinatePolicy expected rank-"
                    f"{rank} features at `{key}`."
                )

            # AllCellFeatureEncoder projects each selected rank to the common
            # input dimension declared in the model config.  ETNN then works in
            # one shared hidden dimension across ranks.
            x[rank] = self.input_projection[str(rank)](features)

        structural_coordinates = None
        physical_coordinates = None
        vertex_memberships = None
        if self.coordinate_policy == "structural_lappe":
            # Structural coordinates are fixed auxiliary inputs in this mode.
            # They are lifted once per batch and reused by all ETNN layers.
            structural_coordinates = _build_lappe_cell_coordinates(
                batch=batch,
                coordinate_attr=self.structural_coordinate_attr,
                max_rank=self.max_rank,
                device=x[0].device,
                dtype=x[0].dtype,
            )
        elif self.coordinate_policy == "physical":
            # Physical mode starts from true rank-0 coordinates.  The tensor is
            # cloned because pos_update should not mutate the input batch.
            physical_coordinates = _validate_physical_coordinates(
                batch=batch,
                coordinate_attr=self.physical_coordinate_attr,
                device=x[0].device,
                dtype=x[0].dtype,
            ).clone()
            # Cell memberships depend only on topology, so they can be reused
            # across layers even when rank-0 coordinates are updated.
            vertex_memberships = _build_vertex_memberships(
                batch=batch,
                max_rank=self.max_rank,
                device=x[0].device,
                dtype=x[0].dtype,
            )

        for layer in self.layers:
            physical_geometry = None
            if self.coordinate_policy == "physical":
                # Recompute physical summaries from the current coordinates
                # before each layer, matching the NSAPH behavior when positions
                # are updated between layers.
                physical_geometry = _summarize_physical_cell_geometry(
                    vertex_coordinates=physical_coordinates,
                    vertex_memberships=vertex_memberships,
                )

            x, physical_coordinates = layer(
                x=x,
                batch=batch,
                structural_coordinates=structural_coordinates,
                physical_geometry=physical_geometry,
                physical_coordinates=physical_coordinates,
            )

        # TuneWrapper expects a dictionary keyed by integer cell rank.
        return {
            rank: self.output_projection[str(rank)](features)
            for rank, features in x.items()
        }


class _ETNNCoordinatePolicyLayer(nn.Module):
    """One ETNN layer with policy-dependent invariant edge attributes.

    The layer mirrors the two-level ETNN aggregation used in the existing
    coordinate-free and LapPE backbones:

    1. A relation-specific message MLP aggregates sender cells into receiver
       cells for each configured TopoBench neighborhood.
    2. A rank-specific update MLP combines the current rank state with the
       ordered list of incoming relation messages.

    This layer does not know which dataset produced the coordinates.  It only
    receives precomputed policy inputs from ``ETNNCoordinatePolicy.forward``:
    structural rank-wise coordinates for ``structural_lappe`` or physical
    centroids/diameters for ``physical``.  This separation keeps data
    validation and message aggregation responsibilities clear.

    Relation order is part of the architecture.  Messages are concatenated in
    ``self.neighborhoods`` order, matching the way the update MLP input widths
    are calculated during initialization.  Changing the neighborhood list is a
    real model/config change and should be reflected in the lifting defaults.

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
    coordinate_policy : str
        Coordinate policy determining relation edge-channel width.
    max_rank : int
        Maximum visible cell rank expected by this layer.
    pos_update : bool
        Whether physical mode should update rank-0 coordinates internally.
    coordinate_update_scale : float
        Scalar multiplier applied to the learned rank-0 coordinate delta.
    coordinate_update_neighborhood : str
        Rank-0 to rank-0 neighborhood used for physical coordinate updates.
    hausdorff_dists : bool
        Whether physical mode appends the two directed Hausdorff-style
        invariant channels.
    invariant_normalization : str
        Physical invariant normalization mode: ``"none"``, ``"batch_norm"``,
        or ``"mean_abs"``.
    invariant_normalization_eps : float
        Numerical floor used by the ``"mean_abs"`` normalization mode.
    """

    def __init__(
        self,
        neighborhoods: list[str],
        routes: list[list[int]],
        hidden_channels: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
        coordinate_policy: str,
        max_rank: int,
        pos_update: bool,
        coordinate_update_scale: float,
        coordinate_update_neighborhood: str,
        hausdorff_dists: bool,
        invariant_normalization: str,
        invariant_normalization_eps: float,
    ) -> None:
        super().__init__()

        # Store routes as tuples so unpacking later is explicit and immutable.
        self.neighborhoods = list(neighborhoods)
        self.routes = [tuple(route) for route in routes]
        self.coordinate_policy = _validate_coordinate_policy(coordinate_policy)
        self.pos_update = bool(pos_update)
        self.coordinate_update_scale = coordinate_update_scale
        self.coordinate_update_neighborhood = coordinate_update_neighborhood
        self.hausdorff_dists = bool(hausdorff_dists)
        self.invariant_normalization = _validate_invariant_normalization(
            invariant_normalization
        )
        self.normalize_invariants = self.invariant_normalization != "none"
        self.invariant_normalization_eps = invariant_normalization_eps

        # A mismatch here means the config and route parser disagree about the
        # ETNN relation set; message ordering would otherwise be undefined.
        if len(self.neighborhoods) != len(self.routes):
            raise ValueError(
                "ETNNCoordinatePolicy expected one route per neighborhood, "
                f"but found {len(self.neighborhoods)} neighborhoods and "
                f"{len(self.routes)} routes."
            )

        # The policy decides how many scalar edge features are passed to every
        # relation-specific psi_N message function.
        self.edge_channels = _edge_channels_for_coordinate_policy(
            self.coordinate_policy,
            hausdorff_dists=self.hausdorff_dists,
        )
        self.invariant_batch_norm = nn.ModuleList(
            [
                nn.BatchNorm1d(self.edge_channels, affine=False)
                for _ in self.neighborhoods
            ]
            if self.coordinate_policy == "physical"
            and self.invariant_normalization == "batch_norm"
            else []
        )

        if self.pos_update and self.coordinate_policy != "physical":
            raise ValueError(
                "`pos_update=True` is only supported for physical ETNN mode."
            )
        if self.pos_update:
            if self.coordinate_update_neighborhood not in self.neighborhoods:
                raise ValueError(
                    "`pos_update=True` requires coordinate update "
                    f"neighborhood `{self.coordinate_update_neighborhood}`."
                )
            self.coordinate_update_route_idx = self.neighborhoods.index(
                self.coordinate_update_neighborhood
            )
            if self.routes[self.coordinate_update_route_idx] != (0, 0):
                raise ValueError(
                    "Physical coordinate update neighborhood must route "
                    "rank 0 to rank 0."
                )
        else:
            self.coordinate_update_route_idx = None

        # Use one message module per relation type. This preserves ETNN's
        # neighborhood-specific message functions instead of sharing all
        # relations through a single MLP.
        self.message_passing = nn.ModuleList(
            [
                _ETNNMessagePassing(
                    hidden_channels=hidden_channels,
                    edge_channels=self.edge_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for _ in self.neighborhoods
            ]
        )

        # NSAPH updates physical coordinates with one bias-free linear map from
        # rank-0 messages to scalar radial weights.  Keeping this as a single
        # Linear layer is less expressive than an MLP, but it is closer to the
        # official coordinate-update implementation.
        self.coordinate_update_mlp = (
            nn.Linear(hidden_channels, 1, bias=False)
            if self.pos_update
            else None
        )
        if self.coordinate_update_mlp is not None:
            nn.init.trunc_normal_(self.coordinate_update_mlp.weight, std=0.02)

        # The destination rank controls how many relation messages will be
        # concatenated before the rank-specific beta_r update.
        incoming_counts = defaultdict(int)
        for _, dst_rank in self.routes:
            incoming_counts[dst_rank] += 1

        # Create one update MLP per visible rank. A rank that receives two
        # relation types gets [current_features, message_1, message_2].
        ranks = list(range(max_rank + 1))
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
        self,
        x: dict[int, torch.Tensor],
        batch,
        structural_coordinates: dict[int, torch.Tensor] | None,
        physical_geometry: _PhysicalCellGeometry | None,
        physical_coordinates: torch.Tensor | None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """Apply one coordinate-policy ETNN feature-update layer.

        Parameters
        ----------
        x : dict[int, torch.Tensor]
            Rank-indexed hidden cell features.
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing sparse neighborhoods.
        structural_coordinates : dict[int, torch.Tensor] | None
            Rank-wise structural coordinates for ``structural_lappe`` mode.
        physical_geometry : _PhysicalCellGeometry | None
            Rank-wise centroids and diameters for ``physical`` mode.
        physical_coordinates : torch.Tensor | None
            Current rank-0 coordinates for ``physical`` mode.

        Returns
        -------
        tuple[dict[int, torch.Tensor], torch.Tensor | None]
            Updated rank-indexed hidden cell features and, in physical mode,
            updated rank-0 coordinates.
        """
        messages_by_rank = defaultdict(list)
        updated_physical_coordinates = physical_coordinates

        # Process neighborhoods in config order. This order is the channel
        # order seen by each rank-specific update MLP.
        for route_idx, (neighborhood, route) in enumerate(
            zip(self.neighborhoods, self.routes, strict=True)
        ):
            src_rank, dst_rank = route

            # Convert TopoBench's sparse matrix relation into explicit
            # sender/receiver indices.  The sparse value is used directly by
            # coordinate-free/structural policies and replaced by physical
            # invariants in physical mode.
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

            # Add no geometry, structural pseudo-geometry, or physical
            # geometry according to the configured coordinate policy.
            edge_attr = self._build_policy_edge_attributes(
                route_idx=route_idx,
                edge_attr=edge_attr,
                edge_index=edge_index,
                src_rank=src_rank,
                dst_rank=dst_rank,
                structural_coordinates=structural_coordinates,
                physical_geometry=physical_geometry,
            )

            # Aggregate relation-specific messages into the destination rank.
            message = self.message_passing[route_idx](
                x_src=x[src_rank],
                x_dst=x[dst_rank],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            messages_by_rank[dst_rank].append(message)

            # Physical coordinate updates use the rank-0 message on the selected
            # rank-0 adjacency, then move coordinates along relative position
            # vectors.  Coordinates are updated after the message is computed, so
            # the next ETNN layer recomputes invariants from the new positions.
            if route_idx == self.coordinate_update_route_idx:
                if updated_physical_coordinates is None:
                    raise ValueError(
                        "Physical coordinate updates require rank-0 "
                        "coordinates."
                    )
                updated_physical_coordinates = self._update_rank_0_coordinates(
                    coordinates=updated_physical_coordinates,
                    edge_index=edge_index,
                    aggregated_message=message,
                )

        out = {}
        for rank, features in x.items():
            # Concatenate messages in the same order used to count incoming
            # relations during initialization.
            update_input = torch.cat(
                [features, *messages_by_rank.get(rank, [])], dim=-1
            )

            # Residual rank-wise feature update, matching the ETNN backbone
            # pattern used by the earlier coordinate-free implementation.
            out[rank] = features + self.update[str(rank)](update_input)
        return out, updated_physical_coordinates

    def _build_policy_edge_attributes(
        self,
        route_idx: int,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        src_rank: int,
        dst_rank: int,
        structural_coordinates: dict[int, torch.Tensor] | None,
        physical_geometry: _PhysicalCellGeometry | None,
    ) -> torch.Tensor:
        """Build policy-specific relation edge attributes.

        Parameters
        ----------
        route_idx : int
            Index of the current relation in ``self.neighborhoods``. Physical
            BatchNorm uses this to select the per-relation normalizer.
        edge_attr : torch.Tensor
            Sparse TopoBench relation values with shape ``[num_edges, 1]``.
            Physical mode validates this tensor shape but replaces the value
            with NSAPH-style invariant features.
        edge_index : torch.Tensor
            Relation edges in ``[sender, receiver]`` format.
        src_rank : int
            Rank of sender cells for the current relation.
        dst_rank : int
            Rank of receiver cells for the current relation.
        structural_coordinates : dict[int, torch.Tensor] | None
            Rank-wise structural coordinates for ``structural_lappe`` mode.
        physical_geometry : _PhysicalCellGeometry | None
            Rank-wise physical centroids and diameters for ``physical`` mode.

        Returns
        -------
        torch.Tensor
            Relation edge attributes with width determined by
            ``coordinate_policy``.
        """
        if edge_attr.ndim != 2 or edge_attr.shape[1] != 1:
            raise ValueError(
                "ETNNCoordinatePolicy expected scalar sparse edge "
                f"attributes with shape [num_edges, 1], found "
                f"{tuple(edge_attr.shape)}."
            )

        if self.coordinate_policy == "none":
            # Coordinate-free mode uses only the sparse relation value.
            return edge_attr

        if self.coordinate_policy == "structural_lappe":
            if structural_coordinates is None:
                raise ValueError(
                    "ETNNCoordinatePolicy structural_lappe mode requires "
                    "rank-wise structural coordinates."
                )

            # LapPE mode appends one invariant structural distance channel.
            # Squared distance is unchanged by rigid transformations of the
            # graph-derived coordinate frame.
            distance_attr = _squared_coordinate_distances(
                src_coordinates=structural_coordinates[src_rank],
                dst_coordinates=structural_coordinates[dst_rank],
                edge_index=edge_index,
                dtype=edge_attr.dtype,
            )
            return torch.cat([edge_attr, distance_attr], dim=-1)

        # The only remaining valid policy is physical. The earlier validation
        # guarantees this branch is not silently handling an unknown mode.
        if physical_geometry is None:
            raise ValueError(
                "ETNNCoordinatePolicy physical mode requires physical cell "
                "geometry."
            )

        # Physical mode uses ETNN-style invariant channels derived from Euclidean
        # rank-0 coordinates and incident vertex memberships. With
        # hausdorff_dists=True this matches NSAPH's default five-channel physical
        # invariant vector; otherwise it keeps the three-channel diameter variant.
        invariant_attr = _physical_relation_invariants(
            src_centroids=physical_geometry.centroids[src_rank],
            dst_centroids=physical_geometry.centroids[dst_rank],
            src_diameters=physical_geometry.diameters[src_rank],
            dst_diameters=physical_geometry.diameters[dst_rank],
            src_membership=physical_geometry.vertex_memberships[src_rank],
            dst_membership=physical_geometry.vertex_memberships[dst_rank],
            vertex_coordinates=physical_geometry.vertex_coordinates,
            edge_index=edge_index,
            dtype=edge_attr.dtype,
            hausdorff_dists=self.hausdorff_dists,
        )
        if self.invariant_normalization == "batch_norm":
            invariant_attr = _batch_norm_physical_invariants(
                invariant_attr=invariant_attr,
                normalizer=self.invariant_batch_norm[route_idx],
            )
        elif self.invariant_normalization == "mean_abs":
            invariant_attr = _normalize_physical_invariants(
                invariant_attr,
                eps=self.invariant_normalization_eps,
            )
        return invariant_attr

    def _update_rank_0_coordinates(
        self,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        aggregated_message: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the learned radial rank-0 coordinate update.

        Parameters
        ----------
        coordinates : torch.Tensor
            Current rank-0 coordinates.
        edge_index : torch.Tensor
            Rank-0 relation edges in ``[sender, receiver]`` format.
        aggregated_message : torch.Tensor
            Rank-0 message tensor produced by the selected update relation.

        Returns
        -------
        torch.Tensor
            Updated rank-0 coordinates.
        """
        if self.coordinate_update_mlp is None:
            return coordinates
        if edge_index.numel() == 0 or aggregated_message.shape[0] == 0:
            return coordinates

        sender, receiver = edge_index

        # The scalar can be positive or negative because the final MLP layer is
        # linear.  Following NSAPH, the scalar is read from the receiver-side
        # message and scattered to the sender along the sender-receiver relative
        # vector.  Relative vectors make the update equivariant to translations,
        # rotations, and reflections of the physical coordinate frame.
        radial_weight = self.coordinate_update_mlp(
            aggregated_message[receiver]
        ).to(coordinates.dtype)
        relative = coordinates[sender] - coordinates[receiver]
        coordinate_delta = coordinates.new_zeros(coordinates.shape)
        coordinate_delta.index_add_(0, sender, relative * radial_weight)
        return coordinates + self.coordinate_update_scale * coordinate_delta


def _validate_coordinate_policy(coordinate_policy: str) -> str:
    """Validate a coordinate-policy name.

    Parameters
    ----------
    coordinate_policy : str
        Requested coordinate policy.

    Returns
    -------
    str
        Validated coordinate policy.
    """
    if coordinate_policy not in _SUPPORTED_COORDINATE_POLICIES:
        supported = ", ".join(sorted(_SUPPORTED_COORDINATE_POLICIES))
        raise ValueError(
            "Unsupported ETNN coordinate policy "
            f"`{coordinate_policy}`. Supported policies are: {supported}."
        )
    return coordinate_policy


def _validate_invariant_normalization(invariant_normalization: str) -> str:
    """Validate a physical invariant-normalization mode.

    Parameters
    ----------
    invariant_normalization : str
        Candidate physical invariant-normalization mode.

    Returns
    -------
    str
        Validated normalization mode.
    """
    if invariant_normalization not in _SUPPORTED_INVARIANT_NORMALIZATIONS:
        supported = ", ".join(sorted(_SUPPORTED_INVARIANT_NORMALIZATIONS))
        raise ValueError(
            "Unsupported ETNN physical invariant normalization "
            f"`{invariant_normalization}`. Supported modes are: {supported}."
        )
    return invariant_normalization


def _resolve_invariant_normalization(
    normalize_invariants: bool,
    invariant_normalization: str,
) -> str:
    """Resolve the legacy boolean flag and explicit normalization mode.

    Parameters
    ----------
    normalize_invariants : bool
        Backward-compatible convenience flag. If true and no explicit mode is
        selected, the physical default resolves to ``"batch_norm"``.
    invariant_normalization : str
        Explicit physical invariant-normalization mode.

    Returns
    -------
    str
        Resolved normalization mode.
    """
    invariant_normalization = _validate_invariant_normalization(
        invariant_normalization
    )
    if normalize_invariants and invariant_normalization == "none":
        return "batch_norm"
    if not normalize_invariants and invariant_normalization != "none":
        return invariant_normalization
    return invariant_normalization


def _edge_channels_for_coordinate_policy(
    coordinate_policy: str,
    hausdorff_dists: bool = False,
) -> int:
    """Return relation edge-channel width for a coordinate policy.

    Parameters
    ----------
    coordinate_policy : str
        Coordinate policy controlling relation edge attributes.
    hausdorff_dists : bool, optional
        Whether physical mode includes the two directed Hausdorff channels.

    Returns
    -------
    int
        Number of scalar edge channels consumed by each relation message MLP.
    """
    coordinate_policy = _validate_coordinate_policy(coordinate_policy)
    if coordinate_policy == "none":
        return 1
    if coordinate_policy == "structural_lappe":
        return 2
    return 5 if hausdorff_dists else 3


class _PhysicalCellGeometry:
    """Rank-wise physical summaries for combinatorial cells.

    Parameters
    ----------
    vertex_coordinates : torch.Tensor
        Current rank-0 physical coordinates.
    vertex_memberships : dict[int, torch.Tensor]
        Rank-indexed vertex-to-cell membership matrices.
    centroids : dict[int, torch.Tensor]
        Rank-indexed centroid coordinates with shape
        ``[num_rank_cells, coordinate_dim]``.
    diameters : dict[int, torch.Tensor]
        Rank-indexed cell diameters with shape ``[num_rank_cells, 1]``.

    Attributes
    ----------
    centroids : dict[int, torch.Tensor]
        Rank-indexed centroid coordinates with shape
        ``[num_rank_cells, coordinate_dim]``.
    diameters : dict[int, torch.Tensor]
        Rank-indexed cell diameters with shape ``[num_rank_cells, 1]``.
    """

    def __init__(
        self,
        vertex_coordinates: torch.Tensor,
        vertex_memberships: dict[int, torch.Tensor],
        centroids: dict[int, torch.Tensor],
        diameters: dict[int, torch.Tensor],
    ) -> None:
        self.vertex_coordinates = vertex_coordinates
        self.vertex_memberships = vertex_memberships
        self.centroids = centroids
        self.diameters = diameters


def _validate_physical_coordinates(
    batch,
    coordinate_attr: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate and return rank-0 physical coordinates.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch containing rank-0 features and physical
        coordinates.
    coordinate_attr : str
        Name of the rank-0 coordinate attribute, usually ``"pos"``.
    device : torch.device
        Device on which the returned coordinate tensor should live.
    dtype : torch.dtype
        Floating dtype for returned coordinates.

    Returns
    -------
    torch.Tensor
        Rank-0 coordinate matrix with shape ``[num_vertices, coordinate_dim]``.
    """
    rank_0_features = getattr(batch, "x_0", None)
    if rank_0_features is None:
        raise AttributeError(
            "Physical ETNN coordinates require rank-0 features at `x_0`."
        )
    coordinates = getattr(batch, coordinate_attr, None)
    if coordinates is None:
        raise AttributeError(
            "Physical ETNN coordinate policy expected coordinates at "
            f"`{coordinate_attr}`."
        )

    coordinates = coordinates.to(device=device, dtype=dtype)
    if coordinates.ndim != 2:
        raise ValueError(
            "Physical ETNN coordinate policy expected a coordinate matrix at "
            f"`{coordinate_attr}`, but found shape {tuple(coordinates.shape)}."
        )
    if coordinates.shape[0] != rank_0_features.shape[0]:
        raise ValueError(
            "Physical ETNN coordinate policy expected one coordinate row per "
            f"rank-0 cell, but found {coordinates.shape[0]} coordinates for "
            f"{rank_0_features.shape[0]} rank-0 cells."
        )
    return coordinates


def _build_vertex_memberships(
    batch,
    max_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Build vertex-to-cell membership matrices for every visible rank.

    TopoBench combinatorial liftings store ``incidence_r`` with rows for
    rank ``r-1`` cells and columns for rank ``r`` cells.  Physical ETNN
    invariants, however, are defined from the physical coordinates of the
    incident rank-0 vertices of each cell.  This helper composes incidence
    matrices so every rank receives a binary vertex-membership matrix.

    The returned matrix for rank ``r`` has shape
    ``[num_rank0_cells, num_rank_r_cells]``.  Entry ``(v, c)`` is one if
    vertex ``v`` belongs to cell ``c`` and zero otherwise.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch containing rank-wise features and incidence
        matrices.
    max_rank : int
        Highest rank for which memberships should be constructed.
    device : torch.device
        Device for returned membership tensors.
    dtype : torch.dtype
        Floating dtype for returned membership tensors.

    Returns
    -------
    dict[int, torch.Tensor]
        Rank-indexed dense binary membership matrices.
    """
    # The visible ranks must be well-defined before composing incidences.
    if max_rank < 0:
        raise ValueError("`max_rank` must be non-negative.")

    # Rank-0 features define the number and ordering of vertices. The physical
    # coordinate tensor is checked separately by _validate_physical_coordinates.
    rank_0_features = getattr(batch, "x_0", None)
    if rank_0_features is None:
        raise AttributeError(
            "Physical ETNN membership construction requires `x_0`."
        )

    # Rank-0 membership is the identity: each vertex is incident to itself.
    num_vertices = rank_0_features.shape[0]
    memberships: dict[int, torch.Tensor] = {
        0: torch.eye(num_vertices, device=device, dtype=dtype)
    }

    for rank in range(1, max_rank + 1):
        # Each higher rank needs both a feature tensor and an incidence matrix.
        # Missing either means the physical cell geometry is under-specified.
        feature_key = f"x_{rank}"
        rank_features = getattr(batch, feature_key, None)
        if rank_features is None:
            raise AttributeError(
                "Physical ETNN membership construction expected "
                f"rank-{rank} features at `{feature_key}`."
            )

        incidence_key = f"incidence_{rank}"
        incidence = getattr(batch, incidence_key, None)
        if incidence is None:
            raise AttributeError(
                "Physical ETNN membership construction needs incidence "
                f"matrix `{incidence_key}`."
            )

        # incidence_r maps lower-rank cells to rank-r cells.  Multiplying the
        # previous vertex-to-lower-cell membership by |incidence_r| gives
        # vertex-to-rank-r incidence.  Binarize after multiplication because
        # a vertex can reach a higher-order cell through multiple lower cells.
        incidence = _dense_absolute_incidence(
            incidence=incidence,
            device=device,
            dtype=dtype,
        )
        lower_membership = memberships[rank - 1]
        num_rank_cells = rank_features.shape[0]

        # Validate sparse axes before multiplying. An axis mismatch would
        # attach vertex sets to the wrong cells.
        if incidence.shape[0] != lower_membership.shape[1]:
            raise ValueError(
                "Cannot compose physical ETNN memberships: "
                f"`{incidence_key}` has {incidence.shape[0]} source rows, "
                f"but rank {rank - 1} has {lower_membership.shape[1]} cells."
            )
        if incidence.shape[1] != num_rank_cells:
            raise ValueError(
                "Cannot compose physical ETNN memberships: "
                f"`{incidence_key}` has {incidence.shape[1]} target columns, "
                f"but rank {rank} has {num_rank_cells} cells."
            )

        # Binarize because a vertex can reach a cell through multiple lower
        # cells. Membership answers "is incident", not "how many paths".
        memberships[rank] = (lower_membership @ incidence > 0).to(dtype)

    return memberships


def _compute_cell_centroids(
    vertex_coordinates: torch.Tensor,
    vertex_memberships: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Compute physical centroids for every cell rank.

    Parameters
    ----------
    vertex_coordinates : torch.Tensor
        Rank-0 physical coordinates with shape
        ``[num_vertices, coordinate_dim]``.
    vertex_memberships : dict[int, torch.Tensor]
        Rank-indexed vertex-membership matrices.

    Returns
    -------
    dict[int, torch.Tensor]
        Rank-indexed centroid tensors.
    """
    centroids = {}
    for rank, membership in vertex_memberships.items():
        if membership.shape[0] != vertex_coordinates.shape[0]:
            raise ValueError(
                "Cannot compute physical ETNN centroids: membership for "
                f"rank {rank} has {membership.shape[0]} vertex rows, but "
                f"coordinates have {vertex_coordinates.shape[0]} rows."
            )

        num_cells = membership.shape[1]
        if num_cells == 0:
            # Preserve empty ranks with the right coordinate dimension.
            centroids[rank] = vertex_coordinates.new_empty(
                (0, vertex_coordinates.shape[1])
            )
            continue

        # Weighted sums are just ordinary incidence averages because
        # memberships are binary after incidence composition.
        counts = membership.sum(dim=0, keepdim=True).T
        coordinate_sums = membership.T @ vertex_coordinates

        # NSAPH stores explicit cell memberships.  A nonempty TopoBench cell
        # with no recovered vertices means the incidence reconstruction is
        # malformed, so fail rather than attach arbitrary zero geometry.
        if torch.any(counts == 0):
            bad = torch.nonzero(counts.flatten() == 0, as_tuple=False)
            raise ValueError(
                "Cannot compute physical ETNN centroids: rank "
                f"{rank} has {bad.numel()} nonempty cells with no incident "
                "rank-0 vertices."
            )

        centroids[rank] = coordinate_sums / counts
    return centroids


def _compute_cell_diameters(
    vertex_coordinates: torch.Tensor,
    vertex_memberships: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Compute max internal physical distance for every cell.

    Parameters
    ----------
    vertex_coordinates : torch.Tensor
        Rank-0 physical coordinates with shape
        ``[num_vertices, coordinate_dim]``.
    vertex_memberships : dict[int, torch.Tensor]
        Rank-indexed vertex-membership matrices.

    Returns
    -------
    dict[int, torch.Tensor]
        Rank-indexed diameter tensors with shape ``[num_cells, 1]``.
    """
    diameters = {}
    for rank, membership in vertex_memberships.items():
        if membership.shape[0] != vertex_coordinates.shape[0]:
            raise ValueError(
                "Cannot compute physical ETNN diameters: membership for "
                f"rank {rank} has {membership.shape[0]} vertex rows, but "
                f"coordinates have {vertex_coordinates.shape[0]} rows."
            )

        rank_diameters = vertex_coordinates.new_zeros((membership.shape[1], 1))
        for cell_idx in range(membership.shape[1]):
            # Select the physical vertices belonging to this cell. Vertices are
            # recovered from membership, not from cell feature values.
            vertex_idx = torch.nonzero(
                membership[:, cell_idx] > 0, as_tuple=False
            ).flatten()
            if vertex_idx.numel() <= 1:
                # A vertex cell or degenerate singleton has zero diameter.
                continue

            # Cell diameter is the maximum pairwise physical distance among
            # incident vertices.  This is intentionally computed from rank-0
            # coordinates, matching the original physical ETNN interpretation.
            cell_coordinates = vertex_coordinates[vertex_idx]
            rank_diameters[cell_idx, 0] = torch.cdist(
                cell_coordinates,
                cell_coordinates,
            ).max()

        diameters[rank] = rank_diameters
    return diameters


def _build_physical_cell_geometry(
    batch,
    coordinate_attr: str,
    max_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _PhysicalCellGeometry:
    """Construct rank-wise physical cell geometry from rank-0 coordinates.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch with rank-wise features, incidence matrices, and
        physical rank-0 coordinates.
    coordinate_attr : str
        Batch attribute containing rank-0 coordinates, usually ``"pos"``.
    max_rank : int
        Highest visible cell rank.
    device : torch.device
        Device for returned tensors.
    dtype : torch.dtype
        Floating dtype for returned tensors.

    Returns
    -------
    _PhysicalCellGeometry
        Rank-wise centroids and diameters.
    """
    # First ensure the requested coordinate attribute is present and aligned
    # with rank-0 cells; all later geometry assumes this ordering.
    vertex_coordinates = _validate_physical_coordinates(
        batch=batch,
        coordinate_attr=coordinate_attr,
        device=device,
        dtype=dtype,
    )

    # Then recover which rank-0 vertices belong to each higher-rank cell.
    vertex_memberships = _build_vertex_memberships(
        batch=batch,
        max_rank=max_rank,
        device=device,
        dtype=dtype,
    )

    # Finally summarize every cell by centroid and diameter.
    return _summarize_physical_cell_geometry(
        vertex_coordinates=vertex_coordinates,
        vertex_memberships=vertex_memberships,
    )


def _summarize_physical_cell_geometry(
    vertex_coordinates: torch.Tensor,
    vertex_memberships: dict[int, torch.Tensor],
) -> _PhysicalCellGeometry:
    """Summarize cells from current rank-0 physical coordinates.

    This helper is separated from batch validation because ``pos_update=True``
    changes rank-0 coordinates between layers while the topology and
    cell-to-vertex memberships stay fixed.

    Parameters
    ----------
    vertex_coordinates : torch.Tensor
        Current rank-0 physical coordinates.
    vertex_memberships : dict[int, torch.Tensor]
        Rank-indexed vertex-to-cell membership matrices.

    Returns
    -------
    _PhysicalCellGeometry
        Rank-wise centroids and diameters.
    """
    return _PhysicalCellGeometry(
        vertex_coordinates=vertex_coordinates,
        vertex_memberships=vertex_memberships,
        centroids=_compute_cell_centroids(
            vertex_coordinates=vertex_coordinates,
            vertex_memberships=vertex_memberships,
        ),
        diameters=_compute_cell_diameters(
            vertex_coordinates=vertex_coordinates,
            vertex_memberships=vertex_memberships,
        ),
    )


def _physical_relation_invariants(
    src_centroids: torch.Tensor,
    dst_centroids: torch.Tensor,
    src_diameters: torch.Tensor,
    dst_diameters: torch.Tensor,
    src_membership: torch.Tensor,
    dst_membership: torch.Tensor,
    vertex_coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    dtype: torch.dtype,
    hausdorff_dists: bool,
) -> torch.Tensor:
    """Compute physical ETNN invariant features for relation edges.

    Parameters
    ----------
    src_centroids : torch.Tensor
        Centroids for sender-rank cells.
    dst_centroids : torch.Tensor
        Centroids for receiver-rank cells.
    src_diameters : torch.Tensor
        Diameters for sender-rank cells.
    dst_diameters : torch.Tensor
        Diameters for receiver-rank cells.
    src_membership : torch.Tensor
        Vertex-to-sender-cell membership matrix.
    dst_membership : torch.Tensor
        Vertex-to-receiver-cell membership matrix.
    vertex_coordinates : torch.Tensor
        Rank-0 physical coordinates.
    edge_index : torch.Tensor
        Relation edges in ``[sender, receiver]`` format.
    dtype : torch.dtype
        Floating dtype for returned invariant features.
    hausdorff_dists : bool
        Whether to append directed Hausdorff-style distances.

    Returns
    -------
    torch.Tensor
        Invariant edge features.  The first three columns are
        ``[centroid_distance, sender_diameter, receiver_diameter]``.  When
        ``hausdorff_dists=True``, two more columns store directed Hausdorff-style
        distances from sender cell to receiver cell and back.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "Physical ETNN relation invariants expected `edge_index` with "
            f"shape [2, num_edges], but found {tuple(edge_index.shape)}."
        )
    if edge_index.numel() == 0:
        # Preserve the expected physical edge-channel width even for empty
        # relations so downstream concatenation remains shape-stable.
        width = 5 if hausdorff_dists else 3
        return src_centroids.new_empty((0, width), dtype=dtype)

    edge_index = edge_index.to(src_centroids.device)
    sender, receiver = edge_index

    # Centroid distance is invariant to translation, rotation, and reflection.
    centroid_distance = torch.linalg.vector_norm(
        src_centroids[sender] - dst_centroids[receiver],
        dim=-1,
        keepdim=True,
    )

    # Sender/receiver diameters describe the size of the incident cells. They
    # are also E(n)-invariant scalar summaries of physical cell geometry.
    base_invariants = torch.cat(
        [
            centroid_distance,
            src_diameters[sender],
            dst_diameters[receiver],
        ],
        dim=-1,
    ).to(dtype)

    if not hausdorff_dists:
        return base_invariants

    directed_hausdorff = _directed_relation_hausdorff_distances(
        src_membership=src_membership,
        dst_membership=dst_membership,
        vertex_coordinates=vertex_coordinates,
        edge_index=edge_index,
        dtype=dtype,
    )
    return torch.cat([base_invariants, directed_hausdorff], dim=-1)


def _normalize_physical_invariants(
    invariant_attr: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize physical invariant channels by batch-local channel scale.

    Physical invariant channels are distances and diameters, so their absolute
    scale can vary across molecular systems or lifted batches.  This lightweight
    normalization divides each channel by its mean absolute value over the
    relation edges in the current batch.  The operation preserves E(n)
    invariance because it depends only on invariant scalar channels.

    Parameters
    ----------
    invariant_attr : torch.Tensor
        Physical invariant edge attributes with shape
        ``[num_edges, num_invariants]``.
    eps : float
        Numerical floor for the mean absolute channel scale.

    Returns
    -------
    torch.Tensor
        Mean-absolute normalized invariant attributes.
    """
    if invariant_attr.numel() == 0:
        return invariant_attr
    scale = invariant_attr.detach().abs().mean(dim=0, keepdim=True)
    return invariant_attr / scale.clamp_min(eps)


def _batch_norm_physical_invariants(
    invariant_attr: torch.Tensor,
    normalizer: nn.BatchNorm1d,
) -> torch.Tensor:
    """Apply NSAPH-style BatchNorm to physical invariant channels.

    NSAPH normalizes invariant edge attributes with per-adjacency
    ``BatchNorm1d(..., affine=False)`` modules.  PyTorch's training-mode
    BatchNorm requires at least two rows while fitting training statistics, so
    singleton training-mode relations are passed through unchanged.  Empty
    relations have no values to normalize in either mode.  In eval mode,
    singleton relations can still use stored running statistics safely.

    Parameters
    ----------
    invariant_attr : torch.Tensor
        Physical invariant edge attributes with shape
        ``[num_edges, num_invariants]``.
    normalizer : nn.BatchNorm1d
        Per-relation BatchNorm module with ``affine=False``.

    Returns
    -------
    torch.Tensor
        Batch-normalized invariant attributes when safe, otherwise unchanged
        empty or singleton training-mode attributes.
    """
    if invariant_attr.shape[0] == 0:
        return invariant_attr
    if invariant_attr.shape[0] == 1 and normalizer.training:
        return invariant_attr
    return normalizer(invariant_attr)


def _directed_relation_hausdorff_distances(
    src_membership: torch.Tensor,
    dst_membership: torch.Tensor,
    vertex_coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute directed Hausdorff-style distances for physical relation edges.

    For each relation edge from sender cell ``d`` to receiver cell ``c``, this
    helper computes:

        H(d,c) = max_{u in d} min_{v in c} ||pos_u - pos_v||
        H(c,d) = max_{v in c} min_{u in d} ||pos_v - pos_u||

    NSAPH uses these two directed distances as optional physical invariant
    channels.  They are invariant to translation, rotation, and reflection
    because they are built only from pairwise Euclidean distances.

    Parameters
    ----------
    src_membership : torch.Tensor
        Dense vertex-to-sender-cell membership matrix with shape
        ``[num_vertices, num_sender_cells]``.
    dst_membership : torch.Tensor
        Dense vertex-to-receiver-cell membership matrix with shape
        ``[num_vertices, num_receiver_cells]``.
    vertex_coordinates : torch.Tensor
        Rank-0 physical coordinates with shape ``[num_vertices, coord_dim]``.
    edge_index : torch.Tensor
        Relation edges in ``[sender_cell, receiver_cell]`` format.
    dtype : torch.dtype
        Floating dtype for the returned distance tensor.

    Returns
    -------
    torch.Tensor
        Directed Hausdorff-style distances with shape ``[num_edges, 2]``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "Physical ETNN Hausdorff distances expected `edge_index` with "
            f"shape [2, num_edges], but found {tuple(edge_index.shape)}."
        )
    if src_membership.shape[0] != vertex_coordinates.shape[0]:
        raise ValueError(
            "Sender membership vertex axis does not match physical coordinates."
        )
    if dst_membership.shape[0] != vertex_coordinates.shape[0]:
        raise ValueError(
            "Receiver membership vertex axis does not match physical coordinates."
        )
    if edge_index.numel() == 0:
        return vertex_coordinates.new_empty((0, 2), dtype=dtype)

    edge_index = edge_index.to(src_membership.device)
    sender, receiver = edge_index
    distances = vertex_coordinates.new_zeros((edge_index.shape[1], 2))

    # The loop is intentionally simple. Physical mode targets molecular and
    # small/medium geometric complexes where cell vertex sets are small; keeping
    # this explicit is easier to audit than a dense all-cell all-cell tensor.
    for edge_pos, (src_cell, dst_cell) in enumerate(
        zip(sender.tolist(), receiver.tolist(), strict=True)
    ):
        src_vertices = torch.nonzero(
            src_membership[:, src_cell] > 0,
            as_tuple=False,
        ).flatten()
        dst_vertices = torch.nonzero(
            dst_membership[:, dst_cell] > 0,
            as_tuple=False,
        ).flatten()
        if src_vertices.numel() == 0 or dst_vertices.numel() == 0:
            raise ValueError(
                "Cannot compute physical ETNN Hausdorff distances for cells "
                "with no incident rank-0 vertices."
            )

        pairwise_distances = torch.cdist(
            vertex_coordinates[src_vertices],
            vertex_coordinates[dst_vertices],
        )
        distances[edge_pos, 0] = pairwise_distances.min(dim=1).values.max()
        distances[edge_pos, 1] = pairwise_distances.min(dim=0).values.max()

    return distances.to(dtype)


def _dense_absolute_incidence(
    incidence: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a dense absolute incidence matrix.

    Physical cell membership ignores orientation signs.  TopoBench incidence
    tensors are sparse COO matrices, so this helper also centralizes sparse
    validation before the physical-coordinate path relies on their axes.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse COO incidence matrix.
    device : torch.device
        Device on which the returned dense matrix should live.
    dtype : torch.dtype
        Floating dtype for the returned dense matrix.

    Returns
    -------
    torch.Tensor
        Dense nonnegative incidence matrix with the same shape as
        ``incidence``.
    """
    if not incidence.is_sparse:
        raise TypeError(
            "Physical ETNN coordinate policy expected sparse COO incidence "
            f"matrices, but received a dense tensor with shape "
            f"{tuple(incidence.shape)}."
        )

    # Absolute values remove orientation signs from boundary operators. Cell
    # membership only needs incidence, not orientation.
    incidence = incidence.coalesce().to(device)
    dense = incidence.to_dense().abs().to(dtype=dtype)
    return dense
