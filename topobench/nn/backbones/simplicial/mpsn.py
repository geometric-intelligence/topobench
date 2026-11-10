"""Sparse boundary-plus-upper SIN/MPSN backbone.

This module implements the concrete Simplicial Isomorphism Network (SIN)
update from Supplement Equation 35 of Bodnar et al. [1]_.  For each simplex,
the boundary and upper-adjacency multisets are transformed independently.  An
upper message receives the *paired* features of an upper neighbour and their
shared coface before aggregation.

Routing is derived solely from the support of the two incidence matrices. It
is therefore independent of incidence signs and orientations, and it never
constructs a dense or square adjacency matrix. Explicitly stored sparse zeros
are excluded from the incidence support. For sparse input on a rank-two
simplicial complex with ``I`` nonzero incidences and ``P`` ordered upper routes,
construction takes ``O(I log I + P)`` time and ``O(I + P)`` memory. Dense CPU
input is supported, but extracting its support necessarily scans the dense
incidence storage. Accelerator incidence must be sparse.

The boundary, upper-adjacency, and generic message-passing definitions are
given by main-paper Equations 2, 5, and 6, respectively.  Theorem 6 shows that
boundary and upper messages suffice for the corresponding colour refinement.
Theorem 10's expressivity conclusion is conditional on injective aggregation;
this finite learned network does not claim guaranteed injectivity or automatic
equivalence to the full SWL procedure.  The implementation was derived from
the paper and checked against the authors' public ``SparseCINCochainConv`` and
``SparseCINConv`` implementation.

References
----------
.. [1] Bodnar, Frasca, Wang, Otter, Montufar, Lio, Bronstein. "Weisfeiler and
   Lehman Go Topological: Message Passing Simplicial Networks." ICML 2021.
   Supplement Eq. 35. https://proceedings.mlr.press/v139/bodnar21a.html
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _IncidenceRoutes:
    """Stored support of one face-to-coface incidence matrix."""

    face: torch.Tensor
    coface: torch.Tensor


@dataclass(frozen=True)
class _RankRoutes:
    """Sparse senders and receivers used to update one simplex rank."""

    boundary_sender: torch.Tensor
    boundary_receiver: torch.Tensor
    upper_receiver: torch.Tensor
    upper_neighbor: torch.Tensor
    upper_coface: torch.Tensor


@dataclass(frozen=True)
class _SimplicialRoutes:
    """Routing for all ranks, built once and reused by every MPSN layer."""

    by_rank: tuple[_RankRoutes, _RankRoutes, _RankRoutes]


def _incidence_routes(incidence: torch.Tensor) -> _IncidenceRoutes:
    """Extract face/coface indices from dense or sparse incidence support.

    Incidence signs and magnitudes are intentionally ignored: signed and
    unsigned matrices with the same nonzero support define the same topology.
    Explicitly stored sparse zeros are filtered from that support. Dense
    incidence is supported on CPU only; accelerators require a sparse tensor so
    extracting support cannot introduce a device-to-host synchronization.

    Parameters
    ----------
    incidence : torch.Tensor
        Rank-two face-to-coface incidence matrix. Dense CPU and sparse tensors
        on any supported device are accepted.

    Returns
    -------
    _IncidenceRoutes
        Row (face) and column (coface) indices of nonzero entries.

    Raises
    ------
    ValueError
        If a dense accelerator incidence tensor is supplied.
    """
    if incidence.layout == torch.strided:
        if incidence.device.type != "cpu":
            msg = (
                "Dense accelerator incidence is unsupported; provide sparse "
                "incidence for accelerator routing."
            )
            raise ValueError(msg)
        indices = torch.nonzero(incidence, as_tuple=False).transpose(0, 1)
    else:
        coalesced = incidence.to_sparse_coo().coalesce()
        indices = coalesced.indices()[:, coalesced.values() != 0]
    return _IncidenceRoutes(face=indices[0], coface=indices[1])


def _upper_routes(
    incidence: _IncidenceRoutes,
    faces_per_coface: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand grouped incidences into ordered upper-neighbour routes.

    A valid simplicial incidence has exactly ``faces_per_coface`` entries per
    column (two vertices per edge or three edges per triangle).  Grouping by
    coface and applying a fixed-size off-diagonal expansion produces the two
    ordered vertex routes per edge and six ordered edge routes per triangle.

    Parameters
    ----------
    incidence : _IncidenceRoutes
        Nonzero face/coface incidence indices grouped by this function.
    faces_per_coface : int
        Required number of faces in each coface: two for edges or three for
        triangles.

    Returns
    -------
    tuple of torch.Tensor
        Receiver-simplex, neighbour-simplex, and shared-coface indices for
        every ordered upper route.
    """
    order = torch.argsort(incidence.coface)
    grouped_faces = incidence.face[order].reshape(-1, faces_per_coface)
    grouped_cofaces = incidence.coface[order].reshape(-1, faces_per_coface)
    off_diagonal = ~torch.eye(
        faces_per_coface,
        dtype=torch.bool,
        device=incidence.face.device,
    )

    receiver = grouped_faces[:, :, None].expand(
        -1, faces_per_coface, faces_per_coface
    )[:, off_diagonal]
    neighbor = grouped_faces[:, None, :].expand(
        -1, faces_per_coface, faces_per_coface
    )[:, off_diagonal]
    route_count = faces_per_coface * (faces_per_coface - 1)
    coface = grouped_cofaces[:, :1].expand(-1, route_count)
    return receiver.reshape(-1), neighbor.reshape(-1), coface.reshape(-1)


def _build_routes(
    incidence_1: torch.Tensor,
    incidence_2: torch.Tensor,
) -> _SimplicialRoutes:
    """Build sparse boundary and upper routes for ranks zero through two.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Vertex-to-edge incidence matrix.
    incidence_2 : torch.Tensor
        Edge-to-triangle incidence matrix.

    Returns
    -------
    _SimplicialRoutes
        Boundary and ordered upper routes for all three simplex ranks.
    """
    b1 = _incidence_routes(incidence_1)
    b2 = _incidence_routes(incidence_2)
    empty = b1.face.new_empty(0)

    upper_0 = _upper_routes(b1, faces_per_coface=2)
    upper_1 = _upper_routes(b2, faces_per_coface=3)
    return _SimplicialRoutes(
        by_rank=(
            _RankRoutes(empty, empty, *upper_0),
            _RankRoutes(b1.face, b1.coface, *upper_1),
            _RankRoutes(b2.face, b2.coface, empty, empty, empty),
        )
    )


def _scatter_sum(
    messages: torch.Tensor,
    receiver: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Sum messages at receiver indices without materializing adjacency.

    Parameters
    ----------
    messages : torch.Tensor
        Route messages shaped ``[num_routes, hidden_dim]``.
    receiver : torch.Tensor
        Destination simplex index for every route.
    output_size : int
        Number of destination simplices.

    Returns
    -------
    torch.Tensor
        Summed messages shaped ``[output_size, hidden_dim]``.
    """
    output = messages.new_zeros((output_size, messages.shape[-1]))
    return output.index_add_(0, receiver, messages)


def _two_layer_mlp(input_dim: int, output_dim: int) -> torch.nn.Sequential:
    """Return the two-layer ELU perceptron used by Eq. 35 branches.

    Parameters
    ----------
    input_dim : int
        Input feature width.
    output_dim : int
        Hidden and output feature width.

    Returns
    -------
    torch.nn.Sequential
        Two linear layers, each followed by ELU.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.ELU(),
        torch.nn.Linear(output_dim, output_dim),
        torch.nn.ELU(),
    )


def _linear_elu(input_dim: int, output_dim: int) -> torch.nn.Sequential:
    """Return the dense-layer-plus-ELU maps used by Eq. 35.

    Parameters
    ----------
    input_dim : int
        Input feature width.
    output_dim : int
        Output feature width.

    Returns
    -------
    torch.nn.Sequential
        One linear layer followed by ELU.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.ELU(),
    )


class _SINRankUpdate(torch.nn.Module):
    """Equation 35 update for one rank in one layer.

    All applicable maps are private to this rank/layer; the top rank omits the
    structurally unreachable pair-message map.  Both self coefficients are
    fixed to ``1 + eps = 1`` as in the SIN experiment; they are not parameters.
    Empty neighbourhoods contribute an all-zero aggregate while their branch
    still transforms the simplex's own representation.

    Parameters
    ----------
    hidden_dim : int
        Feature width shared by all simplex ranks.
    has_upper_messages : bool
        Whether this rank can have higher-dimensional cofaces and therefore
        needs the pair-message map.
    """

    def __init__(self, hidden_dim: int, has_upper_messages: bool) -> None:
        super().__init__()
        self.boundary_mlp = _two_layer_mlp(hidden_dim, hidden_dim)
        self.upper_message_mlp = (
            _linear_elu(2 * hidden_dim, hidden_dim)
            if has_upper_messages
            else None
        )
        self.upper_mlp = _two_layer_mlp(hidden_dim, hidden_dim)
        self.combine_mlp = _linear_elu(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        x_self: torch.Tensor,
        x_boundary: torch.Tensor | None,
        x_coface: torch.Tensor | None,
        routes: _RankRoutes,
    ) -> torch.Tensor:
        """Apply the separate boundary and upper branches for one rank.

        Parameters
        ----------
        x_self : torch.Tensor
            Features for the simplices being updated.
        x_boundary : torch.Tensor or None
            Features for their boundary faces, or ``None`` at rank zero.
        x_coface : torch.Tensor or None
            Features for their cofaces, or ``None`` at the top rank.
        routes : _RankRoutes
            Sparse boundary and ordered upper routes for this rank.

        Returns
        -------
        torch.Tensor
            Updated simplex features with the same shape as ``x_self``.
        """
        boundary_sum = x_self.new_zeros(x_self.shape)
        if x_boundary is not None:
            boundary_sum = _scatter_sum(
                x_boundary[routes.boundary_sender],
                routes.boundary_receiver,
                x_self.shape[0],
            )

        upper_sum = x_self.new_zeros(x_self.shape)
        if x_coface is not None:
            if self.upper_message_mlp is None:
                msg = "Upper cofaces require an upper-message map."
                raise RuntimeError(msg)
            upper_inputs = torch.cat(
                (
                    x_self[routes.upper_neighbor],
                    x_coface[routes.upper_coface],
                ),
                dim=-1,
            )
            upper_sum = _scatter_sum(
                self.upper_message_mlp(upper_inputs),
                routes.upper_receiver,
                x_self.shape[0],
            )

        boundary_branch = self.boundary_mlp(x_self + boundary_sum)
        upper_branch = self.upper_mlp(x_self + upper_sum)
        return self.combine_mlp(
            torch.cat((boundary_branch, upper_branch), dim=-1)
        )


class _MPSNLayer(torch.nn.Module):
    """One sparse boundary-plus-upper SIN layer for ranks zero to two.

    Parameters
    ----------
    hidden_dim : int
        Feature width shared by all simplex ranks.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank_updates = torch.nn.ModuleList(
            _SINRankUpdate(hidden_dim, has_upper_messages=rank < 2)
            for rank in range(3)
        )

    def forward(
        self,
        x_all: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        routes: _SimplicialRoutes,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update all ranks from the same precomputed sparse routes.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Hidden features for vertices, edges, and triangles, each shaped
            ``[num_simplices_at_rank, hidden_dim]``.
        routes : _SimplicialRoutes
            Boundary and ordered upper routes shared by all network layers.

        Returns
        -------
        tuple of torch.Tensor
            Updated vertex, edge, and triangle features with the same shapes as
            ``x_all``.
        """
        x_0, x_1, x_2 = x_all
        return (
            self.rank_updates[0](x_0, None, x_1, routes.by_rank[0]),
            self.rank_updates[1](x_1, x_0, x_2, routes.by_rank[1]),
            self.rank_updates[2](x_2, x_1, None, routes.by_rank[2]),
        )


class MPSN(torch.nn.Module):
    """Boundary-plus-upper SIN instantiation of the MPSN framework.

    The public TopoBench seam is unchanged: input and output features are
    tuples ordered by simplex rank, while topology is provided by node-edge
    and edge-triangle incidence matrices.  The sparse routes are constructed
    once per forward pass and reused by every layer.

    This is the concrete update from Supplement Equation 35, adapted to
    TopoBench's rank-zero-through-two interface.  The architecture uses the
    boundary and upper neighbourhoods justified by Theorem 6.  It does not
    assert the injective-aggregation premise required by Theorem 10.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input feature dimensions on nodes, edges, and triangles.
    hidden_dim : int
        Hidden width shared by all ranks.
    n_layers : int
        Number of stacked SIN layers.
    """

    def __init__(
        self,
        in_channels_all: tuple[int, int, int],
        hidden_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_linear_0 = torch.nn.Linear(in_channels_all[0], hidden_dim)
        self.in_linear_1 = torch.nn.Linear(in_channels_all[1], hidden_dim)
        self.in_linear_2 = torch.nn.Linear(in_channels_all[2], hidden_dim)
        self.layers = torch.nn.ModuleList(
            _MPSNLayer(hidden_dim) for _ in range(n_layers)
        )

    def forward(
        self,
        x_all: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        incidence_all: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project features, build routes once, then apply all SIN layers.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Features ``(x_0, x_1, x_2)`` shaped ``[N_r, in_channels_r]`` for
            vertices, edges, and triangles.
        incidence_all : tuple of torch.Tensor
            Face-to-coface matrices ``(B_1, B_2)`` with shapes ``[N_0, N_1]``
            and ``[N_1, N_2]``. They may be signed or unsigned. Sparse tensors
            are accepted on any device; dense tensors are CPU-only. Each valid
            edge column of ``B_1`` must contain two nonzeros and each valid
            triangle column of ``B_2`` must contain three nonzeros.

        Returns
        -------
        tuple of torch.Tensor
            Hidden features ``(h_0, h_1, h_2)``, with each tensor shaped
            ``[N_r, hidden_dim]``.

        Raises
        ------
        ValueError
            If either incidence matrix is dense on an accelerator.
        """
        routes = _build_routes(incidence_all[0], incidence_all[1])
        x_0, x_1, x_2 = x_all
        hidden_all = (
            self.in_linear_0(x_0),
            self.in_linear_1(x_1),
            self.in_linear_2(x_2),
        )
        for layer in self.layers:
            hidden_all = layer(hidden_all, routes)
        return hidden_all
