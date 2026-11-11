r"""Message Passing Simplicial Networks (MPSN).

Faithful TopoBench implementation of the simplicial message-passing model of

    C. Bodnar, F. Frasca, Y. G. Wang, N. Otter, G. Montufar, P. Lio and
    M. Bronstein. "Weisfeiler and Lehman Go Topological: Message Passing
    Simplicial Networks." ICML 2021. arXiv:2103.03212.

The model operates on a simplicial complex and, following the paper's
Definition 4, propagates information along the four adjacency types defined
for a simplex :math:`\sigma`:

* Boundary     :math:`\mathcal{B}(\sigma)=\{\tau\mid\tau\prec\sigma\}`
  (faces, rank :math:`k-1`).
* Co-boundary  :math:`\mathcal{C}(\sigma)=\{\tau\mid\sigma\prec\tau\}`
  (co-faces, rank :math:`k+1`).
* Lower        :math:`\mathcal{N}_\downarrow(\sigma)=
  \{\tau\mid\exists\delta,\ \delta\prec\sigma\wedge\delta\prec\tau\}`
  (same rank, share a face :math:`\delta`).
* Upper        :math:`\mathcal{N}_\uparrow(\sigma)=
  \{\tau\mid\exists\delta,\ \sigma\prec\delta\wedge\tau\prec\delta\}`
  (same rank, share a co-face :math:`\delta`).

For a simplex :math:`\sigma` of rank :math:`k`, layer :math:`t` computes the
messages and update of the paper's Eqs. (1)-(5):

.. math::

    m_{\mathcal{B}}(\sigma)  &= \textstyle\sum_{\tau\in\mathcal{B}(\sigma)}
        M_{\mathcal{B}}(h_\sigma, h_\tau) \\
    m_{\mathcal{C}}(\sigma)  &= \textstyle\sum_{\tau\in\mathcal{C}(\sigma)}
        M_{\mathcal{C}}(h_\sigma, h_\tau) \\
    m_{\downarrow}(\sigma)   &= \textstyle\sum_{\tau\in\mathcal{N}_\downarrow(\sigma)}
        M_{\downarrow}(h_\sigma, h_\tau, h_\delta) \\
    m_{\uparrow}(\sigma)     &= \textstyle\sum_{\tau\in\mathcal{N}_\uparrow(\sigma)}
        M_{\uparrow}(h_\sigma, h_\tau, h_\delta) \\
    h_\sigma^{t+1}           &= U\!\left(h_\sigma, m_{\mathcal{B}},
        m_{\mathcal{C}}, m_{\downarrow}, m_{\uparrow}\right)

where the aggregation is a permutation-invariant sum, :math:`\delta` is the
shared face / co-face carried by the lower / upper messages, and
:math:`M_\bullet, U` are MLPs. A global readout over the per-rank multisets
(paper Eq. (6)) is handled downstream by the TopoBench readout module.

Messages are routed on the *unsigned* incidence pattern, giving the
orientation-invariant MPSN variant discussed in the paper (the "absolute
value" models used for the graph-lifted benchmarks, where a reference
orientation is arbitrary). Only the boundary (incidence) matrices
:math:`B_1` (nodes-edges) and :math:`B_2` (edges-faces) are required as
input; all four adjacencies are derived from them.
"""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {
    None: nn.Identity,
    "id": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def _get_activation(name: str | None) -> nn.Module:
    """Return an activation module by name.

    Parameters
    ----------
    name : str or None
        Activation identifier. One of ``None``, ``"id"``, ``"relu"``,
        ``"sigmoid"``, ``"tanh"`` or ``"elu"``.

    Returns
    -------
    torch.nn.Module
        The instantiated activation module.
    """
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from {sorted(k for k in _ACTIVATIONS if k)}."
        )
    return _ACTIVATIONS[name]()


def _mlp(in_channels: int, out_channels: int, activation: str) -> nn.Module:
    """Build a two-layer message/update MLP.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output (and hidden) features.
    activation : str
        Name of the intermediate activation function.

    Returns
    -------
    torch.nn.Module
        A ``Linear -> activation -> Linear`` block.
    """
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        _get_activation(activation),
        nn.Linear(out_channels, out_channels),
    )


def _coo_indices(matrix: torch.Tensor) -> torch.Tensor:
    """Return the ``[2, nnz]`` index tensor of a sparse or dense matrix.

    Only the sparsity pattern is used (values, and therefore orientation
    signs, are ignored), yielding orientation-invariant message routing.

    Parameters
    ----------
    matrix : torch.Tensor
        A (possibly sparse) incidence matrix.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[2, nnz]`` with ``(row, col)`` indices.
    """
    if matrix.is_sparse:
        return matrix.coalesce().indices()
    return matrix.nonzero(as_tuple=False).t().contiguous()


def _ordered_pairs(
    group: torch.Tensor, member: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate ordered same-group member pairs and their shared element.

    Given incidences ``(group[i], member[i])``, returns every ordered pair of
    distinct members that share a common group element. This materialises the
    lower / upper adjacencies together with the shared face / co-face
    :math:`\\delta` required by the corresponding MPSN messages.

    Parameters
    ----------
    group : torch.Tensor
        Long tensor of shared-element indices (one per incidence).
    member : torch.Tensor
        Long tensor of member-simplex indices (one per incidence).

    Returns
    -------
    dst : torch.Tensor
        Receiver member indices, one per ordered pair.
    src : torch.Tensor
        Neighbour member indices, one per ordered pair.
    shared : torch.Tensor
        Shared group-element index, one per ordered pair.
    """
    empty = torch.empty(0, dtype=torch.long, device=member.device)
    if group.numel() == 0:
        return empty, empty.clone(), empty.clone()

    order = torch.argsort(group)
    group, member = group[order], member[order]
    uniq, counts = torch.unique_consecutive(group, return_counts=True)

    dst_all, src_all, shared_all = [], [], []
    start = 0
    for value, count in zip(uniq.tolist(), counts.tolist(), strict=True):
        if count >= 2:
            block = member[start : start + count]
            dst = block.repeat_interleave(count)
            src = block.repeat(count)
            keep = dst != src
            dst, src = dst[keep], src[keep]
            dst_all.append(dst)
            src_all.append(src)
            shared_all.append(torch.full_like(dst, value))
        start += count

    if not dst_all:
        return empty, empty.clone(), empty.clone()
    return torch.cat(dst_all), torch.cat(src_all), torch.cat(shared_all)


def _build_adjacency_index(
    incidence_1: torch.Tensor, incidence_2: torch.Tensor
) -> dict[str, tuple[torch.Tensor, ...]]:
    r"""Derive all MPSN adjacency indices from the incidence matrices.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Node-edge incidence matrix :math:`B_1` of shape ``[N0, N1]``.
    incidence_2 : torch.Tensor
        Edge-face incidence matrix :math:`B_2` of shape ``[N1, N2]``.

    Returns
    -------
    dict
        Mapping from adjacency name to index tensors. Boundary / co-boundary
        entries are ``(dst, src)`` pairs; lower / upper entries are
        ``(dst, src, shared)`` triples (see module docstring for the exact
        rank semantics).
    """
    b1 = _coo_indices(incidence_1)  # rows: nodes, cols: edges
    b2 = _coo_indices(incidence_2)  # rows: edges, cols: faces
    node, edge_n = b1[0], b1[1]
    edge_f, face = b2[0], b2[1]

    up0 = _ordered_pairs(edge_n, node)  # nodes sharing an edge (co-face)
    up1 = _ordered_pairs(face, edge_f)  # edges sharing a face (co-face)
    low1 = _ordered_pairs(node, edge_n)  # edges sharing a node (face)
    low2 = _ordered_pairs(edge_f, face)  # faces sharing an edge (face)

    return {
        # boundary: (receiver rank-k cell, its rank-(k-1) face)
        "bnd1": (edge_n, node),
        "bnd2": (face, edge_f),
        # co-boundary: (receiver rank-k cell, its rank-(k+1) co-face)
        "cob0": (node, edge_n),
        "cob1": (edge_f, face),
        # lower / upper: (receiver, neighbour, shared simplex)
        "up0": up0,
        "up1": up1,
        "low1": low1,
        "low2": low2,
    }


class MPSNLayer(nn.Module):
    r"""A single Message Passing Simplicial Network layer.

    Implements the per-rank messages and update of Eqs. (1)-(5) of Bodnar et
    al. (2021) for a complex of dimension up to two. Each existing adjacency
    for a rank owns a dedicated message MLP; absent adjacencies (e.g. vertices
    have no boundary, faces have no co-face in a 2-complex) are omitted.

    Parameters
    ----------
    hidden_channels : int
        Feature dimension shared by all ranks.
    activation : str, optional
        Intermediate activation used inside every MLP (default: ``"relu"``).
    use_boundary : bool, optional
        Whether boundary messages are used (default: True).
    use_coboundary : bool, optional
        Whether co-boundary messages are used (default: True).
    use_lower : bool, optional
        Whether lower-adjacency messages are used (default: True).
    use_upper : bool, optional
        Whether upper-adjacency messages are used (default: True).
    """

    def __init__(
        self,
        hidden_channels: int,
        activation: str = "relu",
        use_boundary: bool = True,
        use_coboundary: bool = True,
        use_lower: bool = True,
        use_upper: bool = True,
    ) -> None:
        super().__init__()
        h = hidden_channels
        self.hidden_channels = h
        self.use_boundary = use_boundary
        self.use_coboundary = use_coboundary
        self.use_lower = use_lower
        self.use_upper = use_upper

        # Rank 0 (nodes): co-boundary (edges) + upper (via edges).
        self.msg_0_cob = _mlp(
            2 * h, h, activation
        )  # Eq. (2): co-boundary msg M_C
        self.msg_0_up = _mlp(
            3 * h, h, activation
        )  # Eq. (4): upper msg M_N^ (shared co-face)
        self.upd_0 = _mlp(3 * h, h, activation)  # Eq. (5): rank-0 update U

        # Rank 1 (edges): boundary + co-boundary + lower + upper.
        self.msg_1_bnd = _mlp(
            2 * h, h, activation
        )  # Eq. (1): boundary msg M_B
        self.msg_1_cob = _mlp(
            2 * h, h, activation
        )  # Eq. (2): co-boundary msg M_C
        self.msg_1_low = _mlp(
            3 * h, h, activation
        )  # Eq. (3): lower msg M_Nv (shared face)
        self.msg_1_up = _mlp(
            3 * h, h, activation
        )  # Eq. (4): upper msg M_N^ (shared co-face)
        self.upd_1 = _mlp(5 * h, h, activation)  # Eq. (5): rank-1 update U

        # Rank 2 (faces): boundary (edges) + lower (via edges).
        self.msg_2_bnd = _mlp(
            2 * h, h, activation
        )  # Eq. (1): boundary msg M_B
        self.msg_2_low = _mlp(
            3 * h, h, activation
        )  # Eq. (3): lower msg M_Nv (shared face)
        self.upd_2 = _mlp(3 * h, h, activation)  # Eq. (5): rank-2 update U

    def _zeros(self, num: int, ref: torch.Tensor) -> torch.Tensor:
        """Allocate a zero message buffer matching ``ref`` device/dtype.

        Parameters
        ----------
        num : int
            Number of rows (simplices of the target rank).
        ref : torch.Tensor
            Reference tensor providing device and dtype.

        Returns
        -------
        torch.Tensor
            Zero tensor of shape ``[num, hidden_channels]``.
        """
        return torch.zeros(
            num, self.hidden_channels, device=ref.device, dtype=ref.dtype
        )

    def _pair_message(
        self,
        num: int,
        mlp: nn.Module,
        dst: torch.Tensor,
        h_dst: torch.Tensor,
        src: torch.Tensor,
        h_src: torch.Tensor,
        shared: torch.Tensor | None = None,
        h_shared: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        """Compute one aggregated message type via summed MLP messages.

        Parameters
        ----------
        num : int
            Number of receiver simplices.
        mlp : torch.nn.Module
            Message function :math:`M_\\bullet`.
        dst : torch.Tensor
            Receiver indices, one per message.
        h_dst : torch.Tensor
            Feature bank of the receiver rank.
        src : torch.Tensor
            Neighbour indices, one per message.
        h_src : torch.Tensor
            Feature bank of the neighbour rank.
        shared : torch.Tensor, optional
            Shared face / co-face indices for lower / upper messages.
        h_shared : torch.Tensor, optional
            Feature bank of the shared-simplex rank.
        enabled : bool, optional
            If False, returns zeros (adjacency ablated).

        Returns
        -------
        torch.Tensor
            Aggregated messages of shape ``[num, hidden_channels]``.
        """
        out = self._zeros(num, h_dst)
        if not enabled or dst.numel() == 0:
            return out
        parts = [h_dst[dst], h_src[src]]
        if shared is not None:
            parts.append(h_shared[shared])
        out.index_add_(0, dst, mlp(torch.cat(parts, dim=-1)))
        return out

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        index: dict[str, tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one round of simplicial message passing.

        Parameters
        ----------
        x : tuple of torch.Tensor
            Feature banks ``(x_0, x_1, x_2)`` for nodes, edges and faces.
        index : dict
            Adjacency indices as returned by ``_build_adjacency_index``.

        Returns
        -------
        tuple of torch.Tensor
            Updated ``(x_0, x_1, x_2)`` feature banks.
        """
        x0, x1, x2 = x
        n0, n1, n2 = x0.size(0), x1.size(0), x2.size(0)

        # --- Rank 0 (nodes) ---
        d, s = index["cob0"]
        m0_cob = self._pair_message(
            n0, self.msg_0_cob, d, x0, s, x1, enabled=self.use_coboundary
        )
        d, s, sh = index["up0"]
        m0_up = self._pair_message(
            n0, self.msg_0_up, d, x0, s, x0, sh, x1, enabled=self.use_upper
        )
        # Eq. (5): h_v = U(x_v, sum m_C, sum m_N^)
        h0 = self.upd_0(torch.cat([x0, m0_cob, m0_up], dim=-1))

        # --- Rank 1 (edges) ---
        d, s = index["bnd1"]
        m1_bnd = self._pair_message(
            n1, self.msg_1_bnd, d, x1, s, x0, enabled=self.use_boundary
        )
        d, s = index["cob1"]
        m1_cob = self._pair_message(
            n1, self.msg_1_cob, d, x1, s, x2, enabled=self.use_coboundary
        )
        d, s, sh = index["low1"]
        m1_low = self._pair_message(
            n1, self.msg_1_low, d, x1, s, x1, sh, x0, enabled=self.use_lower
        )
        d, s, sh = index["up1"]
        m1_up = self._pair_message(
            n1, self.msg_1_up, d, x1, s, x1, sh, x2, enabled=self.use_upper
        )
        # Eq. (5): h_e = U(x_e, sum m_B, sum m_C, sum m_Nv, sum m_N^)
        h1 = self.upd_1(torch.cat([x1, m1_bnd, m1_cob, m1_low, m1_up], dim=-1))

        # --- Rank 2 (faces) ---
        d, s = index["bnd2"]
        m2_bnd = self._pair_message(
            n2, self.msg_2_bnd, d, x2, s, x1, enabled=self.use_boundary
        )
        d, s, sh = index["low2"]
        m2_low = self._pair_message(
            n2, self.msg_2_low, d, x2, s, x2, sh, x1, enabled=self.use_lower
        )
        # Eq. (5): h_t = U(x_t, sum m_B, sum m_Nv)
        h2 = self.upd_2(torch.cat([x2, m2_bnd, m2_low], dim=-1))

        return h0, h1, h2


class MPSN(nn.Module):
    r"""Message Passing Simplicial Network backbone.

    Lifts per-rank input features to a common hidden dimension and applies a
    stack of :class:`MPSNLayer` blocks. Consumes only the incidence matrices
    :math:`B_1` and :math:`B_2`; the four adjacencies of Bodnar et al. (2021,
    Definition 4) are derived internally and shared across layers.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input feature dimensions on ``(nodes, edges, faces)``.
    hidden_channels : int
        Hidden (and output) feature dimension used on every rank.
    n_layers : int, optional
        Number of message-passing layers (default: 2).
    activation : str, optional
        Intermediate activation for all MLPs (default: ``"relu"``).
    use_boundary : bool, optional
        Enable boundary messages (default: True).
    use_coboundary : bool, optional
        Enable co-boundary messages (default: True).
    use_lower : bool, optional
        Enable lower-adjacency messages (default: True).
    use_upper : bool, optional
        Enable upper-adjacency messages (default: True).
    """

    def __init__(
        self,
        in_channels_all: tuple[int, int, int],
        hidden_channels: int,
        n_layers: int = 2,
        activation: str = "relu",
        use_boundary: bool = True,
        use_coboundary: bool = True,
        use_lower: bool = True,
        use_upper: bool = True,
    ) -> None:
        super().__init__()
        assert n_layers >= 1, "n_layers must be a positive integer."
        self.hidden_channels = hidden_channels
        self.in_linear_0 = nn.Linear(in_channels_all[0], hidden_channels)
        self.in_linear_1 = nn.Linear(in_channels_all[1], hidden_channels)
        self.in_linear_2 = nn.Linear(in_channels_all[2], hidden_channels)
        self.layers = nn.ModuleList(
            MPSNLayer(
                hidden_channels,
                activation=activation,
                use_boundary=use_boundary,
                use_coboundary=use_coboundary,
                use_lower=use_lower,
                use_upper=use_upper,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        x_all: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        incidence_all: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward computation.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Feature tensors ``(x_0, x_1, x_2)`` on nodes, edges and faces.
        incidence_all : tuple of torch.Tensor
            Incidence matrices ``(B_1, B_2)`` (node-edge and edge-face).

        Returns
        -------
        tuple of torch.Tensor
            Final hidden states ``(x_0, x_1, x_2)`` on nodes, edges and faces.
        """
        x0, x1, x2 = x_all
        incidence_1, incidence_2 = incidence_all
        index = _build_adjacency_index(incidence_1, incidence_2)

        x = (
            self.in_linear_0(x0),
            self.in_linear_1(x1),
            self.in_linear_2(x2),
        )
        for layer in self.layers:
            x = layer(x, index)
        return x
