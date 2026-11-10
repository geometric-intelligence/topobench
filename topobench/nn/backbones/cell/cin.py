"""Cell Isomorphism Network backbone for cell complexes.

Implements the CIN model from:

    Bodnar, C., Frasca, F., Otter, N., Wang, Y.G., Liò, P., Montúfar, G.,
    & Bronstein, M. (2021).
    *Weisfeiler and Lehman Go Cellular: CW Networks.*
    NeurIPS 2021. https://arxiv.org/abs/2106.12575
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CINLayer(nn.Module):
    """Single layer of the Cell Isomorphism Network (CIN).

    Implements the message types from Section 4, Bodnar et al. (2021),
    applied simultaneously to 0-cells (nodes), 1-cells (edges), and 2-cells
    (rings). Per Theorem 7, *lower*-adjacency (N↓) messages can be dropped
    without loss of expressive power and are therefore omitted here.

    Coboundary messages are **not** dropped: this layer follows the
    ``use_coboundaries=True`` CIN configuration from the reference
    implementation, which is the configuration
    used to obtain the paper's reported SOTA results. Concretely, the up-message
    ``m_↑(σ) = AGG_{τ∈N↑(σ), δ∈C(σ,τ)} M_↑(h_σ, h_τ, h_δ)`` (Section 4) is
    approximated as a *sum of two separately-linear* terms - one over the
    upper-adjacent neighbours τ (via the A_up,p matrix) and one over the
    shared coboundary cells δ (via the B_{p+1} incidence matrix) - which is
    exactly recovered when ``M_↑`` is itself linear in its (h_τ, h_δ)
    arguments (as ``Linear(concat(h_τ, h_δ)) = W_τ h_τ + W_δ h_δ + b``).
    This keeps the coboundary-cell context described in Section 4 while
    reusing TopoBench's precomputed sparse adjacency/incidence matrices
    (which do not carry per-pair (τ, δ) indices).

    Per-dimension message schedule
    --------------------------------
    **0-cells (nodes):**  B(node) = ∅ → no boundary message.
    Upper-adjacency + coboundary context (Section 4 Eq. for m_↑, p=0):

    ``m_↑(node_i) ≈ AGG_{j ∈ N↑(i)} msg_↑_neighbor(h_j)``
                  ``+ AGG_{e ∈ C(i)} msg_↑_coboundary(h_e)``

    Implemented as: ``A_up,0 @ msg_up_0(x_0)  +  B1 @ msg_bridge_e(x_1)``
    where ``B1 [N0, N1]`` rows index nodes and cols index incident edges.

    **1-cells (edges):**  B(edge) = {endpoint nodes} → boundary from nodes.
    Plus upper-adjacency + coboundary context (Section 4 Eq. for m_↑, p=1):

    ``m_B(edge_e)  =  B1^T @ msg_B(x_0)``         (Section 4, Figure 6)
    ``m_↑(edge_e) ≈  A_up,1 @ msg_↑_neighbor(x_1)``
                  ``+  B2 @ msg_↑_coboundary(x_2)``

    Implemented as: ``B1^T @ msg_boundary_1(x_0)``
                    ``+ A_up,1 @ msg_up_1(x_1)``
                    ``+ B2 @ msg_bridge_r(x_2)``

    **2-cells (rings):**  no N↑ (no 3-cells exist).
    Boundary from edges only:

    ``m_B(ring_r)  =  B2^T @ msg_B(x_1)``          (Section 4, Figure 6)

    Two-step update (GIN-style, Appendix E.3)
    ------------------------------------------
    Following Appendix E.3, each incoming multi-set gets
    its own injective GIN-style update, and the two results are combined by
    a third MLP so the overall update stays injective ("the cross product
    of countable spaces is countable"):

    ``out_↑(σ) = MLP_↑,p( (1+ε₁)·h^t_σ + m_↑(σ) )``
    ``out_B(σ) = MLP_B,p( (1+ε₂)·h^t_σ + m_B(σ) )``
    ``h^{t+1}_p(σ) = MLP_combine,p( out_↑(σ) ⊥ out_B(σ) )``

    where ``⊥`` is concatenation. Ranks with an empty neighbourhood keep
    the corresponding stream with a zero message (as in the reference,
    where ``propagate`` returns zeros for absent index sets): 0-cells have
    ``m_B = 0`` (B(node) = ∅) and 2-cells have ``m_↑ = 0`` (no 3-cells).

    Parameters
    ----------
    in_channels_0 : int
        Input feature dimension for 0-cells (nodes).
    in_channels_1 : int
        Input feature dimension for 1-cells (edges).
    in_channels_2 : int
        Input feature dimension for 2-cells (rings). 0 = no 2-cells.
    out_channels : int
        Output feature dimension for all cell dimensions.
    eps : float, optional
        Initial value of the learnable GIN epsilon. Default: 0.0.
    """

    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
        out_channels: int,
        eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.has_2_cells = in_channels_2 > 0

        # ── 0-cell messages: m_↑(node) decomposed ──────────────────────── #
        # Component 1: neighbor node features via A_up,0
        #   m_up_0(i) = A_up,0 @ msg_up_0(x_0)
        self.msg_up_0 = nn.Linear(in_channels_0, out_channels, bias=False)
        # Component 2: bridge edge features via B1 (B1[v,e]=1 if v∈∂e)
        #   m_bridge_e(i) = B1 @ msg_bridge_e(x_1)   [N0,N1]@[N1,d]=[N0,d]
        # Equation: m_↑(node_i) ≈ sum_j msg_up(h_j) + sum_{e sharing j,i} msg_bridge(h_e)
        self.msg_bridge_e = nn.Linear(in_channels_1, out_channels, bias=False)

        # ── 1-cell messages: m_B(edge) + m_↑(edge) decomposed ─────────── #
        # Boundary: nodes on the boundary of each edge (paper Section 4)
        #   m_B(edge_e) = B1^T @ msg_boundary_1(x_0)  [N1,N0]@[N0,d]=[N1,d]
        # This is the "atoms→bonds" boundary stream.
        self.msg_boundary_1 = nn.Linear(
            in_channels_0, out_channels, bias=False
        )
        # Upper-adj component 1: neighbor edge features via A_up,1
        self.msg_up_1 = nn.Linear(in_channels_1, out_channels, bias=False)
        if self.has_2_cells:
            # Upper-adj component 2: bridge ring features via B2
            #   m_bridge_r(e) = B2 @ msg_bridge_r(x_2)  [N1,N2]@[N2,d]=[N1,d]
            # Equation: m_↑(edge_e) ≈ sum_f msg_up(h_f) + sum_{r sharing e,f} msg_bridge(h_r)
            self.msg_bridge_r = nn.Linear(
                in_channels_2, out_channels, bias=False
            )
            # ── 2-cell messages: m_B(ring) ──────────────────────────────── #
            # Boundary: edges on boundary of each ring
            #   m_B(ring_r) = B2^T @ msg_B_2(x_1)  [N2,N1]@[N1,d]=[N2,d]
            self.msg_B_2 = nn.Linear(in_channels_1, out_channels, bias=False)

        # ── GIN epsilon parameters (one per stream per cell dimension,
        #    mirroring eps1/eps2 in SparseCINCochainConv) ─────────────────── #
        self.eps_up_0 = nn.Parameter(torch.tensor(eps))
        self.eps_B_0 = nn.Parameter(torch.tensor(eps))
        self.eps_up_1 = nn.Parameter(torch.tensor(eps))
        self.eps_B_1 = nn.Parameter(torch.tensor(eps))
        if self.has_2_cells:
            self.eps_up_2 = nn.Parameter(torch.tensor(eps))
            self.eps_B_2 = nn.Parameter(torch.tensor(eps))

        # ── Self-feature projections (maps h^t to out_channels) ─────────── #
        self.self_lin_0 = nn.Linear(in_channels_0, out_channels, bias=False)
        self.self_lin_1 = nn.Linear(in_channels_1, out_channels, bias=False)
        if self.has_2_cells:
            self.self_lin_2 = nn.Linear(
                in_channels_2, out_channels, bias=False
            )

        # ── Two-step update: per-stream 2-layer GIN MLPs followed by a #
        #    combine MLP on their concatenation ─────────────────────── #
        self.update_up_0 = self._make_update_mlp(out_channels)
        self.update_B_0 = self._make_update_mlp(out_channels)
        self.combine_0 = self._make_combine_mlp(out_channels)
        self.update_up_1 = self._make_update_mlp(out_channels)
        self.update_B_1 = self._make_update_mlp(out_channels)
        self.combine_1 = self._make_combine_mlp(out_channels)
        if self.has_2_cells:
            self.update_up_2 = self._make_update_mlp(out_channels)
            self.update_B_2 = self._make_update_mlp(out_channels)
            self.combine_2 = self._make_combine_mlp(out_channels)

    @staticmethod
    def _make_update_mlp(channels: int) -> nn.Sequential:
        """Build a per-stream GIN update MLP.

        Mirrors ``update_up_nn`` / ``update_boundaries_nn`` in the reference
        implementation: two Linear layers, each followed by BatchNorm and ReLU.

        Parameters
        ----------
        channels : int
            Input and output feature dimension of the MLP.

        Returns
        -------
        nn.Sequential
            The 2-layer update MLP.
        """
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    @staticmethod
    def _make_combine_mlp(channels: int) -> nn.Sequential:
        """Build the combine MLP of the two-step update.

        a single Linear layer on the concatenation of the two stream outputs, followed by BatchNorm
        and ReLU.

        Parameters
        ----------
        channels : int
            Feature dimension of each input stream; the MLP maps
            ``2 * channels -> channels``.

        Returns
        -------
        nn.Sequential
            The combine MLP.
        """
        return nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(
        self,
        x_0: Tensor,
        x_1: Tensor,
        x_2: Tensor | None,
        adjacency_0: Tensor,
        adjacency_1: Tensor | None,
        incidence_1: Tensor,
        incidence_2: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_0 : Tensor, shape [N0, d0]
            Features of 0-cells (nodes).
        x_1 : Tensor, shape [N1, d1]
            Features of 1-cells (edges).
        x_2 : Tensor or None, shape [N2, d2]
            Features of 2-cells (rings).
        adjacency_0 : Tensor, shape [N0, N0] sparse
            Upper-adjacency matrix for 0-cells (A_up,0). A_up,0[i,j]=1
            iff nodes i,j share a common edge (co-boundary).
        adjacency_1 : Tensor or None, shape [N1, N1] sparse
            Upper-adjacency matrix for 1-cells (A_up,1). A_up,1[e,f]=1
            iff edges e,f share a common ring (co-boundary).
        incidence_1 : Tensor, shape [N0, N1] sparse
            Boundary matrix B1.  B1[v, e] = 1 iff vertex v ∈ boundary(edge e).
            Used as-is for bridge-edge aggregation to nodes (B1 @ x_1),
            and transposed for boundary aggregation to edges (B1^T @ x_0).
        incidence_2 : Tensor or None, shape [N1, N2] sparse
            Boundary matrix B2.  B2[e, r] = 1 iff edge e ∈ boundary(ring r).
            Used as-is for bridge-ring aggregation to edges (B2 @ x_2),
            and transposed for boundary aggregation to rings (B2^T @ x_1).

        Returns
        -------
        Tensor
            Updated 0-cell features, shape [N0, out_channels].
        Tensor
            Updated 1-cell features, shape [N1, out_channels].
        Tensor or None
            Updated 2-cell features, shape [N2, out_channels].

        Notes
        -----
        Equation references are to Bodnar et al. (2021), Section 4.
        Lower-adjacency (N↓) is dropped per Theorem 7. Boundary (B) and
        upper-adjacency (N↑) messages are used, and coboundary-cell (C)
        context is folded into the upper-adjacency message as a linear
        approximation of the ``use_coboundaries=True`` configuration (see
        class docstring) — the setting used for the paper's reported
        results, not the theoretical minimum from Theorem 7.
        """
        N0, N1 = x_0.size(0), x_1.size(0)
        dev = x_0.device

        # ── 0-cell update: m_↑(node) = msg_up(neighbors) + msg_bridge(edges) #
        # B(node) = ∅ → no boundary message for 0-cells.
        # Upper-adj neighbor: A_up,0 @ msg_up_0(x_0)
        m_up_0 = self._upper_aggregate(
            x_neighbor=x_0,
            adjacency=adjacency_0,
            msg_fn=self.msg_up_0,
            n_receivers=N0,
        )  # [N0, out_channels]
        # Upper-adj bridge edge: B1 @ msg_bridge_e(x_1)  (B1 not transposed)
        m_bridge_e = self._incidence_aggregate(
            x_sender=x_1,
            incidence=incidence_1,
            msg_fn=self.msg_bridge_e,
            transpose=True,  # use B1 as-is: [N0,N1] @ msg(x_1) → [N0,d]
            n_receivers=N0,
        )  # [N0, out_channels]
        # Two-step update (App. E.3; SparseCINCochainConv.forward): each
        # multi-set gets its own (1+eps)-weighted GIN update, then the two
        # stream outputs are combined. B(node) = ∅, so the boundary stream
        # receives a zero message (as in the reference's propagate).
        h_0 = self.self_lin_0(x_0)
        out_up_0 = self.update_up_0(
            (1 + self.eps_up_0) * h_0 + m_up_0 + m_bridge_e
        )
        out_B_0 = self.update_B_0((1 + self.eps_B_0) * h_0)
        x_0_new = self.combine_0(torch.cat([out_up_0, out_B_0], dim=-1))

        # ── 1-cell update: m_B(edge) + m_↑(edge) ───────────────────────── #
        # Boundary (atoms→bonds, Section 4): B1^T @ msg_boundary_1(x_0)
        m_B_1 = self._incidence_aggregate(
            x_sender=x_0,
            incidence=incidence_1,
            msg_fn=self.msg_boundary_1,
            transpose=False,  # use B1^T: [N1,N0] @ msg(x_0) → [N1,d]
            n_receivers=N1,
        )  # [N1, out_channels]
        # Upper-adj neighbor: A_up,1 @ msg_up_1(x_1)
        m_up_1 = torch.zeros(N1, self.out_channels, device=dev)
        if adjacency_1 is not None:
            m_up_1 = self._upper_aggregate(
                x_neighbor=x_1,
                adjacency=adjacency_1,
                msg_fn=self.msg_up_1,
                n_receivers=N1,
            )  # [N1, out_channels]
        # Upper-adj bridge ring: B2 @ msg_bridge_r(x_2)  (B2 not transposed)
        m_bridge_r = torch.zeros(N1, self.out_channels, device=dev)
        if self.has_2_cells and x_2 is not None and incidence_2 is not None:
            m_bridge_r = self._incidence_aggregate(
                x_sender=x_2,
                incidence=incidence_2,
                msg_fn=self.msg_bridge_r,
                transpose=True,  # use B2 as-is: [N1,N2] @ msg(x_2) → [N1,d]
                n_receivers=N1,
            )  # [N1, out_channels]
        # Two-step update for 1-cells: separate GIN updates for the upper
        # multi-set (neighbour edges + bridge rings) and the boundary
        # multi-set (endpoint nodes), then combine.
        h_1 = self.self_lin_1(x_1)
        out_up_1 = self.update_up_1(
            (1 + self.eps_up_1) * h_1 + m_up_1 + m_bridge_r
        )
        out_B_1 = self.update_B_1((1 + self.eps_B_1) * h_1 + m_B_1)
        x_1_new = self.combine_1(torch.cat([out_up_1, out_B_1], dim=-1))

        # ── 2-cell update: m_B(ring) from edges ────────────────────────── #
        x_2_new = None
        if self.has_2_cells and x_2 is not None:
            N2 = x_2.size(0)
            # Boundary (bonds→rings, Section 4): B2^T @ msg_B_2(x_1)
            m_B_2 = torch.zeros(N2, self.out_channels, device=dev)
            if incidence_2 is not None:
                m_B_2 = self._incidence_aggregate(
                    x_sender=x_1,
                    incidence=incidence_2,
                    msg_fn=self.msg_B_2,
                    transpose=False,  # use B2^T: [N2,N1] @ msg(x_1) → [N2,d]
                    n_receivers=N2,
                )
            # Two-step update for 2-cells: no 3-cells exist, so the upper
            # stream receives a zero message (as in the reference's
            # propagate); the boundary stream aggregates constituent edges.
            h_2 = self.self_lin_2(x_2)
            out_up_2 = self.update_up_2((1 + self.eps_up_2) * h_2)
            out_B_2 = self.update_B_2((1 + self.eps_B_2) * h_2 + m_B_2)
            x_2_new = self.combine_2(torch.cat([out_up_2, out_B_2], dim=-1))

        return x_0_new, x_1_new, x_2_new

    # ── Private helpers ─────────────────────────────────────────────────── #

    def _incidence_aggregate(
        self,
        x_sender: Tensor,
        incidence: Tensor,
        msg_fn: nn.Linear,
        transpose: bool,
        n_receivers: int,
    ) -> Tensor:
        """Aggregate messages via an incidence matrix.

        Parameters
        ----------
        x_sender : Tensor, shape [N_sender, d]
            Features of the sending cells.
        incidence : Tensor, sparse or dense
            Incidence matrix in its natural storage shape.
            When ``transpose=True`` the caller passes shape
            [N_receivers, N_senders] and the matrix is used as-is.
            When ``transpose=False`` the caller passes shape
            [N_senders, N_receivers] (e.g. B1[N0,N1] for boundary→edges)
            and the matrix is transposed before multiplication.
        msg_fn : nn.Linear
            Projects sender features to ``out_channels`` before aggregation.
        transpose : bool
            If True use incidence as-is: shape [N_receivers, N_senders].
            If False transpose it first: caller passes [N_senders, N_receivers].
        n_receivers : int
            Expected number of receiving cells; used only for the output shape.

        Returns
        -------
        Tensor
            Aggregated messages, shape [n_receivers, out_channels].
        """
        msgs = msg_fn(x_sender)  # [N_sender, out_channels]
        if incidence.is_sparse:
            mat = incidence if transpose else incidence.t()
            return torch.sparse.mm(mat, msgs)
        else:
            mat = incidence if transpose else incidence.t()
            return mat @ msgs

    def _upper_aggregate(
        self,
        x_neighbor: Tensor,
        adjacency: Tensor | None,
        msg_fn: nn.Linear,
        n_receivers: int,
    ) -> Tensor:
        """Aggregate upper-adjacency messages via a sparse adjacency matrix.

        Computes ``m_up = adjacency @ msg_fn(x_neighbor)``.

        Parameters
        ----------
        x_neighbor : Tensor, shape [N, d]
            Features of the (same-dimension) neighbor cells.
        adjacency : Tensor or None, shape [N, N] sparse
            Upper-adjacency matrix. Returns zeros if None or empty.
        msg_fn : nn.Linear
            Projects neighbor features to ``out_channels``.
        n_receivers : int
            Number of receiving cells (= N).

        Returns
        -------
        Tensor
            Aggregated messages, shape [n_receivers, out_channels].
        """
        dev = x_neighbor.device
        if adjacency is None:
            return torch.zeros(n_receivers, self.out_channels, device=dev)

        msgs = msg_fn(x_neighbor)  # [N, out_channels]

        if adjacency.is_sparse:
            if adjacency._nnz() == 0:
                return torch.zeros(n_receivers, self.out_channels, device=dev)
            if not adjacency.is_coalesced():
                adjacency = adjacency.coalesce()
            return torch.sparse.mm(adjacency, msgs)
        else:
            return adjacency @ msgs


class CIN(nn.Module):
    """Cell Isomorphism Network (CIN).

    Stacks ``num_layers`` of :class:`CINLayer` to perform hierarchical
    message passing over a regular cell complex with 0-, 1-, and 2-cells.

    Follows the CIN instantiation in Section 5 of
    Bodnar et al. (2021): GIN-style injective aggregators ensure the model
    is as powerful as the Cellular WL (CWL) test (Theorem 18).

    Message streams per layer (all three ranks updated simultaneously):

    * 0-cells receive: upper-adj from neighboring nodes + bridge edge features.
    * 1-cells receive: boundary from endpoint nodes (Section 4, Figure 6)
      + upper-adj from neighboring edges + bridge ring features.
    * 2-cells receive: boundary from constituent edges (Section 4, Figure 6).

    Parameters
    ----------
    in_channels_0 : int
        Input feature size for 0-cells (nodes).
    in_channels_1 : int
        Input feature size for 1-cells (edges).
    in_channels_2 : int
        Input feature size for 2-cells (rings). 0 = no 2-cells.
    hid_channels : int
        Hidden feature dimension for all intermediate layers.
    n_layers : int
        Number of CIN layers.
    dropout : float, optional
        Dropout rate applied after each layer. Default: 0.0.
    """

    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
        hid_channels: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.has_2_cells = in_channels_2 > 0

        # Initial feature projections (one per cell dimension)
        self.proj_0 = nn.Linear(in_channels_0, hid_channels)
        self.proj_1 = nn.Linear(in_channels_1, hid_channels)
        if self.has_2_cells:
            self.proj_2 = nn.Linear(in_channels_2, hid_channels)

        # Stack of CIN layers
        self.layers = nn.ModuleList(
            [
                CINLayer(
                    in_channels_0=hid_channels,
                    in_channels_1=hid_channels,
                    in_channels_2=hid_channels if self.has_2_cells else 0,
                    out_channels=hid_channels,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x_0: Tensor,
        x_1: Tensor,
        x_2: Tensor | None,
        adjacency_0: Tensor,
        adjacency_1: Tensor | None,
        incidence_1: Tensor,
        incidence_2: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Forward pass of CIN.

        Parameters
        ----------
        x_0 : Tensor, shape [N0, d0]
            Features of 0-cells (nodes).
        x_1 : Tensor, shape [N1, d1]
            Features of 1-cells (edges).
        x_2 : Tensor or None, shape [N2, d2]
            Features of 2-cells (rings).
        adjacency_0 : Tensor, shape [N0, N0] sparse
            Upper-adjacency for 0-cells (A_up,0).
        adjacency_1 : Tensor or None, shape [N1, N1] sparse
            Upper-adjacency for 1-cells (A_up,1).
        incidence_1 : Tensor, shape [N0, N1] sparse
            Boundary matrix B1.
        incidence_2 : Tensor or None, shape [N1, N2] sparse
            Boundary matrix B2.

        Returns
        -------
        Tensor
            Updated 0-cell features, shape [N0, hid_channels].
        Tensor
            Updated 1-cell features, shape [N1, hid_channels].
        Tensor or None
            Updated 2-cell features, shape [N2, hid_channels].
        """
        x_0 = F.elu(self.proj_0(x_0))
        x_1 = F.elu(self.proj_1(x_1))
        if self.has_2_cells and x_2 is not None:
            x_2 = F.elu(self.proj_2(x_2))

        for layer in self.layers:
            x_0, x_1, x_2 = layer(
                x_0=x_0,
                x_1=x_1,
                x_2=x_2,
                adjacency_0=adjacency_0,
                adjacency_1=adjacency_1,
                incidence_1=incidence_1,
                incidence_2=incidence_2,
            )
            x_0 = F.dropout(x_0, p=self.dropout, training=self.training)
            x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
            if x_2 is not None:
                x_2 = F.dropout(x_2, p=self.dropout, training=self.training)

        return x_0, x_1, x_2
