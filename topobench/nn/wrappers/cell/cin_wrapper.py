"""Wrapper for the CIN (Cell Isomorphism Network) backbone.

This wrapper is a new version of
``topobench.nn.wrappers.cell.cwn_wrapper.CWNWrapper`` (not a modification of
it), because the CIN backbone has different argument semantics from
the existing (TopoModelX-sourced) CWN backbone:

Difference 1 — ``adjacency_0`` must be genuine node-level upper-adjacency
    ``CWNWrapper`` passes ``batch.adjacency_1`` (edge-edge upper-adjacency)
    into a backbone parameter that TopoModelX's ``CWN.forward`` happens to
    *name* ``adjacency_0`` — but that backbone only ever updates 1-cells
    (edges); ``x_0`` and ``x_2`` are projected once and never touched again
    by its message passing (see ``topomodelx.nn.cell.cwn.CWN.forward``,
    which loops layers but only reassigns ``x_1``). So for TopoModelX's CWN,
    that argument is correctly edge-edge adjacency despite the misleading
    name, and no node-level messages are computed at all.

    CIN, by contrast, genuinely updates all three ranks every layer
    (Section 4), so it needs the *real* A_up,0 (node-node upper-adjacency,
    TopoBench's ``batch.adjacency_0``) for node messages, in addition to
    A_up,1 (``batch.adjacency_1``) for edge messages. Both are passed here
    under their own names.

Difference 2 — transposition is handled inside the backbone
    ``CWNWrapper`` calls ``batch.incidence_1.T`` before passing it on,
    coupling the wrapper to a backbone-specific orientation convention.
    ``CINWrapper`` passes ``batch.incidence_1`` unmodified; ``CIN`` applies
    ``B1`` or ``B1^T`` internally per message type via
    ``_incidence_aggregate(transpose=...)``.
"""

from topobench.nn.wrappers.base import AbstractWrapper


class CINWrapper(AbstractWrapper):
    r"""Wrapper for the CIN (Cell Isomorphism Network) backbone.

    The CIN model updates embeddings of cells of rank 0 (nodes), 1 (edges),
    and 2 (rings) simultaneously at each layer. This wrapper unpacks the
    required tensors from the batch and feeds them to the backbone with
    the correct argument names, then repacks the outputs into the
    dictionary format expected by downstream readout modules.

    Notes
    -----
    The batch is expected to provide the following attributes:
    ``x_0``, ``x_1`` : Tensor, shape [N0, d0] / [N1, d1]
        Features of 0-cells (nodes) and 1-cells (edges).
    ``x_2`` : Tensor, shape [N2, d2], optional
        Features of 2-cells (rings); absent for complexes with no 2-cells.
    ``adjacency_0`` : Tensor, shape [N0, N0] sparse
        Upper-adjacency matrix for 0-cells (A_up,0).
    ``adjacency_1`` : Tensor, shape [N1, N1] sparse, optional
        Upper-adjacency matrix for 1-cells (A_up,1).
    ``incidence_1`` : Tensor, shape [N0, N1] sparse
        B1 incidence matrix.
    ``incidence_2`` : Tensor, shape [N1, N2] sparse, optional
        B2 incidence matrix.
    ``y`` : Tensor
        Labels.
    ``batch_0`` : Tensor
        Batch assignment for 0-cells.
    """

    def forward(self, batch):
        r"""Forward pass for the CIN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched cell-complex data.

        Returns
        -------
        dict
            Dictionary with keys: ``labels``, ``batch_0``,
            ``x_0``, ``x_1``, ``x_2``.
        """
        # Optional higher-order tensors: absent for complexes with no 2-cells.
        # 2-cells (rings) arise from a skeleton-preserving lifting map f: G → X
        # that attaches 2-cells to induced cycles — Definition 8 and Definition 11.
        x_2 = getattr(batch, "x_2", None)

        # A_up,1 [N1, N1]: upper-adjacency matrix for 1-cells (edges).
        # A_up,1[e,f] = 1 iff edges e and f share a common 2-cell (ring)
        # as their co-boundary — Definition 4(4), N↑(σ) for 1-cells.
        # Absent when no 2-cells exist.
        adjacency_1 = getattr(batch, "adjacency_1", None)

        # B2 [N1, N2]: incidence (boundary) matrix for 2-cells.
        # B2[e, r] = 1 iff edge e ∈ B(ring r) — Definition 3 (boundary relation)
        # and Definition 33 (unsigned boundary matrix B_k with k=2).
        # Used for: m_B(1-cell) ← 2-cells  and  m_B(2-cell) ← 1-cells
        # (both directions of the B2 boundary relation; see Section 4 Eq. 1).
        incidence_2 = getattr(batch, "incidence_2", None)

        # Run the full three-stream hierarchical message passing (Section 4).
        # Each argument below feeds a specific adjacency used by Eqs. (1)–(3):
        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            # x_2: initial 2-cell (ring) features h^0_σ for σ ∈ X^(2).
            # Higher-dim cells are populated as described in Appendix E.3.
            x_2=x_2,
            # A_up,0 [N0, N0]: upper-adjacency for 0-cells (nodes).
            # A_up,0[i, j] = 1 iff nodes i and j share a common edge as
            # their co-boundary — Definition 4(4), N↑(σ) for 0-cells.
            # Feeds m_↑^{t+1}(node i) = AGG_{j ∈ N↑(i)} M_↑(h_i, h_j, h_e)
            # in Eq. (1), Section 4.
            adjacency_0=batch.adjacency_0,
            # A_up,1 [N1, N1]: upper-adjacency for 1-cells (edges).
            # Feeds m_↑^{t+1}(edge e) = AGG_{f ∈ N↑(e)} M_↑(h_e, h_f, h_r)
            # in Eq. (1), Section 4, where r is the shared ring (co-boundary).
            # Per Theorem 7, lower-adjacency (N↓) messages can be dropped
            # without loss of expressive power; this stream is omitted here.
            # Boundary (B) and upper-adjacency (N↑, incl. coboundary context
            # per the backbone's "use_coboundaries" formulation) are kept.
            adjacency_1=adjacency_1,
            # B1 [N0, N1]: incidence (boundary) matrix for 1-cells.
            # B1[v, e] = 1 iff vertex v ∈ B(edge e) — Definition 3 and
            # Definition 33 (unsigned boundary matrix B_k with k=1).
            # Used as B1 [N0, N1] for m_↑(node) coboundary context ← edges,
            # and as B1^T [N1, N0] for m_B(edge) ← nodes, the atoms→bonds
            # boundary stream (Section 4, Figure 6, orange arrows).
            # (No .T here; the backbone applies the correct orientation
            #  internally via _incidence_aggregate(transpose=True/False).)
            incidence_1=batch.incidence_1,
            # B2 [N1, N2]: incidence matrix for 2-cells (see above).
            # Used as B2 [N1, N2] for m_↑(edge) coboundary context ← rings,
            # and as B2^T [N2, N1] for m_B(ring) ← edges, the bonds→rings
            # boundary stream (Section 4, Figure 6).
            incidence_2=incidence_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
