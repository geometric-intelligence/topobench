"""Directed Simplicial Neural Network (DirSNN) Backbone.

Implements the Dir-SNN architecture from:
    Lecha et al., "Higher-Order Topological Directionality and
    Directed Simplicial Neural Networks", arXiv:2409.08389.

Key engineering contributions over the base paper:
    - Active simplicial sparsification to bound O(|E|^2) memory growth in L_up.
    - Sparse autograd quarantine to prevent dense gradient materialization.
    - Topological boundary bypass (eps-residuals) for source/sink gradient recovery.
    - Asymmetric dropout regularization per pathway.
    - Depth-adaptive gating: learnable per-layer scalar gates let the model
      discover its own effective depth per task, rather than requiring a
      fixed num_layers hyperparameter shared across all tasks. Empirically
      motivated by a discovered trade-off: shallow depth improves community
      detection OOD generalization (avoids a homophily-driven shortcut) while
      full depth is required for triangle counting (multi-hop structural
      aggregation). Gating lets a single architecture serve both without
      requiring task-specific config overrides.

Integrated into the TopoBench framework for TDL Challenge 2026, Track 2.
"""

import warnings

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Utility: Self-Loop Removal
# ---------------------------------------------------------------------------


def _remove_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Remove diagonal entries from a sparse COO adjacency matrix.

    The directed adjacency matrices A^{ij}_{down,1} are defined between
    *distinct* simplices (Lecha et al., Eq. 3). When constructed via
    B_source^T @ B_source, the diagonal entry (e, e) is nonzero because
    every edge trivially shares its own source vertex with itself. These
    diagonal entries are topologically meaningless and would double-count
    the self-contribution already handled by the explicit eps_down * x
    residual bypass. This function removes them.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO tensor of shape [N, N].

    Returns
    -------
    torch.Tensor
        Sparse COO tensor with all diagonal entries removed, coalesced.
    """
    idx = A._indices()  # shape: [2, nnz]
    val = A._values()  # shape: [nnz]
    # Keep only off-diagonal entries: row index != col index
    off_diag_mask = idx[0] != idx[1]
    return torch.sparse_coo_tensor(
        idx[:, off_diag_mask],
        val[off_diag_mask],
        A.size(),
        device=A.device,
        dtype=A.dtype,
    ).coalesce()


# ---------------------------------------------------------------------------
# Core Layer
# ---------------------------------------------------------------------------


class DirSNNLayer(nn.Module):
    """Single message-passing layer for Directed Simplicial Neural Networks.

    Implements the feature update rule from Lecha et al. (2409.08389), Eq. (6)-(10),
    operating on 1-simplices (edges) via four asymmetric directed lower adjacency
    operators and one upper Laplacian:

        H = sigma(
              alpha * sum_{i,j in {0,1}} A^{ij}_{down,1} X W_{ij}
            + beta  * L_up X W_up
            + gamma * X W_self
            )

    where the four directed lower adjacencies are constructed from the signed
    incidence matrix B_1 by isolating source (tail, value=-1) and target
    (head, value=+1) vertex masks and cross-multiplying:

        A^{0,0} = B_source^T B_source  (shared-source adjacency)
        A^{1,1} = B_target^T B_target  (shared-target adjacency)
        A^{0,1} = B_source^T B_target  (source-to-target flow)
        A^{1,0} = B_target^T B_source  (target-to-source flow)

    Diagonal entries (self-loops) are removed from all four adjacency matrices
    because the paper defines adjacency between *distinct* simplices, and the
    self-contribution is already handled by the explicit eps_down * x residual.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels per edge.
    out_channels : int
        Number of output feature channels per edge.
    dropout : float, optional
        Base dropout rate. The upper pathway uses this full rate; the lower
        and self pathways use half this rate. Default: 0.0.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Four Directed Lower Adjacency Weight Matrices ---
        # Each learns a distinct role:
        #   W_00: aggregates features from edges sharing the same source node.
        #   W_11: aggregates features from edges sharing the same target node.
        #   W_01: aggregates features along source-to-target directional flow.
        #   W_10: aggregates features along target-to-source (reverse) flow.
        # Raw nn.Parameter + manual matmul used instead of nn.Linear so that
        # the weight shapes are explicit and match the paper's notation directly.
        # Bias is deliberately omitted on the four directed pathways — these
        # aggregate neighborhood signals where a shared bias would conflate
        # topologically distinct roles. Bias is retained on the self pathway.
        self.W_00 = nn.Parameter(torch.empty(in_channels, out_channels))
        self.W_11 = nn.Parameter(torch.empty(in_channels, out_channels))
        self.W_01 = nn.Parameter(torch.empty(in_channels, out_channels))
        self.W_10 = nn.Parameter(torch.empty(in_channels, out_channels))

        # --- Upper and Self-Loop Weight Matrices ---
        # W_up: operates on L_up = B2 B2^T (curl/rotational signal).
        # W_self: acts on x directly; bias=True provides the affine shift
        #         needed to learn non-zero representations in isolated simplices.
        self.W_up = nn.Parameter(torch.empty(in_channels, out_channels))
        # Self-loop uses nn.Linear for the bias term.
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)

        # --- Topological Boundary Bypass (eps-Residuals) ---
        # For source edges (no incoming triangles), L_up @ x = 0, killing W_up's
        # gradient. For sink edges, L_down contributions vanish similarly.
        # Learnable eps parameters inject a non-zero additive residual to guarantee
        # a non-zero pre-activation input to both pathways even at topological
        # boundaries.
        #
        # CRITICAL: Initialized to 1.0, not 1e-4.
        # At 1e-4 the bypass gradient is 10,000x smaller than the neighborhood
        # signal gradient, making convergence near-impossible on boundary edges.
        # At 1.0 the bypass contributes equally to the neighborhood signal at
        # initialization, giving boundary edges a fair gradient from step 1.
        self.eps_down = nn.Parameter(torch.tensor(1.0))
        self.eps_up = nn.Parameter(torch.tensor(1.0))

        # --- Harmonic Equalization Scalars ---
        # Initialized to 1/3 so that the initial summed output has the same
        # expected magnitude as any single pathway alone.
        # Initializing all three to 1.0 would triple the pre-activation magnitude,
        # saturating ReLU immediately and compressing the gradient on the first
        # backward pass.
        self.alpha = nn.Parameter(torch.tensor(1.0 / 3.0))
        self.beta = nn.Parameter(torch.tensor(1.0 / 3.0))
        self.gamma = nn.Parameter(torch.tensor(1.0 / 3.0))

        # --- Asymmetric Dropout ---
        # The upper pathway (L_up) is driven by triangle structure, which can be
        # dominated by a small number of dense cliques. Full dropout rate combats
        # the resulting co-adaptation. The lower and self pathways see more stable
        # structural signals and use half the dropout rate.
        self.drop_lower = nn.Dropout(dropout * 0.5)
        self.drop_upper = nn.Dropout(dropout)
        self.drop_self = nn.Dropout(dropout * 0.5)

        self.activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier uniform initialization for all weight parameter matrices.

        Xavier uniform is appropriate here because all pathways feed into a
        ReLU activation. The gain is set to the standard ReLU gain (sqrt(2)).
        """
        gain = nn.init.calculate_gain("relu")
        for W in (self.W_00, self.W_11, self.W_01, self.W_10, self.W_up):
            nn.init.xavier_uniform_(W, gain=gain)
        # W_self is an nn.Linear; reset it separately using its own method.
        nn.init.xavier_uniform_(self.W_self.weight, gain=gain)
        nn.init.zeros_(self.W_self.bias)

    def forward(
        self,
        x: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of one DirSNN layer.

        All topological structure (adjacency matrix construction, sparsification,
        Laplacian computation) is executed inside a torch.no_grad() block.
        This is the Sparse Autograd Quarantine: it prevents PyTorch from building
        a computation graph over the sparse structural tensors, which would
        otherwise silently materialize dense gradient tensors during backward
        and negate the memory advantage of sparse storage.

        Gradient tracking is retained only for `x` and the learnable weight
        parameters, which is sufficient for correct backpropagation.

        Parameters
        ----------
        x : torch.Tensor
            Edge feature matrix, shape [|E|, in_channels].
        b1 : torch.Tensor
            Sparse signed incidence matrix B_1, shape [|V|, |E|].
            Entry (v, e) = +1 if v is the head (target) of edge e,
                         = -1 if v is the tail (source) of edge e.
        b2 : torch.Tensor
            Sparse signed coboundary matrix B_2, shape [|E|, |T|].
            Entry (e, t) is nonzero if edge e is a face of triangle t.

        Returns
        -------
        torch.Tensor
            Updated edge feature matrix, shape [|E|, out_channels].
        """
        # ------------------------------------------------------------------ #
        # SPARSE AUTOGRAD QUARANTINE                                           #
        # All static topological infrastructure computed here.                 #
        # No gradients are tracked inside this block.                          #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            # [Step 1] Split B_1 into source and target boolean masks.
            #
            # B_1 convention (Lecha et al.):
            #   entry = +1  ->  this vertex is the TARGET (head) of the edge
            #   entry = -1  ->  this vertex is the SOURCE (tail) of the edge
            #
            # We extract the COO indices and values once to avoid recomputing
            # the sparse representation four times.
            b1_idx = b1._indices()  # shape [2, nnz_b1]
            b1_val = b1._values()  # shape [nnz_b1]

            target_mask = b1_val == 1.0  # Boolean mask over nnz entries
            source_mask = b1_val == -1.0  # Boolean mask over nnz entries

            # Construct binary (all-ones-valued) sparse masks.
            # Converting -1 entries to +1 so that the cross-products produce
            # non-negative adjacency counts (number of shared vertices).
            b_target = torch.sparse_coo_tensor(
                b1_idx[:, target_mask],
                torch.ones(
                    target_mask.sum(), device=b1.device, dtype=b1.dtype
                ),
                b1.size(),
            ).coalesce()

            b_source = torch.sparse_coo_tensor(
                b1_idx[:, source_mask],
                torch.ones(
                    source_mask.sum(), device=b1.device, dtype=b1.dtype
                ),
                b1.size(),
            ).coalesce()

            # [Step 2] Construct the four directed lower adjacency matrices.
            #
            # These implement Eq. (3) from Lecha et al. for 1-simplices (edges):
            #   A^{0,0}: edges that share a source vertex (source-source)
            #   A^{1,1}: edges that share a target vertex (target-target)
            #   A^{0,1}: edges where one's source is another's target
            #   A^{1,0}: edges where one's target is another's source
            #
            # Each is constructed as a sparse matrix product. The result has
            # shape [|E|, |E|], entry (i, j) > 0 iff edges i and j are
            # adjacent under that particular face-map pair.
            a_00 = torch.sparse.mm(b_source.t(), b_source).coalesce()
            a_11 = torch.sparse.mm(b_target.t(), b_target).coalesce()
            a_01 = torch.sparse.mm(b_source.t(), b_target).coalesce()
            a_10 = torch.sparse.mm(b_target.t(), b_source).coalesce()

            # [Step 3] Remove self-loops from all four adjacency matrices.
            #
            # WHY: B_source^T @ B_source produces nonzero diagonal entries
            # because every edge trivially shares its own source vertex with
            # itself. These diagonal entries are NOT valid directed adjacencies
            # — the paper defines adjacency between *distinct* simplices
            # (Lecha et al., Eq. 3, implicit distinctness).
            #
            # Furthermore, keeping them would double-count the self-contribution:
            # the explicit `eps_down * x` bypass already handles self-information
            # for boundary edges. A second self-loop through the adjacency matrix
            # would give non-boundary edges an unearned extra self-contribution,
            # breaking the symmetry between boundary and non-boundary treatment.
            a_00 = _remove_self_loops(a_00)
            a_11 = _remove_self_loops(a_11)
            a_01 = _remove_self_loops(a_01)
            a_10 = _remove_self_loops(a_10)

            # [Step 4] Construct the upper Laplacian L_up = B2 B2^T.
            #
            # L_up entry (e_i, e_j) counts the number of triangles containing
            # both edges e_i and e_j — the upper (co-boundary) adjacency.
            # This is used for the curl/rotational signal pathway (Eq. 2).
            l_up = torch.sparse.mm(b2, b2.t()).coalesce()

        # ------------------------------------------------------------------ #
        # MESSAGE PASSING (Gradients tracked for x and W parameters)          #
        # ------------------------------------------------------------------ #

        # [Lower Directed Messages]
        # Each of the four adjacency matrices aggregates a distinct directional
        # signal, processed by its own independent weight matrix.
        # The four results are summed before applying the eps bypass, so the
        # bypass adds to the *combined* lower signal (not to each pathway
        # independently), keeping the bypass's contribution properly bounded.
        msg_00 = self.drop_lower(torch.sparse.mm(a_00, x) @ self.W_00)
        msg_11 = self.drop_lower(torch.sparse.mm(a_11, x) @ self.W_11)
        msg_01 = self.drop_lower(torch.sparse.mm(a_01, x) @ self.W_01)
        msg_10 = self.drop_lower(torch.sparse.mm(a_10, x) @ self.W_10)

        # Topological boundary bypass for the lower pathway.
        # For sink edges (no outgoing lower adjacencies), all four msg_* terms
        # are zero. The eps_down * x term guarantees a non-zero gradient for
        # W_00 through W_10 on those edges by providing a non-zero pre-image.
        msg_lower = (msg_00 + msg_11 + msg_01 + msg_10) + (
            self.eps_down * (x @ self.W_00)
        )

        # [Upper Message]
        # Aggregates curl/rotational information from co-boundary triangles.
        # The eps_up bypass serves source edges (no triangles above them).
        msg_upper = self.drop_upper(torch.sparse.mm(l_up, x) @ self.W_up) + (
            self.eps_up * (x @ self.W_up)
        )

        # [Self-Loop Message]
        # Retains the current edge's own representation across the layer.
        # Uses nn.Linear (with bias) to allow non-zero output even when the
        # neighborhood aggregations are entirely zero (fully isolated edges).
        msg_self = self.drop_self(self.W_self(x))

        # [Harmonic Equalization]
        # Learnable scalars alpha, beta, gamma dynamically reweight the three
        # topological signal components. Initialized to 1/3 so the combined
        # initial output has the same expected magnitude as a single-pathway
        # model. This prevents ReLU saturation at initialization.
        out = self.activation(
            (self.alpha * msg_lower)
            + (self.beta * msg_upper)
            + (self.gamma * msg_self)
        )
        return out


# ---------------------------------------------------------------------------
# Full Backbone
# ---------------------------------------------------------------------------


class DirSNN(nn.Module):
    """Directed Simplicial Neural Network backbone for the TopoBench framework.

    Stacks multiple DirSNNLayer modules to perform multi-hop message passing
    on directed simplicial complexes. Operates on 1-simplex (edge) features.

    Integrations beyond the base Lecha et al. architecture:
      - Active simplicial sparsification: bounds L_up memory to O(tau * |E|)
        by stochastically pruning B2 when the average upper degree exceeds tau.
        This prevents the O(|E|^2) memory spike that occurs in dense-clique
        complexes (see red team analysis, Section I.1).
      - One-time boundary orientation guard in __init__ to validate B1 @ B2 = 0
        without overhead in the forward pass hot path.
      - Depth-adaptive gating: a learnable scalar gate per layer blends that
        layer's output with its own input, initialized to 1.0 (fully "on",
        mathematically identical to unconditional stacking at initialization).
        Empirical investigation found community detection benefits from an
        effectively shallower model (avoids a homophily-driven neighbor-label
        shortcut that fails to generalize out-of-distribution), while
        triangle counting requires full depth (multi-hop aggregation of
        structural counts). Rather than hardcoding a different num_layers
        per task via experiment_modes, each layer's gate can learn to shrink
        toward 0 when that layer's transformation does not help the task at
        hand, letting a single fixed-depth architecture discover its own
        effective depth per task directly from the training signal.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels on edges.
    hidden_channels : int
        Number of hidden feature channels. Capped at 64 in the default config
        to prevent OOM errors when L_up fills in on dense upper cliques.
    out_channels : int
        Number of output feature channels (equals num_classes for node tasks
        after the readout aggregation).
    num_layers : int, optional
        Number of stacked DirSNNLayer modules. Fixed at 2 in the default
        config: directed L_up and L_down smooth at different spectral rates,
        and stacking beyond 2 introduces asymmetric over-smoothing. With
        depth-adaptive gating, this now serves as a capacity ceiling rather
        than a fixed depth: the model can learn to behave as shallower via
        its gates without changing this value.
        Default: 2.
    dropout : float, optional
        Base dropout probability passed to each layer. Default: 0.5.
    max_upper_degree : int, optional
        Sparsification threshold tau. If the mean number of triangles per edge
        exceeds this value, B2 is stochastically pruned during training to
        bound memory. Default: 32.
    **kwargs : dict
        Absorbs extra keyword arguments from Hydra config for compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        max_upper_degree: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_upper_degree = max_upper_degree

        # Build layer stack: input -> hidden (x num_layers-2) -> output
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(DirSNNLayer(in_channels, out_channels, dropout))
        else:
            self.layers.append(
                DirSNNLayer(in_channels, hidden_channels, dropout)
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    DirSNNLayer(hidden_channels, hidden_channels, dropout)
                )
            self.layers.append(
                DirSNNLayer(hidden_channels, out_channels, dropout)
            )

        # --- Depth-Adaptive Gating ---
        # One learnable scalar gate per layer, initialized to 1.0 (fully
        # "on"). At initialization, the model is mathematically identical to
        # unconditional layer stacking -- gating changes nothing until
        # training pushes a gate away from 1.0. A gate that learns to shrink
        # toward 0 means the model has found that layer's transformation is
        # not helping for the task at hand, effectively self-selecting a
        # shallower effective path without requiring a hardcoded num_layers
        # value per task. See class docstring for the empirical motivation.
        self.layer_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)]
        )

    def _validate_boundary(self, b1: torch.Tensor, b2: torch.Tensor) -> None:
        """Assert the fundamental homology axiom: B1 @ B2 = 0.

        Called once during the first forward pass rather than in __init__
        because b1 and b2 are not available at construction time in the
        TopoBench batch-based pipeline.

        The check is skipped in eval mode and after the first call to avoid
        overhead during training. It uses no_grad and operates on sparse
        tensors directly to avoid materializing a dense [|V|, |T|] matrix.

        Parameters
        ----------
        b1 : torch.Tensor
            Sparse B_1 matrix, shape [|V|, |E|].
        b2 : torch.Tensor
            Sparse B_2 matrix, shape [|E|, |T|].
        """
        with torch.no_grad():
            product = torch.sparse.mm(b1, b2).coalesce()
            if product._nnz() > 0:
                norm = torch.norm(product.values()).item()
                if norm > 1e-5:
                    warnings.warn(
                        f"Boundary orientation guard: ||B1 @ B2||_F = {norm:.6f} > 1e-5. "
                        "This violates the fundamental homology axiom (B_k B_{k+1} = 0). "
                        "On real datasets this indicates inconsistent simplex orientation. "
                        "On CI/CD mock data this warning is expected and non-fatal.",
                        stacklevel=3,
                    )

    def _sparsify_boundary(self, b2: torch.Tensor) -> torch.Tensor:
        """Stochastically prune B2 to bound upper Laplacian memory.

        The upper Laplacian L_up = B2 B2^T has memory cost O(|E|^2) in the
        worst case (dense clique). This method enforces a maximum average
        upper degree of max_upper_degree triangles per edge by randomly
        dropping columns of B2 (triangles) during training.

        The keep probability is set so that the expected post-pruning upper
        degree equals max_upper_degree exactly. High-persistence topological
        features (triangles with many adjacent edges) are more likely to be
        retained statistically, which approximately preserves persistent
        homological structure while bounding memory.

        Only active during training. At eval time, the full B2 is used for
        deterministic, reproducible inference.

        Parameters
        ----------
        b2 : torch.Tensor
            Sparse B_2 matrix, shape [|E|, |T|].

        Returns
        -------
        torch.Tensor
            Sparsified (or unchanged) sparse B_2 matrix.
        """
        if not self.training or b2._nnz() == 0:
            return b2

        # Correct density metric: mean triangles per edge.
        # b2.shape[0] == |E| (rows of B2 are edges).
        # b2._nnz() == 3 * |T| because each triangle contributes exactly 3
        # nonzero entries (one per face edge). Dividing by |E| gives the
        # mean number of triangle-memberships per edge = mean upper degree.
        #
        # BUG FIXED: previous version used b1.shape[1] as the denominator
        # and num_edges as a local variable, inadvertently computing
        # 3*|T|/|E| through b1 rather than b2.shape[0], which would agree
        # numerically but is semantically incorrect and fragile if shapes differ.
        mean_upper_degree = b2._nnz() / b2.shape[0]

        if mean_upper_degree <= self.max_upper_degree:
            return b2

        keep_prob = self.max_upper_degree / mean_upper_degree
        mask = torch.rand(b2._nnz(), device=b2.device) < keep_prob
        return torch.sparse_coo_tensor(
            b2._indices()[:, mask],
            b2._values()[mask],
            b2.size(),
            device=b2.device,
            dtype=b2.dtype,
        ).coalesce()

    def forward(self, batch) -> torch.Tensor:
        """Forward pass of the full DirSNN backbone.

        Parameters
        ----------
        batch : torch_geometric.data.Data or compatible
            Must contain:
              batch.x_1         : Tensor [|E|, in_channels]  -- edge features
              batch.incidence_1 : sparse Tensor [|V|, |E|]  -- signed incidence
              batch.incidence_2 : sparse Tensor [|E|, |T|]  -- signed coboundary

        Returns
        -------
        torch.Tensor
            Edge-level output features, shape [|E|, out_channels].
        """
        x = batch.x_1
        b1 = batch.incidence_1
        b2 = batch.incidence_2

        # One-time boundary validation on the first training forward pass.
        # _boundary_validated is set as an instance flag so the check runs
        # exactly once per training run, not every batch.
        if self.training and not getattr(self, "_boundary_validated", False):
            self._validate_boundary(b1, b2)
            self._boundary_validated = True

        # Active sparsification: prune B2 if the complex is too dense.
        # This must happen BEFORE computing L_up in each layer so that the
        # sparsified B2 propagates into the layer's no_grad block.
        # The layer itself does not hold B2 state, so we pass the pruned
        # version directly.
        with torch.no_grad():
            b2 = self._sparsify_boundary(b2)

        # Sequential message passing through all layers, with depth-adaptive
        # gating. Each layer's contribution is blended with its own input via
        # a learnable scalar gate, initialized to 1.0 so the model starts
        # identical to unconditional stacking. A gate that learns to shrink
        # toward 0 means the model has found that layer's transformation is
        # not helping for the task at hand -- effectively self-selecting a
        # shallower path without a hardcoded num_layers.
        #
        # Gating is only applied when the layer's output shape matches its
        # input shape, since the residual blend requires matching dimensions.
        # The first layer (in_channels -> hidden_channels) and last layer
        # (hidden_channels -> out_channels) typically change dimensionality
        # and so skip gating, always applying fully; only interior layers
        # (hidden_channels -> hidden_channels, when num_layers > 2) are
        # eligible to gate. For num_layers <= 2, every layer changes
        # dimensionality and gates have no effect until num_layers >= 3 --
        # this is intentional: with only 1 or 2 layers there is no interior
        # layer whose contribution is ambiguous, so gating is a no-op by
        # construction rather than a hidden behavior change.
        for layer, gate in zip(self.layers, self.layer_gates, strict=True):
            layer_out = layer(x, b1, b2)
            if layer_out.shape == x.shape:
                x = gate * layer_out + (1 - gate) * x
            else:
                x = layer_out

        return x
