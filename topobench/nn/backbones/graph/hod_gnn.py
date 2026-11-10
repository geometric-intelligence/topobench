r"""HOD-GNN: High-Order Derivative GNN (first public implementation).

This backbone implements **1-HOD-GNN** from "On the Expressive Power of GNN
Derivatives" (Eitan, Eliasof, Gelberg, Frasca, Bar-Shalom, Maron; ICLR 2026,
[1]_) — specifically the empirical architecture the authors use in *all* of
their experiments (their Appendix F): a deep-and-narrow base GIN whose
**first-order derivatives with respect to its own input features** are
computed analytically, encoded by a pointwise MLP, and passed together with
the base representations to a downstream GIN. To our knowledge no public
implementation of this paper exists; this file is written from the paper
alone.

**Why derivatives?** Message passing is bounded by 1-WL and provably cannot
count triangles ([2]_). The paper's motivating example: for the (linear) GNN
:math:`\mathcal{M}(\bm{A}, \bm{X}) = \bm{A}^3\bm{X}`, the derivative of node
:math:`v`'s output with respect to its own input is exactly
:math:`\bm{A}^3_{v,v}` — the number of closed 3-walks at :math:`v`, i.e.
(six times) the triangles at :math:`v`. Derivatives of a GNN are therefore
*structure-aware* quantities that restore precisely the substructure
information 1-WL message passing is blind to, and Theorem 4.1 of [1]_ places
the resulting models between Subgraph GNNs and the WL hierarchy.

**Relation to random-walk encodings.** Theorem 4.2 of [1]_ is constructive:
with degree-normalised aggregation, the diagonal derivative
:math:`\partial \bm{h}^{(t)}_v / \partial \bm{X}_v` of a suitably initialised
base MPNN equals the :math:`t`-step random-walk return probability
:math:`(\tilde{\bm{A}}^t)_{v,v}` — i.e. the RWSE of GraphGPS ([3]_). HOD-GNN
is thus a *learnable, strictly more expressive generalisation* of a
hand-crafted RWSE: it starts (under ``structural_init``) at the same
random-walk fingerprint and is free to learn task-adapted structural
features end-to-end. The ``structural_init`` scheme below follows Appendix F
of [1]_ (constant-one input embedding, identity MLPs, GIN
:math:`\epsilon = -1`), under which the derivative diagonal at
initialisation equals the walk-return / centrality encoding of [4]_.

**What is computed** (all inside the backbone, from ``edge_index`` alone):

1. The input is projected to a narrow base width :math:`k`
   (``base_channels``); these projected features :math:`\bm{z}` play the
   role of the paper's initial features :math:`\bm{X}`.
2. A base GIN runs for ``base_layers`` layers. Alongside each layer's node
   update, the exact Jacobian
   :math:`\bm{J}^{(t)}[v, u] = \partial \bm{h}^{(t)}_v / \partial \bm{z}_u`
   is propagated analytically with the *same* sparse aggregation used for
   the features (Appendix D of [1]_; for first-order derivatives Faà di
   Bruno's formula reduces to the chain rule). The propagation is fully
   differentiable, so training updates the base GIN *through* the
   derivative computation — the paper's key algorithmic contribution.
3. Following Eq. (73) of [1]_, the per-layer representations and the
   per-layer **diagonal** Jacobians :math:`\bm{J}^{(t)}[v, v]` are
   concatenated across layers with :math:`1/t!` scaling. (The diagonal at
   *every* depth is kept, matching the RWSE correspondence, which needs all
   walk lengths.) The stacked diagonals are encoded by a pointwise MLP
   (:math:`\text{U}^{\text{node}}`, Eq. (74) of [1]_).
4. A downstream GIN consumes ``[base features ⊕ encoded derivatives ⊕
   optionally the raw input]`` and produces the output node embeddings.
   Output-level derivatives :math:`\bm{\mathsf{D}}^{\text{out}}` and
   :math:`k \geq 2` mixed derivatives are deliberately **not** implemented:
   the paper's own experiments omit them (Appendix F).

**Batch-awareness.** Batches contain disjoint graphs. The Jacobian source
axis is indexed *locally per graph* (padded to the largest graph in the
batch), and all propagation uses ``edge_index``, which never crosses graph
boundaries — so a graph's output, including every derivative channel, is
identical whether it is processed alone or inside a batch.

**Cost.** The Jacobian tensor has shape ``[N, S, k, k]`` for ``N`` total
nodes, largest-graph size ``S`` and base width ``k`` — the reason the paper
(and this implementation) keeps the base GIN narrow (:math:`k \le 16`) while
allowing it to be deep. Sparsity in the source axis (nodes further than
:math:`t` hops have zero derivative) is not exploited; for the graph sizes
targeted here (:math:`\le` a few hundred nodes) the dense-padded form is
faster and simpler.

References
----------
.. [1] Eitan, Eliasof, Gelberg, Frasca, Bar-Shalom, Maron. "On the
   Expressive Power of GNN Derivatives." ICLR 2026.
   https://arxiv.org/abs/2510.02565
.. [2] Chen, Chen, Villar, Bruna. "Can Graph Neural Networks Count
   Substructures?" NeurIPS 2020. https://arxiv.org/abs/2002.04025
.. [3] Rampášek et al. "Recipe for a General, Powerful, Scalable Graph
   Transformer" (GraphGPS / RWSE). NeurIPS 2022.
   https://arxiv.org/abs/2205.12454
.. [4] Southern, Eitan, Bar-Shalom, Bronstein, Maron, Frasca. "Balancing
   Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality"
   (HyMN). ICLR 2025. https://openreview.net/forum?id=2hbgKYuao1
"""

import math

import torch
import torch.nn.functional as F


class HODGNN(torch.nn.Module):
    r"""First-order 1-HOD-GNN backbone (Appendix F of the paper).

    A narrow base GIN is run for ``base_layers`` layers while its exact
    input-Jacobian is propagated analytically alongside the features. The
    :math:`1/t!`-scaled per-layer features and diagonal Jacobians are
    concatenated, the diagonals are encoded by a pointwise MLP, and a
    downstream GIN maps ``[features ⊕ encoded derivatives ⊕ input]`` to the
    output node embeddings. See the module docstring for the full method
    description and references.

    Faithfulness notes (for the correctness criterion): this is the
    architecture of Appendix F of the paper — first-order derivatives only,
    output derivatives omitted, diagonal-only encoding, factorial residual —
    which is the configuration used for every experimental result reported
    there. ``aggregation="sum"`` gives the paper's GIN aggregation;
    ``aggregation="mean"`` gives the row-normalised (random-walk) operator
    used in the constructive proof of the paper's Theorem 4.2, whose
    derivative diagonals are degree-normalised and hence scale-stable across
    graph-density regimes. Both are instances of the linear aggregations the
    derivative algorithm supports (Eq. (28) of the paper).

    Parameters
    ----------
    in_channels : int
        Dimension of the input node features.
    hidden_channels : int
        Hidden width of the downstream GIN.
    out_channels : int
        Dimension of the returned node embeddings.
    base_channels : int, optional
        Width :math:`k` of the base GIN. Kept deliberately narrow: the
        Jacobian tensor scales with :math:`k^2`. Default is 8.
    base_layers : int, optional
        Depth :math:`T` of the base GIN; the derivative encoding contains
        walk information up to length :math:`T`. Default is 6.
    downstream_layers : int, optional
        Number of downstream GIN layers. Default is 3.
    encoder_hidden : int, optional
        Hidden width of the derivative-encoder MLP. Default is 64.
    encoder_channels : int, optional
        Output width of the derivative-encoder MLP. Default is 32.
    activation : str, optional
        Activation of base and downstream MLPs, ``"relu"`` or ``"silu"``.
        The paper uses ReLU by default; the analytic SiLU improves
        substructure counting (its Appendix G.1). Default is ``"relu"``.
    aggregation : str, optional
        Neighbour aggregation, ``"sum"`` (GIN, the paper's default) or
        ``"mean"`` (row-normalised random walk, the Theorem 4.2 operator).
        With ``structural_init``, ``"sum"`` yields walk-count-scale features
        that can destabilise training on dense graphs under hot learning
        rates, whereas ``"mean"`` keeps them in ``[0, 1]``.
        Default is ``"sum"``.
    structural_init : bool, optional
        Apply the paper's Appendix F initialisation (constant-one input
        embedding, identity base MLPs, :math:`\epsilon = -1`), under which
        the derivative diagonals at initialisation equal walk-return /
        centrality encodings. Default is True.
    include_input : bool, optional
        Concatenate the raw input features to the downstream GIN's input
        (the feature-preservation component of Eq. (58) of the paper; the
        narrow base width would otherwise bottleneck feature information).
        Default is True.
    base_dropout : float, optional
        Dropout inside the base GIN. Applied consistently to features and
        Jacobians (a dropout mask is a linear map, so its exact derivative
        is the same mask). Default is 0.0.
    dropout : float, optional
        Dropout between downstream GIN layers. Default is 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        base_channels: int = 8,
        base_layers: int = 6,
        downstream_layers: int = 3,
        encoder_hidden: int = 64,
        encoder_channels: int = 32,
        activation: str = "relu",
        aggregation: str = "sum",
        structural_init: bool = True,
        include_input: bool = True,
        base_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation not in ("relu", "silu"):
            raise ValueError(
                f"activation must be 'relu' or 'silu', got {activation!r}"
            )
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.base_layers = base_layers
        self.downstream_layers = downstream_layers
        self.activation = activation
        self.aggregation = aggregation
        self.structural_init = structural_init
        self.include_input = include_input
        self.base_dropout = base_dropout
        self.dropout = dropout

        k = base_channels
        self.lin_in = torch.nn.Linear(in_channels, k)

        # Base GIN: per layer a learnable eps and a two-layer MLP (an
        # activation after each linear layer, matching the recursive
        # "sigma(Wx+b)" decomposition of Appendix D, Eq. (30)).
        self.base_eps = torch.nn.Parameter(torch.zeros(base_layers))
        self.base_lin1 = torch.nn.ModuleList(
            torch.nn.Linear(k, k) for _ in range(base_layers)
        )
        self.base_lin2 = torch.nn.ModuleList(
            torch.nn.Linear(k, k) for _ in range(base_layers)
        )

        # Pointwise derivative encoder U^node (Eq. (74)) over the stacked
        # per-layer diagonal Jacobians.
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(base_layers * k * k, encoder_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_hidden, encoder_channels),
        )

        down_in = (
            base_layers * k
            + encoder_channels
            + (in_channels if include_input else 0)
        )
        self.down_eps = torch.nn.Parameter(torch.zeros(downstream_layers))
        self.down_lin1 = torch.nn.ModuleList()
        self.down_lin2 = torch.nn.ModuleList()
        width = down_in
        for _ in range(downstream_layers):
            self.down_lin1.append(torch.nn.Linear(width, hidden_channels))
            self.down_lin2.append(
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            width = hidden_channels
        self.lin_out = torch.nn.Linear(width, out_channels)

        # 1/t! scaling of the per-layer residual (Eq. (73)).
        self._scales = [
            1.0 / math.factorial(t) for t in range(1, base_layers + 1)
        ]

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reset all learnable parameters.

        Applies the paper's Appendix F initialisation to the base GIN when
        ``structural_init`` is set: the input projection outputs the
        constant-one vector, the base MLPs are identities and the GIN
        :math:`\epsilon` is :math:`-1`, so that the derivative diagonals at
        initialisation equal walk-return / centrality encodings.

        Returns
        -------
        None
            This method updates the module in place.
        """
        self.lin_in.reset_parameters()
        for lin in (*self.base_lin1, *self.base_lin2):
            lin.reset_parameters()
        if self.structural_init:
            torch.nn.init.zeros_(self.lin_in.weight)
            torch.nn.init.ones_(self.lin_in.bias)
            for lin in (*self.base_lin1, *self.base_lin2):
                torch.nn.init.eye_(lin.weight)
                torch.nn.init.zeros_(lin.bias)
            torch.nn.init.constant_(self.base_eps, -1.0)
        else:
            torch.nn.init.zeros_(self.base_eps)
        for layer in self.encoder:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
        for lin in (*self.down_lin1, *self.down_lin2):
            lin.reset_parameters()
        torch.nn.init.zeros_(self.down_eps)
        self.lin_out.reset_parameters()

    def _act(self, pre: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation.

        Parameters
        ----------
        pre : torch.Tensor
            Pre-activation values.

        Returns
        -------
        torch.Tensor
            Activated values, same shape as ``pre``.
        """
        if self.activation == "relu":
            return F.relu(pre)
        return F.silu(pre)

    def _act_deriv(self, pre: torch.Tensor) -> torch.Tensor:
        r"""First derivative of the configured activation.

        For ReLU this is the (almost-everywhere) step function; for SiLU,
        :math:`\sigma(x) + x\,\sigma(x)(1 - \sigma(x))` with
        :math:`\sigma` the logistic sigmoid. Both are expressed with
        differentiable ops so that training can backpropagate through the
        derivative computation itself.

        Parameters
        ----------
        pre : torch.Tensor
            Pre-activation values.

        Returns
        -------
        torch.Tensor
            Elementwise derivative values, same shape as ``pre``.
        """
        if self.activation == "relu":
            return (pre > 0).to(pre.dtype)
        sig = torch.sigmoid(pre)
        return sig * (1 + pre * (1 - sig))

    def _adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        r"""Build the sparse aggregation operator.

        Row :math:`v` of the returned matrix holds the coefficients
        :math:`b_{u,v}` of the linear neighbour aggregation
        :math:`\sum_{u \in \mathcal{N}(v)} b_{u,v}\,\bm{h}_u` (Eq. (28) of
        the paper): edge weights for ``aggregation="sum"``, degree-normalised
        weights for ``aggregation="mean"``. The same operator aggregates
        features and Jacobians, which is what makes the derivative
        computation exact.

        Parameters
        ----------
        edge_index : torch.Tensor
            Graph connectivity of shape ``[2, num_edges]`` (messages flow
            from row 0 to row 1).
        num_nodes : int
            Total number of nodes in the batch.
        edge_weight : torch.Tensor, optional
            Optional edge weights of shape ``[num_edges]``.
        dtype : torch.dtype
            Floating dtype of the values.

        Returns
        -------
        torch.Tensor
            Sparse COO tensor of shape ``[num_nodes, num_nodes]``.
        """
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            values = torch.ones(
                src.numel(), dtype=dtype, device=edge_index.device
            )
        else:
            values = edge_weight.to(dtype)
        if self.aggregation == "mean":
            deg = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
            deg.index_add_(0, dst, values)
            # Guard only true zero degrees so fractional weighted degrees
            # still yield an exact mean.
            values = values / deg.masked_fill(deg == 0, 1)[dst]
        return torch.sparse_coo_tensor(
            torch.stack([dst, src]),
            values,
            (num_nodes, num_nodes),
        ).coalesce()

    def _propagate_base(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        idx_local: torch.Tensor,
        max_graph_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Run the base GIN and its analytic Jacobian propagation.

        Implements Algorithm 1 of the paper for first-order derivatives:
        the Jacobian ``jac[v, s, i, j]`` :math:`= \partial h^{(t)}_{v,i} /
        \partial z_{u(s),j}` (``s`` indexes source nodes *locally within
        each graph*) is updated with the same aggregation as the features,
        mapped through each linear layer, and scaled by the activation
        derivative at the pre-activations (the first-order case of Faà di
        Bruno's formula). All operations are differentiable, so gradients
        flow through the derivative computation into the base weights.

        Parameters
        ----------
        z : torch.Tensor
            Projected input features of shape ``[num_nodes, k]``.
        adj : torch.Tensor
            Sparse aggregation operator from :meth:`_adjacency`.
        idx_local : torch.Tensor
            Within-graph node index of each node, shape ``[num_nodes]``.
        max_graph_size : int
            Largest graph size in the batch (padded source-axis length).

        Returns
        -------
        tuple of torch.Tensor
            ``(features, diagonals)`` where ``features`` has shape
            ``[num_nodes, base_layers * k]`` and stacks the
            :math:`1/t!`-scaled per-layer representations (Eq. (73)), and
            ``diagonals`` has shape ``[num_nodes, base_layers * k * k]``
            and stacks the equally scaled per-layer diagonal Jacobians.
        """
        n, k = z.shape
        s_dim = max_graph_size
        arange_n = torch.arange(n, device=z.device)

        h = z
        # jac[v, s, i, j] = d h[v, i] / d z[u(s), j], initialised to the
        # identity: each node depends only on itself, coordinate-wise.
        jac = z.new_zeros(n, s_dim, k, k)
        jac[arange_n, idx_local] = torch.eye(k, device=z.device, dtype=z.dtype)

        feats, diags = [], []
        for t in range(self.base_layers):
            # Aggregation step (Eq. (10)): identical linear operator for
            # features and Jacobian.
            eps = self.base_eps[t]
            h_agg = torch.sparse.mm(adj, h)
            jac_agg = torch.sparse.mm(adj, jac.reshape(n, -1)).reshape(
                n, s_dim, k, k
            )
            h = (1 + eps) * h + h_agg
            jac = (1 + eps) * jac + jac_agg

            # DeepSets step: two sigma(Wx+b) layers, each propagating the
            # Jacobian by W (Eq. (32)) then by the chain rule.
            for lin in (self.base_lin1[t], self.base_lin2[t]):
                pre = lin(h)
                jac = torch.einsum("nsij,oi->nsoj", jac, lin.weight)
                jac = jac * self._act_deriv(pre)[:, None, :, None]
                h = self._act(pre)

            if self.base_dropout > 0 and self.training:
                # A dropout mask is a linear map; applying the same mask to
                # the Jacobian is its exact derivative.
                mask = (torch.rand_like(h) >= self.base_dropout).to(
                    h.dtype
                ) / (1 - self.base_dropout)
                h = h * mask
                jac = jac * mask[:, None, :, None]

            scale = self._scales[t]
            feats.append(h * scale)
            diags.append(jac[arange_n, idx_local].reshape(n, k * k) * scale)

        return torch.cat(feats, dim=1), torch.cat(diags, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute HOD-GNN node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Batch assignment of shape ``[num_nodes]``. Keeps the Jacobian
            source axis within graph boundaries. If ``None``, all nodes are
            treated as one graph.
        edge_weight : torch.Tensor, optional
            Optional edge weights of shape ``[num_edges]``, used in the
            aggregation operator.
        **kwargs : dict
            Additional unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        """
        n = x.size(0)
        device = x.device
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=device)

        # Within-graph local index of every node and the padded source-axis
        # length (largest graph in the batch). Assumes nodes of each graph
        # are contiguous in the batch vector, as PyG's Batch guarantees.
        counts = torch.bincount(batch)
        offsets = torch.cumsum(counts, dim=0) - counts
        idx_local = torch.arange(n, device=device) - offsets[batch]
        max_graph_size = int(counts.max().item()) if n > 0 else 0

        adj = self._adjacency(edge_index, n, edge_weight, x.dtype)
        z = self.lin_in(x)
        feats, diags = self._propagate_base(z, adj, idx_local, max_graph_size)

        h_der = [feats, self.encoder(diags)]
        if self.include_input:
            h_der.append(x)
        h = torch.cat(h_der, dim=1)

        for t in range(self.downstream_layers):
            h = (1 + self.down_eps[t]) * h + torch.sparse.mm(adj, h)
            h = self._act(self.down_lin1[t](h))
            h = self._act(self.down_lin2[t](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.lin_out(h)
