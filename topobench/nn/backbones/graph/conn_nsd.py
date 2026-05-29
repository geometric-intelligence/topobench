"""Sheaf Neural Network with Connection Laplacians (Conn-NSD).

Backbone implementing Barbero, Bodnar, Sáez de Ocáriz Borde, Bronstein,
Veličković, Liò. "Sheaf Neural Networks with Connection Laplacians."
ICML 2022 TAG-ML Workshop. arXiv:2206.08702.

Notes
-----
**Scientific delta vs. Bodnar et al.'s NSD-O(d).**
NSD-O(d) learns the orthogonal restriction maps ``F_{vu}`` jointly with
the diffusion weights, recomputing them every layer from
``Φ(x_v, x_u)``. Conn-NSD instead *computes them once* at the start of
the forward pass — deterministically — from a local-PCA-plus-alignment
procedure on the raw node features (see :mod:`conn_nsd_utils.connection`,
Algorithm 1 of the paper). The diffusion equation (paper §2.3, Eq. 5)
is otherwise identical::

    X_{t+1}  =  X_t  −  σ( Δ_F (I_n ⊗ W₁) X_t W₂ )

This removes one network of learnable parameters (the sheaf-learner MLPs
of NSD-O(d)) and one backward pass through an SVD/Cayley parametrisation
per layer. Empirically it acts as a regulariser on heterophilic
node-classification tasks (paper Table 1) and yields a noticeable
wall-clock speedup (paper Table 2).

**Architectural notes.**
The class deliberately mirrors :class:`NSDEncoder` so a reader switching
between Bodnar's and Barbero's models is comparing exactly the parts
that differ: the construction of ``restriction_maps`` and the absence
of ``sheaf_learners``. We reuse the linear-channel left/right
transformations ``W₁``, ``W₂`` and the diffusion residual structure
``x_0 ← (1 + tanh(ε)) x_0 − x`` verbatim from the Bodnar code path;
both are part of the *diffusion* layer, not the *sheaf*.

References
----------
.. [1] F. Barbero, C. Bodnar, H. Sáez de Ocáriz Borde, M. Bronstein,
       P. Veličković, P. Liò. "Sheaf Neural Networks with Connection
       Laplacians." ICML 2022 TAG-ML Workshop. arXiv:2206.08702.
.. [2] C. Bodnar et al. "Neural Sheaf Diffusion." ICLR 2022 Workshop.
       arXiv:2202.04579.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor, nn
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.conn_nsd_utils import build_connection
from topobench.nn.backbones.graph.conn_nsd_utils.fixed_laplacian_builder import (
    FixedConnectionLaplacianBuilder,
)


class ConnNSDEncoder(nn.Module):
    """Inductive Conn-NSD encoder for graph-level and node-level tasks.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of the hidden state per node, *across* all stalk
        components. Must be divisible by ``stalk_dim``.
    num_layers : int, default 2
        Number of sheaf-diffusion layers ``T`` in paper Eq. 5.
    stalk_dim : int, default 3
        Stalk dimension ``d`` (paper §3.2). Setting ``d = 1`` recovers a
        classical normalised-Laplacian GCN; the model is only meaningful
        for ``d ≥ 2``.
    dropout : float, default 0.0
        Dropout on the diffusion state between layers.
    input_dropout : float, default 0.0
        Dropout on the initial feature lift.
    connection_features : str, default ``"raw"``
        Which features to feed Algorithm 1.
        - ``"raw"``: features as received by this backbone, before ``lin1``.
          In the standard TopoBench composition this is post-feature-encoder
          data, because ``AllCellFeatureEncoder`` runs before every backbone.
        - ``"lifted"``: the post-``lin1`` encoded features. Off-spec — kept
          as an ablation knob but not the default behaviour.
    **kwargs : dict
        Ignored. Present so the encoder can be safely instantiated via
        Hydra configs that pass extra wrapper-level keys.

    Attributes
    ----------
    lin1, lin2 : nn.Linear
        Input lift to ``hidden_dim`` and output projection.
    lin_left_weights, lin_right_weights : nn.ModuleList
        Per-layer ``W₁`` (acts on stalk dimension) and ``W₂`` (acts on
        hidden channels) of paper Eq. 5.
    epsilons : nn.ParameterList
        Per-layer residual gates initialised to zero (so the first
        training step recovers a pure diffusion step).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        stalk_dim: int = 3,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        connection_features: str = "raw",
        **kwargs,
    ):
        super().__init__()
        assert stalk_dim >= 2, (
            "Conn-NSD is only meaningful for stalk_dim ≥ 2 (d=1 collapses "
            f"to a normalised GCN). Got stalk_dim={stalk_dim}."
        )
        assert hidden_dim % stalk_dim == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by stalk_dim "
            f"({stalk_dim}) so each stalk component has equal channel width."
        )
        assert connection_features in {"raw", "lifted"}, (
            f"connection_features must be 'raw' or 'lifted', "
            f"got {connection_features!r}"
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stalk_dim = stalk_dim
        self.channels_per_stalk = hidden_dim // stalk_dim
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.connection_features = connection_features

        # Input lift / output projection.
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Per-layer diffusion weights. W₁ ∈ ℝ^{d × d}, W₂ ∈ ℝ^{c × c}.
        self.lin_left_weights = nn.ModuleList()
        self.lin_right_weights = nn.ModuleList()
        for _ in range(num_layers):
            left = nn.Linear(stalk_dim, stalk_dim, bias=False)
            nn.init.eye_(left.weight.data)
            self.lin_left_weights.append(left)
            right = nn.Linear(
                self.channels_per_stalk, self.channels_per_stalk, bias=False
            )
            nn.init.orthogonal_(right.weight.data)
            self.lin_right_weights.append(right)

        # Residual gate ε per layer; tanh(0) = 0 so the initial step is a
        # pure diffusion step.
        self.epsilons = nn.ParameterList(
            nn.Parameter(torch.zeros((stalk_dim, 1)))
            for _ in range(num_layers)
        )

    # ------------------------------------------------------------------
    # Forward — Algorithm 1 (pre-process) ∘ paper Eq. 5 (diffusion).
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        edge_weight: Tensor | None = None,
        batch: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Run Conn-NSD on a single graph (or a PyG-batched union of graphs).

        Parameters
        ----------
        x : torch.Tensor, shape ``[N, input_dim]``
            Node features.
        edge_index : torch.Tensor, shape ``[2, E]``
            Edge indices. Made bidirectional internally; the diffusion
            requires a symmetric sheaf Laplacian.
        edge_attr, edge_weight : ignored
            Conn-NSD operates on node features only, by construction.
        batch : torch.Tensor, optional
            PyG batch vector. Used to keep the local-PCA fallback inside each
            graph of a mini-batch.
        **kwargs : dict
            Ignored. Present for forward-call compatibility with the
            generic ``GNNWrapper`` signature.

        Returns
        -------
        torch.Tensor, shape ``[N, hidden_dim]``
            Node embeddings after ``num_layers`` diffusion steps.
        """
        num_nodes = x.size(0)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # ---- Step 1. Build the deterministic connection (Algorithm 1). ----
        # Detached by construction; see conn_nsd_utils.connection.
        if self.connection_features == "raw":
            features_for_connection = x
        else:  # "lifted"
            features_for_connection = self.lin1(x).detach()
        restriction_maps = build_connection(
            features_for_connection,
            edge_index,
            stalk_dim=self.stalk_dim,
            batch=batch,
        )  # [E, d, d]

        # ---- Step 2. Assemble the normalised Δ_F once. -------------------
        laplacian_builder = FixedConnectionLaplacianBuilder(
            num_nodes, edge_index, d=self.stalk_dim
        )
        sparse_laplacian, _ = laplacian_builder(restriction_maps)
        l_indices, l_values = sparse_laplacian

        # ---- Step 3. Sheaf diffusion (paper Eq. 5). ----------------------
        h = F.dropout(x, p=self.input_dropout, training=self.training)
        h = F.elu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Reshape to [N · d, c]: per node, d stalk vectors of width c.
        h = h.view(num_nodes * self.stalk_dim, self.channels_per_stalk)

        residual = h
        for layer_idx in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self._left_right_linear(
                h,
                self.lin_left_weights[layer_idx],
                self.lin_right_weights[layer_idx],
                num_nodes,
            )
            # Sparse mat-vec with the (fixed) normalised sheaf Laplacian.
            h = torch_sparse.spmm(l_indices, l_values, h.size(0), h.size(0), h)
            h = F.elu(h)

            # Residual gate, same form as Bodnar's bundle diffusion:
            #     x0 ← (1 + tanh(ε)) ⊙ x0  −  x
            gate = 1.0 + torch.tanh(self.epsilons[layer_idx]).tile(
                num_nodes, 1
            )
            residual = gate * residual - h
            h = residual

        h = h.reshape(num_nodes, self.hidden_dim)
        return self.lin2(h)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _left_right_linear(
        self,
        h: Tensor,
        left: nn.Linear,
        right: nn.Linear,
        num_nodes: int,
    ) -> Tensor:
        """Apply ``W₁`` along the stalk dim and ``W₂`` along the channel dim.

        Equivalent to ``(I_n ⊗ W₁) X W₂`` from paper Eq. 5, written as two
        matmuls on the ``[N · d, c]`` reshape. We follow the data layout
        used by the Bodnar bundle diffusion to keep the diffusion contract
        identical.

        Parameters
        ----------
        h : torch.Tensor, shape ``[N * d, c]``
            Stacked-stalk hidden state.
        left : nn.Linear
            ``W₁`` — acts on the stalk dimension ``d``.
        right : nn.Linear
            ``W₂`` — acts on the channel dimension ``c``.
        num_nodes : int
            Number of nodes ``N`` in the current graph.

        Returns
        -------
        torch.Tensor, shape ``[N * d, c]``
            Transformed hidden state.
        """
        # Stalk-mixing: reshape so each row is a stalk vector of width d.
        h = h.t().reshape(-1, self.stalk_dim)
        h = left(h)
        h = h.reshape(-1, num_nodes * self.stalk_dim).t()
        # Channel-mixing.
        return right(h)
