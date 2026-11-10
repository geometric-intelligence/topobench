"""Graph Inductive Bias Transformer (GRIT) backbone.

This module implements GRIT [1]_, a graph transformer that incorporates
graph inductive biases without message passing, based on three design
choices:

1. Learned relative positional encodings initialized with Relative Random
   Walk Probabilities (RRWP, Eq. (1) in [1]_), precomputed by the
   :class:`topobench.transforms.data_manipulations.AddRRWP` transform (or
   computed on the fly as a fallback).
2. A flexible attention mechanism that jointly updates node and node-pair
   representations (Eq. (2)-(3) in [1]_).
3. Injection of degree information through adaptive degree scalers
   (Eq. (5) in [1]_), paired with batch normalization, since layer
   normalization would cancel the degree information (Proposition 3.3
   in [1]_).

References
----------
.. [1] Ma, Lin, Lim, Romero-Soriano, Dokania, Coates, Torr, Lim.
    "Graph Inductive Biases in Transformers without Message Passing."
    ICML 2023. https://arxiv.org/abs/2305.17589
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import coalesce, scatter, softmax

from topobench.transforms.data_manipulations.rrwp_positional_encodings import (
    compute_rrwp,
)


def full_edge_index(
    batch: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    r"""Build the edge index of the fully-connected graph for each example.

    All node pairs (including self-loops) are generated independently for
    every graph in the batch, so no attention edge crosses two different
    graphs.

    Parameters
    ----------
    batch : torch.Tensor
        Batch vector of shape ``[num_nodes]`` assigning each node to an
        example.
    device : torch.device, optional
        Device of the output tensor (default: the device of ``batch``).

    Returns
    -------
    torch.Tensor
        Edge indices of shape ``[2, num_pairs]`` covering all intra-graph
        node pairs.
    """
    device = batch.device if device is None else device
    num_nodes_per_graph = torch.bincount(batch)
    node_offsets = torch.cat(
        [batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)[:-1]]
    )

    index_list = []
    for offset, num_nodes in zip(
        node_offsets.tolist(), num_nodes_per_graph.tolist(), strict=True
    ):
        nodes = torch.arange(num_nodes, device=device)
        pairs = torch.cartesian_prod(nodes, nodes).t()
        index_list.append(pairs + offset)
    return torch.cat(index_list, dim=1)


class GRITAttention(nn.Module):
    r"""GRIT multi-head attention conditioned on node-pair representations.

    Implements Eq. (2) of the GRIT paper for a sparse set of attention
    pairs: given node representations :math:`\mathbf{x}` and node-pair
    representations :math:`\mathbf{e}`, it computes

    .. math::
        \hat{\mathbf{e}}_{i,j} &= \sigma(\rho((\mathbf{W}_Q \mathbf{x}_i +
        \mathbf{W}_K \mathbf{x}_j) \odot \mathbf{W}_{Ew} \mathbf{e}_{i,j})
        + \mathbf{W}_{Eb} \mathbf{e}_{i,j}) \\
        \alpha_{ij} &= \mathrm{Softmax}_{j}(\mathbf{W}_A
        \hat{\mathbf{e}}_{i,j}) \\
        \hat{\mathbf{x}}_i &= \sum_j \alpha_{ij} (\mathbf{W}_V \mathbf{x}_j
        + \mathbf{W}_{Ev} \hat{\mathbf{e}}_{i,j}),

    where :math:`\rho(x) = \mathrm{ReLU}(x)^{1/2} -
    \mathrm{ReLU}(-x)^{1/2}` is the signed square root that stabilizes
    training.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node and node-pair representations.
    num_heads : int
        Number of attention heads; must divide ``hidden_dim``.
    attn_dropout : float, optional
        Dropout applied to the attention weights (default: 0).
    clamp : float, optional
        Symmetric clamping value for attention logits; ``None`` disables
        clamping (default: 5.0).
    edge_enhance : bool, optional
        Whether to add the pair-representation term
        :math:`\mathbf{W}_{Ev}\hat{\mathbf{e}}_{i,j}` to the aggregated
        node update (default: True).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        clamp: float | None = 5.0,
        edge_enhance: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.clamp = abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_E = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_E.weight)
        nn.init.xavier_normal_(self.W_V.weight)

        # W_A in Eq. (2): maps each head's pair representation to a logit.
        self.W_A = nn.Parameter(torch.zeros(self.head_dim, num_heads, 1))
        nn.init.xavier_normal_(self.W_A)

        if self.edge_enhance:
            # W_Ev in Eq. (2), applied after aggregating \hat{e}_{i,j}.
            self.W_Ev = nn.Parameter(
                torch.zeros(self.head_dim, num_heads, self.head_dim)
            )
            nn.init.xavier_normal_(self.W_Ev)

    @staticmethod
    def signed_sqrt(x: torch.Tensor) -> torch.Tensor:
        r"""Signed square root :math:`\rho` from Eq. (2) of the GRIT paper.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Elementwise signed square root of the input.
        """
        return torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))

    def forward(
        self,
        x: torch.Tensor,
        pair_index: torch.Tensor,
        pair_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute the attention update for nodes and node pairs.

        Parameters
        ----------
        x : torch.Tensor
            Node representations of shape ``[num_nodes, hidden_dim]``.
        pair_index : torch.Tensor
            Attention pairs of shape ``[2, num_pairs]``; row 0 holds the
            source (attended) nodes, row 1 the target (attending) nodes.
        pair_attr : torch.Tensor
            Node-pair representations of shape ``[num_pairs, hidden_dim]``.

        Returns
        -------
        x_out : torch.Tensor
            Updated node representations, ``[num_nodes, num_heads,
            head_dim]``.
        pair_out : torch.Tensor
            Updated node-pair representations, ``[num_pairs, hidden_dim]``.
        """
        source, target = pair_index[0], pair_index[1]
        num_nodes = x.size(0)

        q = self.W_Q(x).view(-1, self.num_heads, self.head_dim)
        k = self.W_K(x).view(-1, self.num_heads, self.head_dim)
        v = self.W_V(x).view(-1, self.num_heads, self.head_dim)
        e = self.W_E(pair_attr).view(-1, self.num_heads, self.head_dim * 2)
        e_w, e_b = e[..., : self.head_dim], e[..., self.head_dim :]

        # \hat{e}_{i,j} (Eq. 2), with i = target, j = source.
        score = q[target] + k[source]
        score = self.signed_sqrt(score * e_w) + e_b
        pair_update = torch.relu(score)

        # Attention logits and per-target softmax.
        logits = torch.einsum("phd,dho->pho", pair_update, self.W_A)
        if self.clamp is not None:
            logits = torch.clamp(logits, min=-self.clamp, max=self.clamp)
        alpha = softmax(logits, index=target, num_nodes=num_nodes)
        alpha = self.attn_dropout(alpha)

        # Aggregate messages towards target nodes.
        x_out = scatter(
            v[source] * alpha,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        if self.edge_enhance:
            pair_agg = scatter(
                pair_update * alpha,
                target,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
            x_out = x_out + torch.einsum("nhd,dhc->nhc", pair_agg, self.W_Ev)

        return x_out, pair_update.flatten(1)


class GRITTransformerLayer(nn.Module):
    r"""GRIT transformer layer.

    Combines the GRIT attention mechanism with the adaptive degree scaler
    (Eq. (5) in the GRIT paper), residual connections, batch normalization
    and a feed-forward network. Batch normalization is used instead of
    layer normalization because the latter cancels out degree information
    (Proposition 3.3 in the GRIT paper).

    Parameters
    ----------
    hidden_dim : int
        Dimension of node and node-pair representations.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout applied to node/pair updates and inside the FFN
        (default: 0).
    attn_dropout : float, optional
        Dropout applied to the attention weights (default: 0).
    clamp : float, optional
        Symmetric clamping value for attention logits (default: 5.0).
    deg_scaler : bool, optional
        Whether to apply the adaptive degree scaler of Eq. (5)
        (default: True).
    edge_enhance : bool, optional
        Whether the attention update uses the pair-representation term
        (default: True).
    update_pair_rep : bool, optional
        Whether to update the node-pair representations (learned RRWP);
        if False, the input pair representations are passed through
        unchanged (default: True).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        clamp: float | None = 5.0,
        deg_scaler: bool = True,
        edge_enhance: bool = True,
        update_pair_rep: bool = True,
    ):
        super().__init__()
        self.deg_scaler = deg_scaler
        self.update_pair_rep = update_pair_rep
        self.dropout = dropout

        self.attention = GRITAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            clamp=clamp,
            edge_enhance=edge_enhance,
        )

        self.W_O_node = nn.Linear(hidden_dim, hidden_dim)
        self.W_O_pair = nn.Linear(hidden_dim, hidden_dim)

        if self.deg_scaler:
            # theta_1, theta_2 in Eq. (5), stacked on the last dimension.
            self.deg_coef = nn.Parameter(torch.zeros(1, hidden_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        self.norm1_node = nn.BatchNorm1d(hidden_dim)
        self.norm1_pair = nn.BatchNorm1d(hidden_dim)
        self.norm2_node = nn.BatchNorm1d(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        pair_index: torch.Tensor,
        pair_attr: torch.Tensor,
        log_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Update node and node-pair representations.

        Parameters
        ----------
        x : torch.Tensor
            Node representations of shape ``[num_nodes, hidden_dim]``.
        pair_index : torch.Tensor
            Attention pairs of shape ``[2, num_pairs]``.
        pair_attr : torch.Tensor
            Node-pair representations of shape ``[num_pairs, hidden_dim]``.
        log_deg : torch.Tensor
            Log-degrees :math:`\log(1 + d_i)` of shape ``[num_nodes, 1]``.

        Returns
        -------
        x : torch.Tensor
            Updated node representations, ``[num_nodes, hidden_dim]``.
        pair_attr : torch.Tensor
            Updated node-pair representations, ``[num_pairs, hidden_dim]``.
        """
        x_res, pair_res = x, pair_attr

        x_attn, pair_update = self.attention(x, pair_index, pair_attr)

        x = F.dropout(
            x_attn.flatten(1), p=self.dropout, training=self.training
        )
        if self.deg_scaler:
            # x' = x * theta_1 + log(1 + d) * x * theta_2 (Eq. 5).
            x = torch.stack([x, x * log_deg], dim=-1)
            x = (x * self.deg_coef).sum(dim=-1)
        x = self.W_O_node(x)
        x = self.norm1_node(x + x_res)

        if self.update_pair_rep:
            pair_attr = F.dropout(
                pair_update, p=self.dropout, training=self.training
            )
            pair_attr = self.W_O_pair(pair_attr)
            pair_attr = self.norm1_pair(pair_attr + pair_res)
        else:
            pair_attr = pair_res

        x = self.norm2_node(x + self.ffn(x))

        return x, pair_attr


class GRITBackbone(nn.Module):
    r"""GRIT backbone: a graph transformer without message passing.

    The backbone expects node features already projected to ``hidden_dim``
    (e.g. by TopoBench's ``AllCellFeatureEncoder``) and RRWP positional
    encodings attached to the batch by the ``AddRRWP`` transform. When the
    RRWP attributes are missing (e.g. the transform was not configured),
    they are computed on the fly per graph, which is functionally
    equivalent but slower than precomputing them once.

    Following the GRIT paper, node representations are initialized as
    :math:`\mathbf{x}_i = \mathbf{x}'_i + \mathrm{FC}(\mathbf{P}_{i,i})`
    and node-pair representations as :math:`\mathbf{e}_{i,j} =
    \mathrm{FC}(\mathbf{P}_{i,j}) (+ \mathrm{FC}(\mathbf{e}'_{i,j}))`,
    after which a stack of :class:`GRITTransformerLayer` updates both.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node and node-pair representations.
    num_layers : int, optional
        Number of GRIT transformer layers (default: 4).
    num_heads : int, optional
        Number of attention heads (default: 8).
    walk_length : int, optional
        Number of RRWP channels :math:`K`; must match the ``walk_length``
        used by the ``AddRRWP`` transform (default: 8).
    dropout : float, optional
        Dropout applied to node/pair updates and inside the FFN
        (default: 0).
    attn_dropout : float, optional
        Dropout applied to the attention weights (default: 0.2).
    clamp : float, optional
        Symmetric clamping value for attention logits (default: 5.0).
    pad_to_full_graph : bool, optional
        Whether to extend the attention pairs to all intra-graph node
        pairs (global attention). If False, attention is restricted to
        pairs reachable by walks shorter than ``walk_length``
        (default: True).
    deg_scaler : bool, optional
        Whether to apply the adaptive degree scaler (default: True).
    edge_enhance : bool, optional
        Whether the attention update uses the pair-representation term
        (default: True).
    update_pair_rep : bool, optional
        Whether layers update the node-pair representations
        (default: True).
    edge_dim : int, optional
        Dimension of the raw edge attributes. If set, edge attributes are
        linearly projected and fused into the initial node-pair
        representations; if None, edge attributes are ignored
        (default: None).
    **kwargs : dict
        Additional arguments (ignored).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        walk_length: int = 8,
        dropout: float = 0.0,
        attn_dropout: float = 0.2,
        clamp: float | None = 5.0,
        pad_to_full_graph: bool = True,
        deg_scaler: bool = True,
        edge_enhance: bool = True,
        update_pair_rep: bool = True,
        edge_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length
        self.pad_to_full_graph = pad_to_full_graph

        # Linear encoders for the absolute (node) and relative (pair) RRWP.
        self.abs_pe_encoder = nn.Linear(walk_length, hidden_dim, bias=False)
        self.rel_pe_encoder = nn.Linear(walk_length, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.abs_pe_encoder.weight)
        nn.init.xavier_uniform_(self.rel_pe_encoder.weight)

        self.edge_attr_encoder = (
            nn.Linear(edge_dim, hidden_dim) if edge_dim is not None else None
        )

        self.layers = nn.ModuleList(
            [
                GRITTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    clamp=clamp,
                    deg_scaler=deg_scaler,
                    edge_enhance=edge_enhance,
                    update_pair_rep=update_pair_rep,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        rrwp: torch.Tensor | None = None,
        rrwp_index: torch.Tensor | None = None,
        rrwp_val: torch.Tensor | None = None,
        log_deg: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Forward pass of the GRIT backbone.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, hidden_dim]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Batch vector assigning each node to an example; a single graph
            is assumed if None (default: None).
        edge_attr : torch.Tensor, optional
            Raw edge attributes of shape ``[num_edges, edge_dim]``; only
            used when ``edge_dim`` was configured (default: None).
        rrwp : torch.Tensor, optional
            Precomputed node-level RRWP of shape ``[num_nodes,
            walk_length]`` (default: None).
        rrwp_index : torch.Tensor, optional
            Precomputed relative RRWP indices, ``[2, num_pairs]``
            (default: None).
        rrwp_val : torch.Tensor, optional
            Precomputed relative RRWP values, ``[num_pairs, walk_length]``
            (default: None).
        log_deg : torch.Tensor, optional
            Precomputed log-degrees of shape ``[num_nodes]``
            (default: None).
        **kwargs : dict
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Updated node representations of shape ``[num_nodes,
            hidden_dim]``.
        """
        num_nodes = x.size(0)
        if batch is None:
            batch = x.new_zeros(num_nodes, dtype=torch.long)

        if any(attr is None for attr in (rrwp, rrwp_index, rrwp_val, log_deg)):
            rrwp, rrwp_index, rrwp_val, log_deg = self._rrwp_fallback(
                edge_index, batch
            )
        if rrwp_val.size(-1) != self.walk_length:
            raise ValueError(
                f"RRWP encodings have {rrwp_val.size(-1)} channels but the "
                f"backbone was configured with walk_length="
                f"{self.walk_length}. Make sure the AddRRWP transform and "
                f"the GRIT backbone use the same walk_length."
            )

        # Initialize node and node-pair representations from RRWP.
        x = x + self.abs_pe_encoder(rrwp)
        pair_index, pair_attr = self._build_pair_representations(
            rrwp_index, rrwp_val, edge_index, edge_attr, batch, num_nodes
        )

        log_deg = log_deg.view(num_nodes, 1)
        for layer in self.layers:
            x, pair_attr = layer(x, pair_index, pair_attr, log_deg)

        return x

    def _build_pair_representations(
        self,
        rrwp_index: torch.Tensor,
        rrwp_val: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Build the initial node-pair representations.

        Combines (by sum over coinciding pairs) the projected relative
        RRWP, optionally the projected edge attributes, and, when
        ``pad_to_full_graph`` is set, zero paddings for all remaining
        intra-graph node pairs.

        Parameters
        ----------
        rrwp_index : torch.Tensor
            Relative RRWP indices of shape ``[2, num_pairs]``.
        rrwp_val : torch.Tensor
            Relative RRWP values of shape ``[num_pairs, walk_length]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``.
        edge_attr : torch.Tensor, optional
            Raw edge attributes of shape ``[num_edges, edge_dim]``.
        batch : torch.Tensor
            Batch vector of shape ``[num_nodes]``.
        num_nodes : int
            Total number of nodes in the batch.

        Returns
        -------
        pair_index : torch.Tensor
            Attention pair indices of shape ``[2, num_pairs_out]``.
        pair_attr : torch.Tensor
            Initial pair representations, ``[num_pairs_out, hidden_dim]``.
        """
        pair_index = rrwp_index
        pair_attr = self.rel_pe_encoder(rrwp_val)

        if edge_attr is not None and self.edge_attr_encoder is not None:
            edge_attr = edge_attr.view(edge_index.size(1), -1).float()
            pair_index = torch.cat([pair_index, edge_index], dim=1)
            pair_attr = torch.cat(
                [pair_attr, self.edge_attr_encoder(edge_attr)], dim=0
            )

        if self.pad_to_full_graph:
            padding_index = full_edge_index(batch)
            padding_attr = pair_attr.new_zeros(
                padding_index.size(1), pair_attr.size(1)
            )
            pair_index = torch.cat([pair_index, padding_index], dim=1)
            pair_attr = torch.cat([pair_attr, padding_attr], dim=0)

        return coalesce(
            pair_index, pair_attr, num_nodes=num_nodes, reduce="sum"
        )

    @torch.no_grad()
    def _rrwp_fallback(
        self,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute RRWP encodings on the fly for a (batched) graph.

        Each graph in the batch is processed independently, which is
        equivalent to applying the ``AddRRWP`` transform before batching.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``.
        batch : torch.Tensor
            Batch vector of shape ``[num_nodes]``.

        Returns
        -------
        rrwp : torch.Tensor
            Node-level RRWP, ``[num_nodes, walk_length]``.
        rrwp_index : torch.Tensor
            Relative RRWP indices, ``[2, num_pairs]``.
        rrwp_val : torch.Tensor
            Relative RRWP values, ``[num_pairs, walk_length]``.
        log_deg : torch.Tensor
            Log-degrees, ``[num_nodes]``.
        """
        num_nodes_per_graph = torch.bincount(batch, minlength=1)
        node_offsets = torch.cat(
            [batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)[:-1]]
        )
        edge_graph = batch[edge_index[0]]

        abs_list, index_list, val_list, deg_list = [], [], [], []
        for graph_id, (offset, graph_num_nodes) in enumerate(
            zip(
                node_offsets.tolist(),
                num_nodes_per_graph.tolist(),
                strict=True,
            )
        ):
            graph_edge_index = edge_index[:, edge_graph == graph_id] - offset
            abs_pe, rel_index, rel_val, deg = compute_rrwp(
                graph_edge_index, graph_num_nodes, self.walk_length
            )
            abs_list.append(abs_pe)
            index_list.append(rel_index + offset)
            val_list.append(rel_val)
            deg_list.append(deg)

        return (
            torch.cat(abs_list, dim=0),
            torch.cat(index_list, dim=1),
            torch.cat(val_list, dim=0),
            torch.log1p(torch.cat(deg_list, dim=0)),
        )
