"""OGFormer: a graph Transformer with optimized attention scores [1].

OGFormer replaces the multi-layer multi-head attention of the vanilla
Transformer with a simplified Single-head Self-Attention (SSA) mechanism
built on a symmetric positive-definite kernel. Local structural information
is injected as an unbiased additive term on the attention scores
(structural encoding, Eqs. (9)-(11) of [1]) instead of being compressed
into the node features. Attention scores are further optimized end-to-end
with the Neighborhood Maximum Homogeneity loss (Eqs. (13)-(16) of [1]),
implemented in :class:`topobench.loss.model.OGFormerLoss`.

Adapted from the official implementation
https://github.com/LittleBlackBearLiXin/OGFormer.

[1] Zhang et al. "A graph transformer with optimized attention scores for
node classification". Scientific Reports (2025).
https://doi.org/10.1038/s41598-025-15551-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def standardize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Standardize each row of a matrix to zero mean and unit variance.

    Implements :math:`\hat{Q}_i = (Q_i - \mu_i) / (\sigma_i + \epsilon)`
    used inside the attention score computation (Eq. (7) of the paper).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [num_nodes, num_features].
    eps : float, optional
        Small value preventing division by zero (default: 1e-8).

    Returns
    -------
    torch.Tensor
        Row-standardized tensor with the same shape as the input.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + eps)


def symmetric_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    r"""Symmetrically normalize a dense adjacency/attention matrix.

    Computes :math:`D^{-1/2} A D^{-1/2}` (GCN-style normalization), one of
    the normalization options for the attention score matrix in Eq. (7) of
    the paper. Rows with zero degree are left untouched.

    Parameters
    ----------
    adjacency : torch.Tensor
        Dense matrix of shape [num_nodes, num_nodes] with non-negative
        entries.

    Returns
    -------
    torch.Tensor
        Symmetrically normalized matrix of shape [num_nodes, num_nodes].
    """
    d_inv_sqrt = torch.pow(adjacency.sum(dim=1), -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * adjacency * d_inv_sqrt.unsqueeze(0)


class OGFormerAttention(nn.Module):
    r"""Single-head self-attention with structural encoding of OGFormer.

    Queries and keys share the same sigmoid projection, defining a
    symmetric positive-definite kernel between node pairs (Eq. (6)). The raw attention scores are the squared dot products of the
    row-standardized queries, row-normalized over all nodes (Eq. (7)).
    Structural encoding enters as an additive bias term ``alpha * se`` on
    the attention scores (Eqs. (9)-(10)).

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Dimension of the shared query/key latent space.
    alpha : float
        Weight of the structural encoding bias. The paper recommends
        values in [0.7, 1] for homophilous graphs and [0, 0.2] for
        heterophilous graphs.
    """

    def __init__(self, in_channels: int, hidden_channels: int, alpha: float):
        super().__init__()
        self.lin_q = nn.Linear(in_channels, hidden_channels)
        self.alpha = alpha
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters following the official implementation."""
        nn.init.xavier_normal_(self.lin_q.weight, gain=1)
        nn.init.normal_(self.lin_q.bias, mean=0.0, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        se: torch.Tensor,
        graph_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute queries and structurally-biased attention scores.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        se : torch.Tensor
            Structural encoding of shape [num_nodes, num_nodes]: the dense
            adjacency matrix at the first layer, the previous layer's
            attention score matrix afterwards (Eq. (10) of the paper).
        graph_mask : torch.Tensor, optional
            Boolean matrix of shape [num_nodes, num_nodes] that is True
            for node pairs belonging to the same graph. Restricts global
            attention to each graph of a batched input (default: None).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Queries ``q`` of shape [num_nodes, hidden_channels] and the
            attention score matrix ``r = Norm((q_hat q_hat^T)^2) + alpha * se``
            of shape [num_nodes, num_nodes].
        """
        q = torch.sigmoid(self.lin_q(x))
        q_hat = standardize_rows(q)
        scores = (q_hat @ q_hat.T).pow(2)
        if graph_mask is not None:
            scores = scores * graph_mask
        scores = F.normalize(scores, p=1, dim=1)
        r = scores + self.alpha * se
        return q, r


class OGFormerPropagation(nn.Module):
    r"""Message passing over the optimized attention scores.

    Normalizes the attention score matrix (random-walk or symmetric
    normalization) and aggregates node features through it,
    :math:`Z^l = \hat{R} Z^{l-1} W^{l-1}` (Eq. (11)).

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    out_channels : int
        Number of output features per node.
    sym_norm : bool, optional
        If True use symmetric (GCN-style) normalization of the attention
        scores, otherwise random-walk (row L1) normalization
        (default: False).
    """

    def __init__(
        self, in_channels: int, out_channels: int, sym_norm: bool = False
    ):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.sym_norm = sym_norm
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters following the official implementation."""
        nn.init.xavier_normal_(self.lin.weight, gain=1)
        nn.init.zeros_(self.lin.bias)

    def forward(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Aggregate node features through normalized attention scores.

        Parameters
        ----------
        r : torch.Tensor
            Attention score matrix of shape [num_nodes, num_nodes].
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].

        Returns
        -------
        torch.Tensor
            Aggregated node features of shape [num_nodes, out_channels].
        """
        if self.sym_norm:
            r_hat = symmetric_normalize(r)
        else:
            r_hat = F.normalize(r, p=1, dim=1)
        return self.lin(r_hat @ x)


class OGFormerLayer(nn.Module):
    r"""One OGFormer layer: SSA with structural encoding and residual.

    Implements the node update of Eqs. (5) and (11) of the paper,
    :math:`Z^l = \mathrm{ReLU}(\hat{R} Z^{l-1} W^{l-1}) + f(Z^{l-1})`,
    where the residual branch ``f`` is the shared query projection of the
    attention module.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    out_channels : int
        Number of output features per node (also the query dimension).
    alpha : float
        Structural encoding weight for this layer.
    sym_norm : bool, optional
        Whether to use symmetric instead of random-walk normalization of
        the attention scores (default: False).
    apply_activation : bool, optional
        Whether to apply ReLU to the aggregated messages; the official
        implementation omits it in the last layer (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float,
        sym_norm: bool = False,
        apply_activation: bool = True,
    ):
        super().__init__()
        self.attention = OGFormerAttention(in_channels, out_channels, alpha)
        self.propagation = OGFormerPropagation(
            in_channels, out_channels, sym_norm
        )
        self.apply_activation = apply_activation

    def reset_parameters(self):
        """Reset parameters of the attention and propagation modules."""
        self.attention.reset_parameters()
        self.propagation.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        se: torch.Tensor,
        graph_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Update node embeddings through one OGFormer layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        se : torch.Tensor
            Structural encoding of shape [num_nodes, num_nodes].
        graph_mask : torch.Tensor, optional
            Same-graph mask of shape [num_nodes, num_nodes] for batched
            inputs (default: None).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated node embeddings of shape [num_nodes, out_channels],
            the queries ``q`` and the attention score matrix ``r`` (used
            as the next layer's structural encoding and by the
            Neighborhood Maximum Homogeneity loss).
        """
        q, r = self.attention(x, se, graph_mask)
        h = self.propagation(r, x)
        if self.apply_activation:
            h = F.relu(h)
        return h + q, q, r


class OGFormer(nn.Module):
    r"""OGFormer backbone for node and graph level tasks.

    Stacks :class:`OGFormerLayer` modules. The structural encoding of the
    first layer is the dense (block-diagonal for batched inputs)
    adjacency matrix; deeper layers reuse the previous layer's attention
    score matrix (Eq. (10)). When the module is in training
    mode, the per-layer queries and row-normalized attention scores are
    returned as auxiliary outputs so that
    :class:`topobench.loss.model.OGFormerLoss` can optimize the attention
    scores end-to-end (Eq. (16)).

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Number of hidden (and output) features per node.
    n_layers : int, optional
        Number of OGFormer layers; the paper uses 2 for most datasets
        (default: 2).
    alpha : float or list[float], optional
        Structural encoding weight(s). A single float is broadcast to all
        layers; a list specifies one value per layer. The paper recommends
        [0.7, 1] for homophilous and [0, 0.2] for heterophilous graphs
        (default: 0.8).
    sym_norm : bool, optional
        Use symmetric instead of random-walk normalization of the
        attention scores (default: False).
    dropout : float, optional
        Dropout rate applied to node embeddings between layers
        (default: 0.0).
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_layers: int = 2,
        alpha: float | list[float] = 0.8,
        sym_norm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if isinstance(alpha, (int, float)):
            alphas = [float(alpha)] * n_layers
        else:
            alphas = [float(a) for a in alpha]
        if len(alphas) != n_layers:
            raise ValueError(
                f"Expected {n_layers} alpha values, got {len(alphas)}"
            )

        self.out_channels = hidden_channels
        self.dropout = dropout
        self.layers = nn.ModuleList(
            OGFormerLayer(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                alphas[i],
                sym_norm=sym_norm,
                apply_activation=i < n_layers - 1,
            )
            for i in range(n_layers)
        )

    def reset_parameters(self):
        """Reset parameters of all layers."""
        for layer in self.layers:
            layer.reset_parameters()

    @staticmethod
    def build_dense_adjacency(
        edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build a dense symmetric binary adjacency matrix.

        For batched inputs the (globally indexed) edges of different
        graphs never overlap, so the result is block-diagonal.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges].
        num_nodes : int
            Total number of nodes.
        dtype : torch.dtype
            Data type of the resulting matrix.

        Returns
        -------
        torch.Tensor
            Dense adjacency matrix of shape [num_nodes, num_nodes].
        """
        adjacency = torch.zeros(
            num_nodes, num_nodes, dtype=dtype, device=edge_index.device
        )
        adjacency[edge_index[0], edge_index[1]] = 1.0
        adjacency[edge_index[1], edge_index[0]] = 1.0
        return adjacency

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict | None]:
        r"""Forward pass of the OGFormer backbone.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges].
        batch : torch.Tensor, optional
            Batch vector of shape [num_nodes] assigning each node to a
            graph. Attention is restricted to nodes of the same graph
            (default: None).
        **kwargs : dict
            Additional arguments (ignored; e.g. ``edge_weight``).

        Returns
        -------
        tuple[torch.Tensor, dict | None]
            Node embeddings of shape [num_nodes, hidden_channels] and, in
            training mode, a dictionary with the per-layer ``queries`` and
            row-normalized ``attention_scores`` consumed by
            :class:`topobench.loss.model.OGFormerLoss` (None otherwise).
        """
        num_nodes = x.size(0)
        se = self.build_dense_adjacency(edge_index, num_nodes, x.dtype)

        graph_mask = None
        if batch is not None and batch.numel() > 0 and batch.max() > 0:
            graph_mask = (batch.unsqueeze(0) == batch.unsqueeze(1)).to(x.dtype)

        queries, attention_scores = [], []
        for i, layer in enumerate(self.layers):
            if i > 0 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x, q, r = layer(x, se, graph_mask)
            se = r
            if self.training:
                queries.append(q)
                attention_scores.append(F.normalize(r, p=1, dim=1))

        aux = (
            {"queries": queries, "attention_scores": attention_scores}
            if self.training
            else None
        )
        return x, aux
