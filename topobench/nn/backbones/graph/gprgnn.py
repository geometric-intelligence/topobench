"""Generalized PageRank GNN (GPR-GNN) backbone.

This module implements the GPR-GNN model and its learnable Generalized
PageRank propagation layer for the TopoBench framework.

References
----------
.. [1] Eli Chien, Jianhao Peng, Pan Li, Olgica Milenkovic. "Adaptive
   Universal Generalized PageRank Graph Neural Network." ICLR 2021.
   https://arxiv.org/abs/2006.07988
   Official implementation: https://github.com/jianhao2016/GPRGNN
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

VALID_INITS = ("SGC", "PPR", "NPPR", "Random", "WS")


class GPRProp(MessagePassing):
    r"""Generalized PageRank propagation layer.

    Implements the GPR propagation of Chien et al. (2021): given node
    hidden features :math:`H`, it computes
    :math:`Z = \sum_{k=0}^{K} \gamma_k \tilde{A}^{k} H`, where
    :math:`\tilde{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}` is the
    symmetrically normalized adjacency with self-loops and the
    coefficients :math:`\gamma_k` are learnable (Eq. 5-6 in the paper).
    Adapting :math:`\gamma_k` lets the layer realise low- or high-pass
    filters, which is what makes GPR-GNN robust to heterophily.

    Parameters
    ----------
    K : int
        Number of propagation steps (highest power of the adjacency).
    alpha : float
        Teleport / decay coefficient used to initialize the GPR weights.
    init : {"SGC", "PPR", "NPPR", "Random", "WS"}, optional
        Initialization scheme for the GPR coefficients (default: "PPR").
    gamma : torch.Tensor, optional
        Warm-start coefficients of shape ``(K + 1,)``, required when
        ``init="WS"`` (default: None).
    **kwargs
        Additional arguments forwarded to
        :class:`torch_geometric.nn.MessagePassing`.
    """

    def __init__(self, K, alpha, init="PPR", gamma=None, **kwargs):
        super().__init__(aggr="add", **kwargs)
        if init not in VALID_INITS:
            raise ValueError(
                f"Unknown init scheme {init!r}; expected one of {VALID_INITS}."
            )
        self.K = K
        self.alpha = alpha
        self.init = init
        self.gamma = gamma
        self.temp = Parameter(torch.empty(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the learnable GPR coefficients to their initial values."""
        with torch.no_grad():
            k = torch.arange(self.K + 1, dtype=self.temp.dtype)
            if self.init == "SGC":
                # SGC limit: a single non-zero coefficient at hop ``alpha``.
                self.temp.zero_()
                self.temp[int(self.alpha)] = 1.0
            elif self.init == "PPR":
                # Personalized PageRank: gamma_k = alpha (1 - alpha)^k.
                self.temp.copy_(self.alpha * (1 - self.alpha) ** k)
                self.temp[-1] = (1 - self.alpha) ** self.K
            elif self.init == "NPPR":
                # Negative PPR: gamma_k = alpha^k, L1-normalized.
                temp = self.alpha**k
                self.temp.copy_(temp / temp.abs().sum())
            elif self.init == "Random":
                # Uniform init scaled by sqrt(3 / (K + 1)), L1-normalized.
                bound = (3.0 / (self.K + 1)) ** 0.5
                self.temp.uniform_(-bound, bound)
                self.temp.div_(self.temp.abs().sum())
            elif self.init == "WS":
                # Warm start from externally provided coefficients.
                if self.gamma is None:
                    raise ValueError("init='WS' requires `gamma`.")
                self.temp.copy_(torch.as_tensor(self.gamma))

    def forward(self, x, edge_index, edge_weight=None):
        """Propagate features with learnable Generalized PageRank weights.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, num_features)``.
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape ``(2, num_edges)``.
        edge_weight : torch.Tensor, optional
            Optional edge weights of shape ``(num_edges,)`` (default: None).

        Returns
        -------
        torch.Tensor
            Propagated node features of shape ``(num_nodes, num_features)``.
        """
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        # k = 0 term of the sum: gamma_0 * H.
        hidden = x * self.temp[0]
        for k in range(self.K):
            # Apply one more hop of \tilde{A}, then add the gamma_{k+1} term.
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        """Scale neighbor messages by the normalized adjacency weights.

        Parameters
        ----------
        x_j : torch.Tensor
            Source-node features gathered for each edge.
        norm : torch.Tensor
            Symmetric normalization coefficient for each edge.

        Returns
        -------
        torch.Tensor
            The normalized messages.
        """
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        """Return a string representation of the layer."""
        return f"{self.__class__.__name__}(K={self.K}, temp={self.temp})"


class GPRGNN(torch.nn.Module):
    r"""Generalized PageRank Graph Neural Network backbone.

    GPR-GNN (Chien et al., 2021) decouples feature transformation from
    propagation: an MLP first maps the input features to hidden
    embeddings, and a :class:`GPRProp` layer then mixes information from
    multiple hops with learnable weights :math:`\gamma_k`. This adaptive
    multi-hop mixing lets the model interpolate between low- and
    high-pass behaviour, making it robust across both homophilic and
    heterophilic graphs.

    The backbone returns node embeddings of dimension ``hidden_channels``;
    the task-specific classification head is handled by the TopoBench
    readout, so no final projection or log-softmax is applied here.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden units; also the dimension of the returned node
        embeddings.
    K : int, optional
        Number of propagation steps (default: 10).
    alpha : float, optional
        Coefficient used to initialize the GPR weights (default: 0.1).
    init : {"SGC", "PPR", "NPPR", "Random", "WS"}, optional
        Initialization scheme for the GPR coefficients (default: "PPR").
    gamma : torch.Tensor, optional
        Warm-start coefficients, required when ``init="WS"``
        (default: None).
    dprate : float, optional
        Dropout rate applied to the hidden features before propagation
        (default: 0.5).
    dropout : float, optional
        Dropout rate applied within the feature-transformation MLP
        (default: 0.5).
    **kwargs
        Additional arguments (ignored), kept for config compatibility.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        K=10,
        alpha=0.1,
        init="PPR",
        gamma=None,
        dprate=0.5,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.prop1 = GPRProp(K, alpha, init, gamma)
        self.dprate = dprate
        self.dropout = dropout
        self.out_channels = hidden_channels

    def reset_parameters(self):
        """Reset all learnable parameters of the backbone."""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        """Compute node embeddings with decoupled GPR propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape ``(2, num_edges)``.
        edge_weight : torch.Tensor, optional
            Optional edge weights of shape ``(num_edges,)`` (default: None).
        **kwargs
            Additional arguments (e.g. ``batch``) ignored by this backbone.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``(num_nodes, hidden_channels)``.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop1(x, edge_index, edge_weight=edge_weight)
