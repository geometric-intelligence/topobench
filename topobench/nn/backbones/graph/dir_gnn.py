"""Directed Graph Neural Network (Dir-GNN).

Implements the Dir-GNN framework from:

    Rossi et al., "Edge Directionality Improves Learning on
    Heterophilic Graphs", Learning on Graphs Conference, 2024.
    https://arxiv.org/abs/2305.10498

The key insight is that treating a graph as *directed* increases the
effective homophily of the graph in heterophilic settings, because
the separate in- and out-neighbourhoods are individually more
homophilic than the combined symmetric neighbourhood.

Dir-GNN extends any Message Passing Neural Network (MPNN) to
directed graphs by performing **separate aggregations** of incoming
and outgoing edges (Eq. 4 in the paper)::

    h_v^{(l+1)} = sigma(
        W_self * h_v^{(l)}
        + alpha     * W_in  * AGG_{u:(u,v)∈E}( h_u^{(l)} )
        + (1-alpha) * W_out * AGG_{u:(v,u)∈E}( h_u^{(l)} )
    )

where alpha ∈ [0, 1] controls the balance between in- and
out-neighbour aggregations.

Theoretical result (Theorem 1, Rossi et al.): Dir-GNN matches the
expressivity of the Directed Weisfeiler-Lehman (DWL) test, strictly
exceeding conventional undirected MPNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class DirSageConv(MessagePassing):
    """Single Dir-GNN convolutional layer (mean / degree-normalised).

    Performs separate mean-aggregations over in-neighbours and
    out-neighbours, then combines them with the node's own features
    via learned linear maps (Eq. 4 in Rossi et al., 2024).

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    out_channels : int
        Dimensionality of output node features.
    alpha : float, optional
        Mixing coefficient in [0, 1] for in- vs out-aggregation.
        ``alpha = 0.5`` gives equal weight to both directions.
        Default: ``0.5``.

    Examples
    --------
    >>> import torch
    >>> from topobench.nn.backbones.graph.dir_gnn import DirSageConv
    >>> conv = DirSageConv(in_channels=8, out_channels=16)
    >>> x = torch.randn(10, 8)
    >>> edge_index = torch.randint(0, 10, (2, 20))
    >>> out = conv(x, edge_index)
    >>> out.shape
    torch.Size([10, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 0.5,
    ) -> None:
        # Use 'add' aggr; we apply degree normalisation manually so
        # that in-degree and out-degree normalisation are kept separate.
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha

        # Separate transforms for in-agg, out-agg, and identity (self).
        # Eq. 4: W_in, W_out, W_self from the paper.
        self.lin_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise weights with Glorot uniform initialisation."""
        nn.init.xavier_uniform_(self.lin_in.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass for one Dir-GNN layer.

        Parameters
        ----------
        x : Tensor, shape [N, in_channels]
            Node feature matrix.
        edge_index : Tensor, shape [2, E]
            Edge indices in COO format.
            ``edge_index[0]`` = source nodes,
            ``edge_index[1]`` = target nodes.

        Returns
        -------
        Tensor
            Updated node embeddings of shape [N, out_channels].
        """
        num_nodes = x.size(0)

        # --- In-neighbour aggregation (Eq. 4, first agg term) -------
        # Standard propagation: messages flow from src → dst.
        # dst receives from its *in-neighbours* (nodes that point to it).
        deg_in = degree(
            edge_index[1], num_nodes=num_nodes, dtype=x.dtype
        ).clamp(min=1.0)
        # Per-edge norm: 1 / deg_in[dst]
        norm_in = 1.0 / deg_in[edge_index[1]]
        agg_in = self.propagate(edge_index, x=x, norm=norm_in)

        # --- Out-neighbour aggregation (Eq. 4, second agg term) ------
        # Reverse edges so that each node now *receives* from its
        # out-neighbours (nodes it originally pointed to).
        edge_index_t = edge_index.flip(0)  # shape [2, E]: [dst, src]
        deg_out = degree(
            edge_index_t[1], num_nodes=num_nodes, dtype=x.dtype
        ).clamp(min=1.0)
        norm_out = 1.0 / deg_out[edge_index_t[1]]
        agg_out = self.propagate(edge_index_t, x=x, norm=norm_out)

        # --- Combine (Eq. 4) -----------------------------------------
        out = (
            self.lin_self(x)
            + self.alpha * self.lin_in(agg_in)
            + (1.0 - self.alpha) * self.lin_out(agg_out)
            + self.bias
        )
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """Compute degree-normalised messages.

        Parameters
        ----------
        x_j : Tensor, shape [E, in_channels]
            Source-node features for each edge.
        norm : Tensor, shape [E]
            Per-edge normalisation factor (1 / degree of target node).

        Returns
        -------
        Tensor
            Normalised messages of shape [E, in_channels].
        """
        return norm.view(-1, 1) * x_j


class DirGNN(nn.Module):
    """Directed Graph Neural Network (Dir-GNN).

    A multi-layer stack of :class:`DirSageConv` layers with ReLU
    activations and optional dropout.  Implements the framework from
    Rossi et al. (2024) which proves that separate in/out aggregations
    match the expressive power of the Directed Weisfeiler-Lehman test.

    The model is particularly effective on heterophilic graphs, where
    using directed edges provably increases effective homophily compared
    to symmetrised (undirected) variants.

    The forward signature accepts ``**kwargs`` so that the standard
    :class:`~topobench.nn.wrappers.GNNWrapper` (which passes
    ``batch`` and ``edge_weight``) can wrap this backbone without
    modification.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_channels : int
        Dimensionality of hidden layer representations.
    out_channels : int
        Dimensionality of the final output representations.
    num_layers : int, optional
        Total number of Dir-GNN convolutional layers.  Default: ``2``.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
        Default: ``0.5``.
    alpha : float, optional
        In/out mixing coefficient passed to every
        :class:`DirSageConv`.  Default: ``0.5``.

    Examples
    --------
    >>> import torch
    >>> from topobench.nn.backbones.graph.dir_gnn import DirGNN
    >>> model = DirGNN(
    ...     in_channels=16, hidden_channels=64,
    ...     out_channels=7, num_layers=2,
    ... )
    >>> x = torch.randn(100, 16)
    >>> edge_index = torch.randint(0, 100, (2, 300))
    >>> out = model(x, edge_index)
    >>> out.shape
    torch.Size([100, 7])
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.out_channels = out_channels

        if num_layers < 1:
            msg = f"num_layers must be >= 1, got {num_layers}"
            raise ValueError(msg)

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                DirSageConv(in_channels, out_channels, alpha=alpha)
            )
        else:
            # Input layer
            self.convs.append(
                DirSageConv(in_channels, hidden_channels, alpha=alpha)
            )
            # Intermediate hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(
                    DirSageConv(hidden_channels, hidden_channels, alpha=alpha)
                )
            # Output layer
            self.convs.append(
                DirSageConv(hidden_channels, out_channels, alpha=alpha)
            )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tensor:
        """Forward pass through all Dir-GNN layers.

        Parameters
        ----------
        x : Tensor, shape [N, in_channels]
            Node feature matrix.
        edge_index : Tensor, shape [2, E]
            Edge connectivity in COO format.
        **kwargs : dict
            Ignored extra keyword arguments (e.g. ``batch``,
            ``edge_weight``) passed by
            :class:`~topobench.nn.wrappers.GNNWrapper`.

        Returns
        -------
        Tensor
            Node-level output representations of shape
            [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
