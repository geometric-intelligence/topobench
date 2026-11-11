"""Neural building blocks of the SMCN backbone.

These layers implement the components of Scalable Multi-Cellular
Networks: the CIN-style higher-order message-passing block, the marked
subcomplex-bag layers (SCL), and the bag initialization/pooling. The
semantics follow the official SMCN implementation
(https://github.com/yoavgelberg/SMCN); the code here is written for the
batched structures of
:mod:`topobench.nn.backbones.cell.smcn_utils.structures`.

References
----------
Eitan et al. "Topological Blindspots: Understanding and Extending
Topological Deep Learning Through the Lens of Expressivity." ICLR 2025.
https://arxiv.org/abs/2408.05486
Bodnar et al. "Weisfeiler and Lehman Go Cellular: CW Networks."
NeurIPS 2021. https://arxiv.org/abs/2106.12575
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GINConv


class SafeBatchNorm(nn.BatchNorm1d):
    """Batch norm that passes single-row inputs through unchanged.

    A batch can contain a single 2-cell in total (sparse graphs under
    the cycle lifting), and batch statistics are undefined for one row,
    so :class:`torch.nn.BatchNorm1d` raises during training. Evaluation
    mode is unaffected: it uses the running statistics.
    """

    def forward(self, x):
        """Normalize ``x``, skipping single-row batches in training.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``[n, dim]``.

        Returns
        -------
        torch.Tensor
            Normalized features (or ``x`` itself when ``n < 2`` in
            training mode).
        """
        if self.training and x.size(0) < 2:
            return x
        return super().forward(x)


def homp_mlp(dim, num_layers=2):
    """Build the width-preserving MLP used by the CIN-block convolutions.

    Parameters
    ----------
    dim : int
        Input, hidden, and output dimension.
    num_layers : int, optional
        Number of Linear-BatchNorm-ReLU stages. Default is 2.

    Returns
    -------
    torch.nn.Sequential
        The MLP.
    """
    layers = []
    for _ in range(num_layers):
        layers += [
            nn.Linear(dim, dim),
            SafeBatchNorm(dim),
            nn.ReLU(),
        ]
    return nn.Sequential(*layers)


def scl_mlp(in_dim, hidden_dim):
    """Build the two-stage MLP used by the subcomplex convolutions.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    hidden_dim : int
        Hidden and output dimension.

    Returns
    -------
    torch.nn.Sequential
        The MLP.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        SafeBatchNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        SafeBatchNorm(hidden_dim),
        nn.ReLU(),
    )


class BridgeGIN(MessagePassing):
    """GIN convolution whose messages carry the mediating-cell features.

    Messages between two cells adjacent through a common higher-rank
    cell (the *bridge*) concatenate the source features with the bridge
    features, pass through a linear layer and a ReLU, and are
    sum-aggregated before the usual GIN update.

    Parameters
    ----------
    mlp : torch.nn.Module
        Update network applied after aggregation; its first layer
        defines the input width.
    edge_dim : int
        Dimension of the bridge features.
    train_eps : bool, optional
        Whether the GIN epsilon is learnable. Default is True.
    """

    def __init__(self, mlp, edge_dim, train_eps=True):
        super().__init__(aggr="add")
        self.mlp = mlp
        in_dim = mlp[0].in_features
        self.lin = nn.Linear(in_dim + edge_dim, in_dim)
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        """Run the bridge-aware GIN update.

        Parameters
        ----------
        x : torch.Tensor
            Cell features of shape ``[n, in_dim]``.
        edge_index : torch.Tensor
            Adjacency pairs of shape ``[2, P]``.
        edge_attr : torch.Tensor
            Bridge features of shape ``[P, edge_dim]``.
        edge_weight : torch.Tensor, optional
            Multiplicative per-pair gate of shape ``[P]`` (used by the
            learned-lifting variant). Default is None.

        Returns
        -------
        torch.Tensor
            Updated features of shape ``[n, out_dim]``.
        """
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight
        )
        out = out + (1 + self.eps) * x
        return self.mlp(out)

    def message(self, x_j, edge_attr, edge_weight):
        """Compute one bridge-aware message.

        Parameters
        ----------
        x_j : torch.Tensor
            Source-cell features of shape ``[P, in_dim]``.
        edge_attr : torch.Tensor
            Bridge features of shape ``[P, edge_dim]``.
        edge_weight : torch.Tensor or None
            Optional per-pair gate of shape ``[P]``.

        Returns
        -------
        torch.Tensor
            Messages of shape ``[P, in_dim]``.
        """
        msg = self.lin(torch.cat([x_j, edge_attr], dim=-1)).relu()
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        return msg


class ConcatHead(nn.Module):
    """Merge concatenated branch outputs back to the embedding width.

    Parameters
    ----------
    num_branches : int
        Number of concatenated branches.
    dim : int
        Embedding dimension of each branch and of the output.
    """

    def __init__(self, num_branches, dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(num_branches * dim, dim),
            SafeBatchNorm(dim),
            nn.ReLU(),
        )

    def forward(self, parts):
        """Concatenate the branch outputs and apply the head.

        Parameters
        ----------
        parts : list of torch.Tensor
            Branch outputs, each of shape ``[n, dim]``.

        Returns
        -------
        torch.Tensor
            Merged features of shape ``[n, dim]``.
        """
        return self.head(torch.cat(parts, dim=-1))


class MarkingEmbedding(nn.Module):
    """Embedding of the bucketed hop-distance marking of bag rows.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    max_dist : int, optional
        Distance cutoff used by the marking buckets. Default is 10.
    """

    def __init__(self, dim, max_dist=10):
        super().__init__()
        self.embed = nn.Embedding(max_dist + 2, dim)

    def forward(self, marking):
        """Embed the marking buckets.

        Parameters
        ----------
        marking : torch.Tensor
            Bucketed distances of shape ``[R]``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``[R, dim]``.
        """
        return self.embed(marking)


class TwoCellInit(nn.Module):
    """Initialize 2-cell features from their constituent nodes.

    A GIN update over the node-to-2-cell incidence, so every 2-cell
    aggregates the features of its boundary nodes (CIN-style
    initialization).

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_mlp_layers : int, optional
        Depth of the update MLP. Default is 2.
    """

    def __init__(self, dim, num_mlp_layers=2):
        super().__init__()
        self.conv = GINConv(homp_mlp(dim, num_mlp_layers), train_eps=True)

    def forward(self, x_0, x_2, inc02_pairs):
        """Compute the initial 2-cell features.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features of shape ``[n0, dim]``.
        x_2 : torch.Tensor
            Incoming 2-cell features of shape ``[n2, dim]``.
        inc02_pairs : torch.Tensor
            Node-2-cell incidence pairs of shape ``[2, I]``.

        Returns
        -------
        torch.Tensor
            Initialized 2-cell features of shape ``[n2, dim]``.
        """
        return self.conv(x=(x_0, x_2), edge_index=inc02_pairs)


class CINBlock(nn.Module):
    """One CIN-style higher-order message-passing block.

    Updates the three cochains sequentially (each update sees the ones
    already computed in this block, as in the reference): nodes from
    edge-mediated adjacency, edges from node incidence and 2-cell
    mediated adjacency, and 2-cells from edge incidence.

    Parameters
    ----------
    dim : int
        Embedding dimension of all ranks.
    num_mlp_layers : int, optional
        Depth of the convolution MLPs. Default is 2.
    max_rank : int, optional
        Highest rank updated by the block (0, 1, or 2). Default is 2.
    """

    def __init__(self, dim, num_mlp_layers=2, max_rank=2):
        super().__init__()
        if max_rank not in (0, 1, 2):
            raise ValueError("max_rank must be 0, 1, or 2")
        self.max_rank = max_rank
        self.node_conv = BridgeGIN(homp_mlp(dim, num_mlp_layers), edge_dim=dim)
        self.node_head = ConcatHead(1, dim)
        if max_rank >= 1:
            self.edge_inc_conv = GINConv(
                homp_mlp(dim, num_mlp_layers), train_eps=True
            )
            self.edge_adj_conv = BridgeGIN(
                homp_mlp(dim, num_mlp_layers), edge_dim=dim
            )
            self.edge_head = ConcatHead(2, dim)
        if max_rank >= 2:
            self.cell_conv = GINConv(
                homp_mlp(dim, num_mlp_layers), train_eps=True
            )
            self.cell_head = ConcatHead(1, dim)

    def forward(self, x_0, x_1, x_2, structures, a12_gate=None):
        """Run the block.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features ``[n0, dim]``.
        x_1 : torch.Tensor
            Edge features ``[n1, dim]``.
        x_2 : torch.Tensor
            2-cell features ``[n2, dim]``.
        structures : SMCNStructures
            Batch connectivity structures.
        a12_gate : torch.Tensor, optional
            Per-pair gate on the 2-cell-mediated edge adjacency (used by
            the learned-lifting variant). Default is None.

        Returns
        -------
        tuple of torch.Tensor
            Updated ``(x_0, x_1, x_2)``.
        """
        s = structures
        x_0 = self.node_head(
            [self.node_conv(x_0, s.a01_pairs, edge_attr=x_1[s.a01_bridge])]
        )
        if self.max_rank >= 1:
            x_1 = self.edge_head(
                [
                    self.edge_inc_conv(x=(x_0, x_1), edge_index=s.inc01_pairs),
                    self.edge_adj_conv(
                        x_1,
                        s.a12_pairs,
                        edge_attr=x_2[s.a12_bridge],
                        edge_weight=a12_gate,
                    ),
                ]
            )
        if self.max_rank >= 2:
            x_2 = self.cell_head(
                [self.cell_conv(x=(x_1, x_2), edge_index=s.inc12_pairs)]
            )
        return x_0, x_1, x_2


class BagInit(nn.Module):
    """Initialize the subcomplex-bag features.

    Every bag row ``(u, e)`` concatenates the features of node ``u``,
    edge ``e``, and the embedded distance marking, merged by a linear
    head.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    max_dist : int, optional
        Marking distance cutoff. Default is 10.
    """

    def __init__(self, dim, max_dist=10):
        super().__init__()
        self.marking = MarkingEmbedding(dim, max_dist)
        self.head = ConcatHead(3, dim)

    def forward(self, x_0, x_1, structures):
        """Build the initial bag features.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features ``[n0, dim]``.
        x_1 : torch.Tensor
            Edge features ``[n1, dim]``.
        structures : SMCNStructures
            Batch structures with the bag indices.

        Returns
        -------
        torch.Tensor
            Bag features of shape ``[R, dim]``.
        """
        s = structures
        return self.head(
            [
                x_0[s.bag_low_index],
                x_1[s.bag_high_index],
                self.marking(s.bag_marking),
            ]
        )


class SCLLayer(nn.Module):
    """One subcomplex (SCL) layer.

    The bag features are updated by the sum of three branches: a GIN
    over the marked-row broadcast edges, a bridge-aware GIN over the
    replicated node adjacency (bridge features taken from the edge
    cochain), and a re-embedded distance marking.

    Parameters
    ----------
    in_dim : int
        Input bag-feature dimension.
    hidden_dim : int
        Hidden and output dimension.
    edge_dim : int
        Dimension of the (frozen) edge features used as bridges.
    max_dist : int, optional
        Marking distance cutoff. Default is 10.
    """

    def __init__(self, in_dim, hidden_dim, edge_dim, max_dist=10):
        super().__init__()
        self.inc_conv = GINConv(scl_mlp(in_dim, hidden_dim), train_eps=True)
        self.low_conv = BridgeGIN(
            scl_mlp(in_dim, hidden_dim), edge_dim=edge_dim
        )
        self.marking = MarkingEmbedding(hidden_dim, max_dist)

    def forward(self, x_bag, x_1, structures):
        """Run the layer.

        Parameters
        ----------
        x_bag : torch.Tensor
            Bag features of shape ``[R, in_dim]``.
        x_1 : torch.Tensor
            Edge features used as bridge attributes, ``[n1, edge_dim]``.
        structures : SMCNStructures
            Batch structures.

        Returns
        -------
        torch.Tensor
            Updated bag features of shape ``[R, hidden_dim]``.
        """
        s = structures
        out = self.inc_conv(x=x_bag, edge_index=s.bag_inc_pairs)
        out = out + self.low_conv(
            x_bag,
            s.bag_low_adj_pairs,
            edge_attr=x_1[s.bag_low_adj_bridge],
        )
        return out + self.marking(s.bag_marking)


class BagPool(nn.Module):
    """Sum-pool the bag features back to nodes and edges."""

    def forward(self, x_bag, structures, num_nodes, num_edges):
        """Pool the bag.

        Parameters
        ----------
        x_bag : torch.Tensor
            Bag features of shape ``[R, dim]``.
        structures : SMCNStructures
            Batch structures with the bag indices.
        num_nodes : int
            Total number of nodes in the batch.
        num_edges : int
            Total number of edges in the batch.

        Returns
        -------
        x_0 : torch.Tensor
            Node features of shape ``[num_nodes, dim]``.
        x_1 : torch.Tensor
            Edge features of shape ``[num_edges, dim]``.
        """
        s = structures
        dim = x_bag.size(1)
        x_0 = x_bag.new_zeros(num_nodes, dim).index_add_(
            0, s.bag_low_index, x_bag
        )
        x_1 = x_bag.new_zeros(num_edges, dim).index_add_(
            0, s.bag_high_index, x_bag
        )
        return x_0, x_1
