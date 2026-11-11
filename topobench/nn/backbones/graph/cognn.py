"""Co-GNN backbone from "Cooperative Graph Neural Networks" (ICML 2024).

Paper: Finkelshtein, Huang, Bronstein, Ceylan,
"Cooperative Graph Neural Networks", ICML 2024.
https://arxiv.org/abs/2310.01267

Original implementation: https://github.com/benfinkelshtein/CoGNN
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
}


def weighted_aggregate(x, edge_index, num_nodes, edge_weight=None, aggr="sum"):
    r"""Aggregate (optionally weighted) neighbor messages.

    Computes :math:`\bigoplus_{u \in \mathcal{N}(v)} w_{u,v}
    \mathbf{h}_u` for every node :math:`v`, where messages flow from
    source ``edge_index[0]`` to target ``edge_index[1]``. This is the
    weighted message-passing primitive shared by all Co-GNN layers
    (`Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_,
    Eq. (2)): with the action-derived binary weights
    :math:`w_{u,v} \in \{0, 1\}`, only edges whose source broadcasts and
    whose target listens contribute.

    Parameters
    ----------
    x : torch.Tensor
        Node features of shape [num_nodes, channels].
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    num_nodes : int
        Number of nodes.
    edge_weight : torch.Tensor, optional
        Edge weights of shape [num_edges] (default: None).
    aggr : str, optional
        Aggregation scheme, "sum" or "mean" (default: "sum").

    Returns
    -------
    torch.Tensor
        Aggregated messages of shape [num_nodes, channels].

    Notes
    -----
    For ``aggr="mean"`` the weighted messages are averaged over the *full*
    in-degree of each node, i.e. edges gated to weight 0 still count in
    the denominator. This deliberately reproduces the original Co-GNN
    implementation (``models/layers.py``: messages are scaled by
    ``edge_weight`` inside ``message()`` and then reduced with PyG's
    standard ``aggr='mean'``), which is the exact MeanGNN (:math:`\mu`)
    semantics used to produce the results reported in the paper.
    """
    messages = x[edge_index[0]]
    if edge_weight is not None:
        messages = edge_weight.view(-1, 1) * messages
    return scatter(
        messages, edge_index[1], dim=0, dim_size=num_nodes, reduce=aggr
    )


class WeightedGNNConv(nn.Module):
    r"""Edge-weighted SumGNN (:math:`\Sigma`) / MeanGNN (:math:`\mu`) layer.

    Implements the environment/action layer used by Co-GNN
    (`Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_,
    Section 4). The node update concatenates the node state with the
    (weighted) aggregation of its neighbors' states, followed by a linear
    transformation:

    .. math::
        \mathbf{h}_v' = \mathbf{W} \left[ \mathbf{h}_v \,\Vert\,
        \bigoplus_{u \in \mathcal{N}(v)} w_{u,v} \mathbf{h}_u \right],

    where :math:`\bigoplus` is a sum (SumGNN, :math:`\Sigma`) or a mean
    (MeanGNN, :math:`\mu`) and :math:`w_{u,v}` is the edge weight computed
    from the sampled actions (Eq. (2) of the paper reduces to this weighted
    form with :math:`w_{u,v} \in \{0, 1\}`).

    Follows ``WeightedGNNConv`` in the original implementation
    (``models/layers.py``).

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    out_channels : int
        Number of output features per node.
    aggr : str, optional
        Aggregation scheme, "sum" or "mean" (default: "sum").
    bias : bool, optional
        Whether the linear layer uses a bias term (default: True).
    """

    def __init__(self, in_channels, out_channels, aggr="sum", bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges] (default: None).

        Returns
        -------
        torch.Tensor
            Updated node features of shape [num_nodes, out_channels].
        """
        out = weighted_aggregate(
            x, edge_index, x.size(0), edge_weight, aggr=self.aggr
        )
        return self.lin(torch.cat((x, out), dim=-1))


class WeightedGCNConv(nn.Module):
    r"""Edge-weighted GCN (:math:`*`) layer used by Co-GNN.

    GCN layer (Kipf & Welling, 2017) that supports differentiable edge
    weights, used as an action/environment network component in Co-GNN
    (`Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_,
    Section 4). The action-derived edge weights :math:`w_{u,v}` enter the
    symmetric normalization computed by
    :func:`torch_geometric.nn.conv.gcn_conv.gcn_norm`.

    Follows ``WeightedGCNConv`` in the original implementation
    (``models/layers.py``); self-loops are handled by ``gcn_norm``.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    out_channels : int
        Number of output features per node.
    bias : bool, optional
        Whether the linear layer uses a bias term (default: True).
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges] (default: None).

        Returns
        -------
        torch.Tensor
            Updated node features of shape [num_nodes, out_channels].
        """
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            x.size(0),
            improved=False,
            add_self_loops=True,
            dtype=x.dtype,
        )
        out = weighted_aggregate(
            x, edge_index, x.size(0), edge_weight, aggr="sum"
        )
        return self.lin(out)


class WeightedGINConv(nn.Module):
    r"""Edge-weighted GIN (:math:`\epsilon`) layer used by Co-GNN.

    GIN layer (Xu et al., 2019) with support for edge weights, used as an
    action/environment network component in Co-GNN
    (`Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_,
    Section 4):

    .. math::
        \mathbf{h}_v' = \mathrm{MLP}\left((1 + \epsilon) \mathbf{h}_v +
        \sum_{u \in \mathcal{N}(v)} w_{u,v} \mathbf{h}_u \right).

    Follows ``WeightedGINConv`` in the original implementation
    (``models/layers.py``) with the default MLP factory of the reference
    code (Linear -> BatchNorm -> ReLU -> Linear).

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    out_channels : int
        Number of output features per node.
    bias : bool, optional
        Whether the linear layers use bias terms (default: True).
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * in_channels, bias=bias),
            nn.BatchNorm1d(2 * in_channels),
            nn.ReLU(),
            nn.Linear(2 * in_channels, out_channels, bias=bias),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges] (default: None).

        Returns
        -------
        torch.Tensor
            Updated node features of shape [num_nodes, out_channels].
        """
        out = weighted_aggregate(
            x, edge_index, x.size(0), edge_weight, aggr="sum"
        )
        return self.mlp((1 + self.eps) * x + out)


def build_conv_layer(conv_type, in_channels, out_channels):
    """Build a single Co-GNN convolution layer.

    Maps the layer names of the paper (Section 4 and Table 9 of
    `Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_)
    to their implementations: "sum_gnn" (:math:`\\Sigma`), "mean_gnn"
    (:math:`\\mu`), "gcn" (:math:`*`) and "gin" (:math:`\\epsilon`).

    Parameters
    ----------
    conv_type : str
        One of "sum_gnn", "mean_gnn", "gcn", "gin".
    in_channels : int
        Number of input features per node.
    out_channels : int
        Number of output features per node.

    Returns
    -------
    torch_geometric.nn.conv.MessagePassing
        The convolution layer.

    Raises
    ------
    ValueError
        If ``conv_type`` is not supported.
    """
    if conv_type == "sum_gnn":
        return WeightedGNNConv(in_channels, out_channels, aggr="sum")
    if conv_type == "mean_gnn":
        return WeightedGNNConv(in_channels, out_channels, aggr="mean")
    if conv_type == "gcn":
        return WeightedGCNConv(in_channels, out_channels)
    if conv_type == "gin":
        return WeightedGINConv(in_channels, out_channels)
    raise ValueError(f"Convolution type '{conv_type}' is not supported.")


class CoGNNActionNet(nn.Module):
    r"""Action network :math:`\pi` of Co-GNN.

    Lightweight GNN that predicts per-node action logits from the node
    state and its neighborhood, Eq. (1) of
    `Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_:

    .. math::
        \mathbf{p}_v^{(\ell)} = \pi \left( \mathbf{h}_v^{(\ell)},
        \{\!\!\{ \mathbf{h}_u^{(\ell)} \mid u \in \mathcal{N}_v \}\!\!\}
        \right).

    Each of the two binary decisions (listen / broadcast) is predicted by
    its own instance of this network, so the output has two logits per
    node (keep vs. drop). Follows ``ActionNet`` in the original
    implementation (``models/action.py``).

    Parameters
    ----------
    in_channels : int
        Number of input features per node (the environment state size).
    hidden_channels : int
        Number of hidden features per node.
    num_layers : int
        Number of convolution layers.
    conv_type : str, optional
        Convolution type, see :func:`build_conv_layer`
        (default: "mean_gnn").
    dropout : float, optional
        Dropout rate applied between layers (default: 0.0).
    act : str, optional
        Activation between layers, "relu" or "gelu" (default: "relu").
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        conv_type="mean_gnn",
        dropout=0.0,
        act="relu",
    ):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [2]
        self.convs = nn.ModuleList(
            [
                build_conv_layer(conv_type, d_in, d_out)
                for d_in, d_out in zip(dims[:-1], dims[1:], strict=True)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.act = ACTIVATIONS[act]

    def forward(self, x, edge_index, edge_weight=None):
        """Compute per-node action logits.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges] (default: None).

        Returns
        -------
        torch.Tensor
            Action logits of shape [num_nodes, 2] (keep vs. drop).
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.dropout(x)
            x = self.act(x)
        return self.convs[-1](x, edge_index, edge_weight=edge_weight)


class CoGNNTempSoftPlus(nn.Module):
    r"""Learnable Gumbel-softmax temperature module of Co-GNN.

    Predicts a per-node temperature :math:`\tau_v` for the Gumbel-softmax
    estimator from the node state (Appendix E.1 of
    `Finkelshtein et al., ICML 2024 <https://arxiv.org/abs/2310.01267>`_):

    .. math::
        \tau_v = \left( \mathrm{softplus}(\mathbf{w}^\top
        \mathbf{h}_v) + \tau_0 \right)^{-1}.

    Follows ``TempSoftPlus`` in the original implementation
    (``models/temp.py``) with its default linear temperature model.

    Parameters
    ----------
    hidden_channels : int
        Number of input features per node (the environment state size).
    tau0 : float, optional
        Temperature offset :math:`\tau_0` (default: 0.5).
    """

    def __init__(self, hidden_channels, tau0=0.5):
        super().__init__()
        self.lin = nn.Linear(hidden_channels, 1, bias=False)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = tau0

    def forward(self, x):
        """Compute per-node Gumbel-softmax temperatures.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, hidden_channels].

        Returns
        -------
        torch.Tensor
            Per-node temperatures of shape [num_nodes, 1].
        """
        temp = self.softplus(self.lin(x)) + self.tau0
        temp = temp.pow(-1)
        return temp.masked_fill(temp == float("inf"), 0.0)


class CoGNN(nn.Module):
    r"""Co-GNN backbone from "Cooperative Graph Neural Networks".

    In a Co-GNN (`Finkelshtein et al., ICML 2024
    <https://arxiv.org/abs/2310.01267>`_, Section 4) every node chooses at
    every layer one of four actions: Standard (S), Listen (L),
    Broadcast (B) or Isolate (I). The actions are predicted by action
    networks :math:`\pi` (Eq. (1)) and sampled with the straight-through
    Gumbel-softmax estimator; the environment network :math:`\eta` then
    performs message passing only over edges :math:`(u, v)` where
    :math:`u` broadcasts and :math:`v` listens (Eq. (2)):

    .. math::
        \mathbf{h}_v^{(\ell+1)} = \eta^{(\ell)} \left(
        \mathbf{h}_v^{(\ell)}, \{\!\!\{ \mathbf{h}_u^{(\ell)} \mid
        u \in \mathcal{N}_v,\ a_u^{(\ell)} \in \{S, B\} \}\!\!\} \right)
        \quad \text{if } a_v^{(\ell)} \in \{S, L\}.

    As in the original implementation, the four-way action is factorized
    into two binary decisions predicted by two separate action networks:
    an *in* network (listen vs. not) and an *out* network (broadcast vs.
    not); the four actions are recovered as S = (listen, broadcast),
    L = (listen, -), B = (-, broadcast), I = (-, -). Eq. (2) is then
    realized by giving each edge :math:`(u, v)` the binary weight

    .. math::
        w_{u,v} = p_v^{\mathrm{in}} \cdot p_u^{\mathrm{out}},

    where the straight-through estimator makes the hard samples
    :math:`p^{\mathrm{in}}, p^{\mathrm{out}} \in \{0, 1\}`
    differentiable. The paper denotes by Co-GNN(:math:`\pi, \eta`) the
    architecture with action network type :math:`\pi` and environment
    network type :math:`\eta`, e.g. Co-GNN(:math:`\mu, \mu`) uses
    MeanGNN layers for both.

    This backbone corresponds to the hidden part of ``CoGNN`` in the
    original implementation (``models/CoGNN.py``): the input encoder is a
    linear layer, and the task decoder/pooling is delegated to the
    TopoBench readout module.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Number of hidden features per node (environment state size).
    num_layers : int, optional
        Number of environment layers (default: 3).
    env_conv_type : str, optional
        Environment network layer type :math:`\eta`, one of "sum_gnn",
        "mean_gnn", "gcn", "gin" (default: "mean_gnn").
    act_conv_type : str, optional
        Action network layer type :math:`\pi`, one of "sum_gnn",
        "mean_gnn", "gcn", "gin" (default: "mean_gnn").
    act_num_layers : int, optional
        Number of layers of each action network (default: 1).
    act_hidden_channels : int, optional
        Hidden size of the action networks (default: 16).
    dropout : float, optional
        Dropout rate (default: 0.0).
    temp : float, optional
        Gumbel-softmax temperature :math:`\tau` used when ``learn_temp``
        is False (default: 0.01).
    learn_temp : bool, optional
        Whether to learn the temperature with
        :class:`CoGNNTempSoftPlus` (default: False).
    tau0 : float, optional
        Temperature offset of the learned temperature model
        (default: 0.5).
    layer_norm : bool, optional
        Whether to apply layer normalization to the hidden states
        (default: False).
    skip : bool, optional
        Whether to use residual connections between environment layers
        (default: False).
    act : str, optional
        Activation function, "relu" or "gelu" (default: "relu").
    **kwargs
        Additional (ignored) keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=3,
        env_conv_type="mean_gnn",
        act_conv_type="mean_gnn",
        act_num_layers=1,
        act_hidden_channels=16,
        dropout=0.0,
        temp=0.01,
        learn_temp=False,
        tau0=0.5,
        layer_norm=False,
        skip=False,
        act="relu",
        **kwargs,
    ):
        super().__init__()
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.skip = skip
        self.temp = temp
        self.learn_temp = learn_temp

        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.env_convs = nn.ModuleList(
            [
                build_conv_layer(
                    env_conv_type, hidden_channels, hidden_channels
                )
                for _ in range(num_layers)
            ]
        )
        self.in_act_net = CoGNNActionNet(
            hidden_channels,
            act_hidden_channels,
            act_num_layers,
            conv_type=act_conv_type,
            dropout=dropout,
            act=act,
        )
        self.out_act_net = CoGNNActionNet(
            hidden_channels,
            act_hidden_channels,
            act_num_layers,
            conv_type=act_conv_type,
            dropout=dropout,
            act=act,
        )
        if learn_temp:
            self.temp_model = CoGNNTempSoftPlus(hidden_channels, tau0=tau0)

        self.hidden_layer_norm = (
            nn.LayerNorm(hidden_channels) if layer_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.act = ACTIVATIONS[act]

    def create_edge_weight(self, edge_index, keep_in_prob, keep_out_prob):
        r"""Combine the sampled actions into differentiable edge weights.

        Realizes Eq. (2) of `Finkelshtein et al., ICML 2024
        <https://arxiv.org/abs/2310.01267>`_ as an edge weight
        :math:`w_{u,v} = p_v^{\mathrm{in}} \cdot p_u^{\mathrm{out}}`:
        an edge :math:`(u, v)` carries a message iff the target node
        :math:`v` listens and the source node :math:`u` broadcasts.
        Follows ``CoGNN.create_edge_weight`` in the original
        implementation (``models/CoGNN.py``).

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        keep_in_prob : torch.Tensor
            Per-node probability of listening, shape [num_nodes].
        keep_out_prob : torch.Tensor
            Per-node probability of broadcasting, shape [num_nodes].

        Returns
        -------
        torch.Tensor
            Edge weights of shape [num_edges].
        """
        u, v = edge_index
        return keep_in_prob[v] * keep_out_prob[u]

    def forward(self, x, edge_index, batch=None, edge_weight=None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        batch : torch.Tensor, optional
            Batch assignment vector of shape [num_nodes]; unused, kept
            for compatibility with the TopoBench ``GNNWrapper``
            (default: None).
        edge_weight : torch.Tensor, optional
            Input edge weights of shape [num_edges]; multiplied with the
            action-derived edge weights when provided (default: None).

        Returns
        -------
        torch.Tensor
            Node embeddings of shape [num_nodes, hidden_channels].
        """
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.act(x)

        for env_conv in self.env_convs:
            x = self.hidden_layer_norm(x)

            # Eq. (1): per-node action logits for the two binary
            # decisions (listen / broadcast).
            in_logits = self.in_act_net(x, edge_index)
            out_logits = self.out_act_net(x, edge_index)

            # Straight-through Gumbel-softmax sampling (Section 3).
            temp = self.temp_model(x) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(out_logits, tau=temp, hard=True)

            # Eq. (2) as differentiable edge weights.
            gate = self.create_edge_weight(
                edge_index, in_probs[:, 0], out_probs[:, 0]
            )
            if edge_weight is not None:
                gate = gate * edge_weight

            out = env_conv(x, edge_index, edge_weight=gate)
            out = self.dropout(out)
            out = self.act(out)

            x = x + out if self.skip else out

        return self.hidden_layer_norm(x)
