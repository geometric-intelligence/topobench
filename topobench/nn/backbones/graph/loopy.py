"""Loopy backbone: message passing over r-neighbourhood paths.

Port of the `"loopy" <https://github.com/RPaolino/loopy>`_ layers to the
TopoBench node-to-node backbone contract. The r-neighbourhood paths are
precomputed by the ``RNeighbourhood`` transform and handed in by ``LoopyWrapper`` as the
``loopy_n`` / ``loopy_a`` dictionaries.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter

ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def get_activation(name):
    """Resolve an activation module from its name.

    Parameters
    ----------
    name : str
        Name of the activation, one of the keys of ``ACTIVATIONS``.

    Returns
    -------
    torch.nn.Module
        A freshly built activation module.
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Available options: {sorted(ACTIVATIONS)}."
        )
    return ACTIVATIONS[name]()


def _path_propagate(x):
    """Sum each path node's features with those of its path neighbours.

    In a path every node is linked to the previous and next node, so
    aggregating over the path neighbourhood is a convolution with the
    kernel ``[1, 0, 1]`` (zero padded at the two ends) along the path
    dimension.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(path_length, num_paths, channels)``.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape holding, at each position, the sum of the
        two path-neighbours' features.
    """
    out = torch.zeros_like(x)
    out[:-1] = out[:-1] + x[1:]
    out[1:] = out[1:] + x[:-1]
    return out


class MLP(nn.Module):
    """Two or more layer MLP with an optional normalization.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    num_layers : int, optional
        Total number of linear layers.
    nonlinearity : str, optional
        Name of the activation applied between layers.
    norm : str, optional
        Name of a ``torch.nn`` normalization class applied between layers,
        e.g. ``"BatchNorm1d"`` or ``"Identity"``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        nonlinearity="relu",
        norm="Identity",
    ):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels)])
        for _ in range(num_layers - 1):
            self.lins.append(nn.Linear(out_channels, out_channels))
        self.norm = getattr(nn, norm)(out_channels)
        self.act = get_activation(nonlinearity)

    def reset_parameters(self):
        """Reset the parameters of the linear layers and the norm."""
        for lin in self.lins:
            lin.reset_parameters()
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    def forward(self, x):
        """Apply the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension is ``in_channels``.

        Returns
        -------
        torch.Tensor
            Output tensor whose last dimension is ``out_channels``.
        """
        x = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.norm(x)
            x = self.act(x)
            x = lin(x)
        return x


class CustomGINConv(nn.Module):
    """GIN-style convolution along a path, aware of hop distances.

    Parameters
    ----------
    mlp : torch.nn.Module
        Update network applied after aggregation.
    in_channels : int
        Number of input features.
    num_embeddings : int
        Number of distinct hop distances to embed.
    train_eps : bool, optional
        Whether ``eps`` is learnable.
    """

    def __init__(self, mlp, in_channels, num_embeddings, train_eps=True):
        super().__init__()
        self.mlp = mlp
        self.eps = nn.Parameter(torch.ones(1), requires_grad=train_eps)
        self.embedding = nn.Embedding(num_embeddings, in_channels)
        self.transform = nn.Linear(2 * in_channels, in_channels)

    def reset_parameters(self):
        """Reset the parameters of the submodules."""
        self.mlp.reset_parameters()
        self.embedding.reset_parameters()
        self.transform.reset_parameters()

    def forward(self, x, atomic_type):
        """Aggregate a path and return one vector per path.

        Parameters
        ----------
        x : torch.Tensor
            Path node features of shape ``(path_length, num_paths,
            in_channels)``.
        atomic_type : torch.Tensor
            Hop distance of each path node to the centre, of shape
            ``(path_length, num_paths)``.

        Returns
        -------
        torch.Tensor
            One embedding per path, of shape ``(num_paths, out_channels)``.
        """
        out = self.transform(
            torch.cat([x, self.embedding(atomic_type)], dim=-1)
        )
        out = _path_propagate(out)
        out = self.mlp((1 + self.eps) * x + out)
        return out.sum(0)


class LoopyLayer(nn.Module):
    """One loopy layer aggregating all r-neighbourhood orders.

    The layer requires ``in_channels == out_channels`` because the order-0
    (direct neighbour) contribution keeps the input width while the higher
    order contributions are summed with it.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features (equal to ``in_channels``).
    r : int
        Maximal neighbourhood order.
    nonlinearity : str, optional
        Activation used by the final MLP.
    norm : str, optional
        Normalization used by the final MLP.
    shared : bool, optional
        Whether a single convolution is shared across the orders instead of
        one per order.
    path_chunk_size : int, optional
        Number of paths processed per gradient-checkpointed chunk.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        r,
        nonlinearity="relu",
        norm="BatchNorm1d",
        shared=False,
        path_chunk_size=8192,
    ):
        super().__init__()
        self.r = r
        self.shared = shared
        self.path_chunk_size = path_chunk_size
        self.eps = nn.Parameter(torch.zeros(1))
        self.r_eps = nn.Parameter(torch.zeros(r + 1))
        num_embeddings = r + 2
        num_convs = 1 if shared else r
        self.convs = nn.ModuleList(
            [
                CustomGINConv(
                    MLP(in_channels, out_channels, num_layers=2),
                    in_channels=in_channels,
                    num_embeddings=num_embeddings,
                )
                for _ in range(num_convs)
            ]
        )
        self.conv_final = MLP(
            in_channels,
            out_channels,
            num_layers=2,
            nonlinearity=nonlinearity,
            norm=norm,
        )

    def forward(self, x, loopy_n, loopy_a, num_nodes):
        """Aggregate every neighbourhood order onto the nodes.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(num_nodes, channels)``.
        loopy_n : dict[int, torch.Tensor]
            Per-order node-index paths of shape ``(L + 2, num_paths)`` with
            global indices.
        loopy_a : dict[int, torch.Tensor]
            Per-order hop distances of shape ``(L + 2, num_paths)``.
        num_nodes : int
            Number of nodes in the batch.

        Returns
        -------
        torch.Tensor
            Updated node features of shape ``(num_nodes, channels)``.
        """
        x = x.float()
        r_contribution = 0
        for order in range(self.r + 1):
            paths = loopy_n[order]
            num_paths = paths.shape[1]
            if num_paths == 0:
                continue
            conv = (
                None
                if order == 0
                else (self.convs[0] if self.shared else self.convs[order - 1])
            )
            centres = paths[0]
            node_idx = paths[1:]
            hops = loopy_a[order][1:]
            # Process the paths in chunks and scatter each chunk onto its
            # centre nodes. The per-chunk work is gradient-checkpointed while
            # training, so activations are recomputed in the backward pass
            # rather than stored: this bounds the memory to one chunk, which
            # dense graphs (with very many paths) would otherwise blow past.
            acc = x.new_zeros(num_nodes, x.shape[1])
            for start in range(0, num_paths, self.path_chunk_size):
                end = min(start + self.path_chunk_size, num_paths)
                contribution = self._process_chunk(
                    conv, x, node_idx[:, start:end], hops[:, start:end]
                )
                acc = acc + scatter(
                    contribution,
                    centres[start:end],
                    dim=0,
                    dim_size=num_nodes,
                    reduce="sum",
                )
            r_contribution = r_contribution + (1 + self.r_eps[order]) * acc
        return self.conv_final((1 + self.eps) * x + r_contribution)

    def _process_chunk(self, conv, x, node_idx, hops):
        """Gather and convolve one chunk of paths.

        Parameters
        ----------
        conv : torch.nn.Module or None
            Convolution for this order, or ``None`` for the order-0 (direct
            neighbour) chunk.
        x : torch.Tensor
            Node features of shape ``(num_nodes, channels)``.
        node_idx : torch.Tensor
            Global indices of the non-centre path nodes, of shape
            ``(L + 1, chunk)``.
        hops : torch.Tensor
            Hop distances of the non-centre path nodes, of shape
            ``(L + 1, chunk)``.

        Returns
        -------
        torch.Tensor
            One embedding per path in the chunk, of shape
            ``(chunk, channels)``.
        """

        def run(node_features):
            """Gather node features and apply the convolution.

            Parameters
            ----------
            node_features : torch.Tensor
                Node features of shape ``(num_nodes, channels)``.

            Returns
            -------
            torch.Tensor
                One embedding per path in the chunk.
            """
            gathered = node_features[node_idx]
            if conv is None:
                return gathered.squeeze(0)
            return conv(gathered, hops)

        if self.training and x.requires_grad:
            return checkpoint(run, x, use_reentrant=False)
        return run(x)


class Loopy(nn.Module):
    """Loopy backbone stacking several loopy layers.

    Maps node features to node embeddings. The order-``L`` path tensors are
    supplied by ``LoopyWrapper`` and must cover ``L`` in ``0 .. r``, with
    ``r`` matching the ``RNeighbourhood`` transform.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden features (kept constant across the layers).
    num_layers : int, optional
        Number of loopy layers.
    r : int, optional
        Maximal neighbourhood order; must match the transform.
    dropout : float, optional
        Dropout probability applied between layers.
    nonlinearity : str, optional
        Activation used by the layer MLPs.
    norm : str, optional
        Normalization used by the final MLP of each layer.
    shared : bool, optional
        Whether each layer shares one convolution across the orders.
    path_chunk_size : int, optional
        Number of paths processed per gradient-checkpointed chunk.
    **kwargs : dict, optional
        Ignored, kept for compatibility with the TopoBench model
        instantiation.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        r=2,
        dropout=0.0,
        nonlinearity="relu",
        norm="BatchNorm1d",
        shared=False,
        path_chunk_size=8192,
        **kwargs,
    ):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [
                LoopyLayer(
                    hidden_channels,
                    hidden_channels,
                    r=r,
                    nonlinearity=nonlinearity,
                    norm=norm,
                    shared=shared,
                    path_chunk_size=path_chunk_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        edge_index,
        batch=None,
        loopy_n=None,
        loopy_a=None,
        edge_weight=None,
        **kwargs,
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Edge indices; unused, kept for signature compatibility.
        batch : torch.Tensor, optional
            Batch assignment vector; unused, pooling is delegated to the
            readout.
        loopy_n : dict[int, torch.Tensor]
            Per-order node-index paths of shape ``(L + 2, num_paths)`` with
            global indices.
        loopy_a : dict[int, torch.Tensor]
            Per-order hop distances of shape ``(L + 2, num_paths)``.
        edge_weight : torch.Tensor, optional
            Unused, kept for signature compatibility.
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``(num_nodes, hidden_channels)``.
        """
        x = self.encoder(x)
        num_nodes = x.shape[0]
        for layer in self.layers:
            x = layer(x, loopy_n, loopy_a, num_nodes)
            x = self.dropout(x)
        return x
