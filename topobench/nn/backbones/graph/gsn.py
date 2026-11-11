"""
Graph Substructure Network (GSN) GIN+VN message-passing layer and model.

Implements the GIN-with-virtual-node variant of the GSN family from Bouritsas
et al. (2022), in which precomputed substructure (orbit) counts are injected
into the aggregation as node-level structural encodings in the additive GSN-v
form.

- `GSNGINVirtualNodeLayerV` is a single GIN+VN message-passing layer.
- `GSNGINVirtualNodeModel` stacks these layers, projecting the node (and
  optional edge) features into a shared embedding space and maintaining a
  per-graph virtual node.

The module-level helpers `mlp_dimension_builder` and `mlp_builder` assemble the
per-layer MLPs used by these layers.
"""

from typing import Literal

import torch
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.typing import OptTensor, Tensor


def mlp_dimension_builder(layer_dims: list[int]) -> list[tuple[int, int]]:
    """
    Turn a flat MLP architecture into per-layer (in, out) channel pairs.

    Converts a list of layer dimensions (input dimension, hidden widths,
    output dimension) into the list of ``(in_channels, out_channels)``
    tuples describing each consecutive linear layer.

    Parameters
    ----------
    layer_dims : list of int
        Layer dimensions, ordered from input to output. Must contain at
        least two entries (the input and output dimension).

    Returns
    -------
    list of (int, int)
        One ``(in_channels, out_channels)`` tuple per linear layer, i.e.
        ``len(layer_dims) - 1`` tuples.
    """
    return [
        (layer_dims[j], layer_dims[j + 1]) for j in range(len(layer_dims) - 1)
    ]


def mlp_builder(
    layer_dims: list[tuple[int, int]],
    activation_fn=torch.nn.ReLU,
    batch_norm: bool = False,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """
    Build a sequential MLP from per-layer (in, out) channel pairs.

    Stacks one ``torch.nn.Linear`` per entry in ``layer_dims``, inserting
    (optionally) batch normalization, an activation function, and
    (optionally) dropout between consecutive linear layers. Following the
    usual GIN convention, batch normalization is placed after the linear
    layer and before the activation, and dropout after the activation. None
    of these are appended after the final linear layer, so the MLP output is
    unbounded and carries the raw layer statistics.

    Parameters
    ----------
    layer_dims : list of (int, int)
        One ``(in_channels, out_channels)`` tuple per linear layer, as
        produced by `mlp_dimension_builder`.
    activation_fn : type of torch.nn.Module, optional
        Activation module class instantiated between consecutive linear
        layers. Default is ``torch.nn.ReLU``.
    batch_norm : bool, optional
        If True, insert a ``torch.nn.BatchNorm1d`` after every non-final
        linear layer (before its activation). Default is False.
    dropout : float, optional
        Dropout probability applied after every non-final activation. If 0
        (default), no dropout layers are added.

    Returns
    -------
    torch.nn.Sequential
        The assembled MLP, with batch norm / activations / dropout between
        (but not after) the linear layers.
    """
    model = torch.nn.Sequential()
    for lidx, (ic, oc) in enumerate(layer_dims):
        model.append(torch.nn.Linear(ic, oc))

        # check if we are in the last layer already, then dont append the
        # batch norm / activation / dropout block
        if lidx < len(layer_dims) - 1:
            if batch_norm:
                model.append(torch.nn.BatchNorm1d(oc))
            model.append(activation_fn())
            if dropout > 0.0:
                model.append(torch.nn.Dropout(dropout))

    return model


class GSNGINVirtualNodeLayerV(MessagePassing):
    """
    Single GIN+VN message-passing layer (node mode) following Eq. (11).

    Computes ``h_v' = UP(h~_v + sum_u sigma(h~_u + e'_{v,u}))`` for a node
    representation ``h~`` that is expected to already include the virtual-node
    and structural contributions (``h~_v = h_v + G + W_V x_v``), which are
    formed by the enclosing :class:`GSNGINVirtualNodeModel`. The self term is
    added explicitly (``x + propagate(...)``); messages apply the activation to
    the neighbour features plus the (already-embedded) edge features.

    Parameters
    ----------
    out_channels : int
        Dimensionality of the layer output (the ``UP`` MLP output width).
    common_embedding_dim : int
        Shared embedding dimensionality ``d_embed`` of the incoming node
        representations ``h~``.
    edge_dim : int, optional
        Dimensionality of the incoming edge embeddings. Default is 0.
    activation_fn : type of torch.nn.Module, optional
        Activation class applied inside the message function. Default is
        ``torch.nn.ReLU``.
    batch_norm : bool, optional
        If True, apply batch normalization on the hidden layer of the update
        MLP. Default is False.
    dropout : float, optional
        Dropout probability applied on the hidden layer of the update MLP.
        Default is 0.0 (no dropout).

    Attributes
    ----------
    UP : torch.nn.Sequential
        Update MLP (2-layer by default) applied after the self-plus-neighbour
        aggregation.
    activation_fn : torch.nn.Module
        Instantiated activation used in :meth:`message`.
    """

    def __init__(
        self,
        out_channels: int,
        common_embedding_dim: int,
        edge_dim: int = 0,
        activation_fn=torch.nn.ReLU,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="sum")

        self.out_channels = out_channels
        self.d_embed = common_embedding_dim
        self.edge_dim = edge_dim
        self.activation_fn = activation_fn()
        self.batch_norm = batch_norm
        self.dropout = dropout

        # 2-layer MLP update (hidden width = embedding dim) to keep the
        # update non-linear, consistent with the other GSN layers
        self.UP = mlp_builder(
            mlp_dimension_builder(
                [self.d_embed, self.d_embed, self.out_channels]
            ),
            batch_norm=self.batch_norm,
            dropout=self.dropout,
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute updated node representations from pre-augmented features.

        Parameters
        ----------
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity in COO format.
        x : torch.Tensor of shape (N, common_embedding_dim)
            Node representations ``h~`` that already include the virtual-node
            and structural contributions.
        edge_attr : torch.Tensor of shape (E, edge_dim) or None, optional
            Edge embeddings added to neighbour features in the message.
            Default is None.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Updated node representations.
        """
        # we expect the node embeddings to already contain the addition of big G
        # i.e. this method already receives tilde(h)_v^t = h_v^t + G^t

        out = x + self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr
        )

        return self.UP(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor = None):
        """
        Construct messages from source nodes and edge embeddings.

        Returns ``sigma(h~_u + e'_{v,u})``, or ``sigma(h~_u)`` when no edge
        embeddings are provided.

        Parameters
        ----------
        x_j : torch.Tensor of shape (E, common_embedding_dim)
            Source node representations for each edge.
        edge_attr : torch.Tensor of shape (E, edge_dim) or None, optional
            Edge embeddings added to the source features before the
            activation. Default is None.

        Returns
        -------
        torch.Tensor of shape (E, common_embedding_dim)
            Per-edge messages after the activation.
        """
        if edge_attr is None:
            return self.activation_fn(x_j)
        return self.activation_fn(x_j + edge_attr)


class GSNGINVirtualNodeModel(torch.nn.Module):
    """
    Multi-layer GIN+VN model with additive GSN structural encodings.

    Implements the GIN-with-virtual-node scheme of Eqs. (10)/(11) from
    Bouritsas et al. (2022), extended with structural identifiers in the GSN-v
    (node) additive form. Initial node (and edge) features are linearly
    projected into a shared embedding space ``d_embed``. At every layer the
    per-node representation is augmented as ``h~_v = h_v + G + W_V^t x_v`` (the
    virtual node ``G`` broadcast by graph plus a per-layer projection of the
    structural counts), the virtual node is updated from the pre-update
    aggregation ``G' = MLP(G + sum_u h~_u)``, and a
    :class:`GSNGINVirtualNodeLayerV` produces the next node representation.

    Only node mode is currently implemented.

    Parameters
    ----------
    mode : {'node', 'edge'}
        Structural-encoding mode. Only ``'node'`` is implemented; ``'edge'``
        raises ``NotImplementedError``.
    num_layers : int
        Number of message-passing layers.
    in_channels : int or list of int
        Dimensionality of input node features. A single-element list (as
        produced by the ``infer_in_channels`` config resolver, e.g. ``[7]``)
        is accepted and unwrapped to its scalar element.
    out_channels : int
        Dimensionality of the final per-node output.
    gsn_channels : int
        Dimensionality of the GSN structural encodings.
    common_embedding_dim : int or None, optional
        Shared embedding dimensionality ``d_embed``. If None, ``in_channels``
        is used. Default is None.
    edge_dim : int, optional
        Dimensionality of the original edge features. If greater than 0,
        ``data.edge_attr`` is projected into ``d_embed`` and fed to every
        layer. Default is 0.
    gsn_kword : str, optional
        Base name under which a caller/wrapper can locate the GSN encodings on
        a batch object; stored as ``self.gsn_kword = f"{mode}_{gsn_kword}"`` but
        not used inside ``forward`` (which receives the encodings as a tensor).
        Default is ``"gsn_encodings"``.
    G_updater_architecture : list of int or None, optional
        Hidden layer widths for each per-layer virtual-node update MLP. If
        None, a single hidden layer of width ``d_embed`` is used, giving a
        2-layer MLP. Default is None.
    batch_norm : bool, optional
        If True, apply batch normalization inside every layer's update MLP
        and virtual-node update MLP, and to the intermediate node
        representations between message-passing layers. Default is False.
    dropout : float, optional
        Dropout probability applied inside every layer's update MLP and
        virtual-node update MLP, and to the intermediate node representations
        between message-passing layers. Default is 0.0 (no dropout).

    Attributes
    ----------
    initial_node_projector : torch.nn.Linear
        Projection ``W_h^0`` of input node features into ``d_embed``.
    initial_edge_projector : torch.nn.Linear or None
        Projection of input edge features into ``d_embed`` (None if
        ``edge_dim == 0``).
    _gsn_encoders : torch.nn.ModuleList
        Per-layer structural-count projections ``W_V^t``.
    _layers : torch.nn.ModuleList
        Per-layer :class:`GSNGINVirtualNodeLayerV` modules.
    _between_layer_norms : torch.nn.ModuleList
        Per-layer batch-norm modules applied to the intermediate node
        representations (identity modules for the final layer and whenever
        ``batch_norm`` is False).
    G_updater_MLPs : torch.nn.ModuleList
        Per-layer virtual-node update MLPs (``num_layers - 1`` of them).

    Raises
    ------
    NotImplementedError
        If ``mode`` is not ``'node'``.
    """

    def __init__(
        self,
        mode: Literal["node", "edge"],
        num_layers: int,
        in_channels: int,
        out_channels: int,
        gsn_channels: int,
        common_embedding_dim: int | None = None,
        edge_dim: int = 0,
        gsn_kword: str = "gsn_encodings",
        G_updater_architecture: list[int] | None = None,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # `infer_in_channels` (used by the model config) passes the node-feature
        # dimension as a single-element list, e.g. [7]; accept either that or a
        # bare int and normalize to an int for the projection layers below.
        if not isinstance(in_channels, int):
            if len(in_channels) != 1:
                raise ValueError(
                    "GSNGINVirtualNodeModel expects a single node-feature "
                    f"dimension, got in_channels={in_channels!r}."
                )
            in_channels = int(in_channels[0])

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.node_mode = mode == "node"  # boolean flag
        self.gsn_kword = f"{mode}_{gsn_kword}"
        self.in_gsn_channels = gsn_channels
        self.edge_dim = edge_dim
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.d_embed: int = (
            self.in_channels
            if common_embedding_dim is None
            else common_embedding_dim
        )

        self.initial_node_projector = torch.nn.Linear(
            self.in_channels, self.d_embed
        )
        self.initial_edge_projector = (
            torch.nn.Linear(self.edge_dim, self.d_embed)
            if self.edge_dim > 0
            else None
        )

        if not self.node_mode:
            raise NotImplementedError("Edge mode not yet implemented!")
        self._layer_cls = GSNGINVirtualNodeLayerV

        # default to a single hidden layer (i.e. a 3-layer MLP) when unspecified
        arch = (
            [self.d_embed]
            if G_updater_architecture is None
            else G_updater_architecture
        )
        arch = [self.d_embed] + arch + [self.d_embed]
        self.G_updater_MLPs = torch.nn.ModuleList(
            [
                mlp_builder(
                    mlp_dimension_builder(arch),
                    batch_norm=self.batch_norm,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers - 1)
            ]
        )

        _layers: list[GSNGINVirtualNodeLayerV] = []
        _gsn_encoders: list[torch.nn.Linear] = []
        _between_layer_norms: list[torch.nn.Module] = []

        for layer_index in range(self.num_layers):
            oc: int = -1

            _gsn_encoders.append(
                torch.nn.Linear(self.in_gsn_channels, self.d_embed)
            )

            if layer_index == 0:
                oc = self.d_embed if self.num_layers > 1 else self.out_channels
            else:
                if layer_index == self.num_layers - 1:
                    oc = self.out_channels
                else:
                    oc = self.d_embed

            _layers.append(
                self._layer_cls(
                    out_channels=oc,
                    common_embedding_dim=self.d_embed,
                    edge_dim=edge_dim,
                    batch_norm=self.batch_norm,
                    dropout=self.dropout,
                )
            )

            # batch-norm the intermediate representations between layers; the
            # final layer output is left raw for the downstream readout / head
            is_last = layer_index == self.num_layers - 1
            _between_layer_norms.append(
                torch.nn.BatchNorm1d(oc)
                if (self.batch_norm and not is_last)
                else torch.nn.Identity()
            )

        self._gsn_encoders = torch.nn.ModuleList(_gsn_encoders)
        self._layers = torch.nn.ModuleList(_layers)
        self._between_layer_norms = torch.nn.ModuleList(_between_layer_norms)
        # dropout on the intermediate node representations between layers
        self._between_layer_dropout = torch.nn.Dropout(self.dropout)

    def forward(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        gsn_embeddings: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run a full GIN+VN forward pass over all layers.

        The argument order mirrors the per-layer ``forward``; pulling the right
        fields off a ``Data``/batch object is the caller's (e.g. the wrapper's)
        job.

        Parameters
        ----------
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity in COO format.
        x : torch.Tensor of shape (N, in_channels)
            Input node features.
        gsn_embeddings : torch.Tensor of shape (N, gsn_channels)
            Per-node GSN structural encodings.
        edge_attr : torch.Tensor of shape (E, edge_dim) or None, optional
            Original edge features; required when ``edge_dim > 0`` and must be
            absent otherwise. Default is None.
        batch : torch.Tensor of shape (N,) or None, optional
            Node-to-graph assignment used to keep the virtual node per-graph.
            If None, all nodes are treated as a single graph. Default is None.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Node representations after all message-passing layers.

        Raises
        ------
        RuntimeError
            If ``edge_attr`` is present while ``edge_dim == 0``.
        """
        device = x.device
        dtype = x.dtype

        num_nodes = x.size(0)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

        if (edge_attr is not None) and (self.edge_dim == 0):
            edge_attr = None

        num_graphs: int = int(batch.max().item()) + 1

        # we initialize the virtual node matrix G
        small_G = torch.zeros(
            num_graphs, self.d_embed, device=device, dtype=dtype
        )

        # need to project node & edge features initially
        x = self.initial_node_projector(x)

        edge_attr = (
            self.initial_edge_projector(edge_attr)
            if edge_attr is not None
            else None
        )

        for lidx, layer in enumerate(self._layers):
            big_G = small_G[batch]

            x = x + self._gsn_encoders[lidx](gsn_embeddings) + big_G

            if lidx < len(self._layers) - 1:
                # now we need to update the virtual nodes:
                small_G = self.G_updater_MLPs[lidx](
                    small_G + global_add_pool(x, batch=batch)
                )

            x = layer(edge_index=edge_index, x=x, edge_attr=edge_attr)

            # normalize / drop the intermediate representations; both are
            # no-ops on the final layer (Identity norm, and dropout skipped)
            # so the model output is left raw for the downstream head
            if lidx < len(self._layers) - 1:
                x = self._between_layer_norms[lidx](x)
                x = self._between_layer_dropout(x)

        return x
