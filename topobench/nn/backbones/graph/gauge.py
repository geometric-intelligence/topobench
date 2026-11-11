"""Riemannian Graph Foundation Model with neural vector bundles.

This module implements the gauge-equivariant graph model described in
"Are Common Substructures Transferable: Riemannian Graph Foundation Model
with Neural Vector Bundles".

The model learns, for every node, a local orthonormal frame (a gauge) that
spans an ``r``-dimensional subspace of the ``d``-dimensional embedding space,
smooths those frames across the graph, and uses them to update the node
features. The building blocks correspond directly to the equations of the
paper:

- :class:`LocalCoordinatesLayer` -- equations (2)-(4).
- :class:`GatedFlatteningLayer` -- equations (5)-(8).
- :class:`NodeUpdateLayer` -- equations (9)-(10).
- :class:`GaugeLayer` -- one full message-passing step combining the above.
- :class:`GaugeModel` -- the full stack of gauge layers.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

activation_dict: dict[str, Callable] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


class MultiHeadLinear(nn.Module):
    """Per-head (per-subspace) linear layer.

    Applies ``r_dim`` independent linear maps, one per head, so that head ``h``
    transforms its own slice of the input with its own weight matrix and bias.
    The per-head weights are stored stacked in a single parameter and applied
    with a batched ``einsum``.

    Parameters
    ----------
    in_channels : int
        Number of input features per head.
    out_channels : int
        Number of output features per head.
    r_dim : int
        Number of heads (subspaces) ``r``.
    bias : bool, optional
        Whether each head uses a bias term (default: True).
    device : torch.device or str or None, optional
        Device on which to allocate the parameters (default: None).
    dtype : torch.dtype or None, optional
        Data type of the parameters (default: None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.r_dim = r_dim

        self.superW = nn.Parameter(
            torch.empty(
                self.r_dim,
                self.out_channels,
                self.in_channels,
                **factory_kwargs,
            )
        )

        if self.bias:
            self.superB = nn.Parameter(
                torch.empty(self.r_dim, self.out_channels, **factory_kwargs),
            )
        else:
            self.register_parameter("superB", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the per-head weights and biases.

        Uses the same scheme as :class:`torch.nn.Linear` (a uniform
        distribution bounded by ``1 / sqrt(in_channels)``), with the fan-in
        taken per head rather than over the stacked parameter.
        """
        bound = 1 / math.sqrt(self.in_channels)

        torch.nn.init.uniform_(self.superW, -bound, bound)

        if self.bias:
            torch.nn.init.uniform_(self.superB, -bound, bound)

    def forward(self, Z: Tensor) -> Tensor:
        """Apply the per-head linear maps.

        Parameters
        ----------
        Z : Tensor
            Input tensor of shape ``[N, r, in_channels]`` where ``r`` is the
            number of heads.

        Returns
        -------
        Tensor
            Output tensor of shape ``[N, r, out_channels]``.
        """
        if self.bias:
            return torch.einsum("roi,Nri->Nro", self.superW, Z) + self.superB

        return torch.einsum("roi,Nri->Nro", self.superW, Z)


class MultiHeadFF(nn.Module):
    """Per-head feed-forward network (a distinct MLP per subspace).

    Stacks :class:`MultiHeadLinear` layers interleaved with activations and
    dropout so that each of the ``r`` heads is transformed by its own
    multi-layer perceptron. Activations and dropout are applied only between
    layers, never after the output layer.

    Parameters
    ----------
    in_channels : int
        Number of input features per head.
    out_channels : int
        Number of output features per head.
    r : int
        Number of heads (subspaces) ``r``.
    hidden_dims : list of int or None, optional
        Widths of the hidden layers. If ``None`` the network is a single
        per-head linear map (default: None).
    act : str, optional
        Name of the activation applied between layers, resolved via
        ``activation_dict`` (default: "leaky_relu").
    drop : float, optional
        Dropout probability applied between layers (default: 0.0).
    bias : bool, optional
        Whether each per-head linear layer uses a bias term (default: True).
    final_activation : bool, optional
        Whether to apply the activation after the output layer as well. When
        ``False`` the activation is applied only between layers (default: False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r: int,
        hidden_dims: list[int] | None = None,
        act: str = "leaky_relu",
        drop: float = 0.0,
        bias: bool = True,
        final_activation: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.dropout = drop
        self.r = r
        self.bias = bias
        self.final_activation = final_activation

        self.layer_sizes = [self.in_channels]

        if hidden_dims is not None:
            self.layer_sizes += [j for j in hidden_dims]
        self.layer_sizes += [self.out_channels]

        els = []
        for j in range(len(self.layer_sizes) - 1):
            els.append(
                MultiHeadLinear(
                    self.layer_sizes[j],
                    self.layer_sizes[j + 1],
                    r_dim=self.r,
                    bias=self.bias,
                )
            )

            if j < len(self.layer_sizes) - 2:
                els.append(activation_dict[self.act]())
                els.append(nn.Dropout(self.dropout))

        # Optionally activate the output as well (e.g. the reference score_lin
        # applies its activation to the final score before the softmax).
        if self.final_activation:
            els.append(activation_dict[self.act]())

        self.model = nn.Sequential(*els)

    def forward(self, Z: Tensor) -> Tensor:
        """Apply the per-head feed-forward network.

        Parameters
        ----------
        Z : Tensor
            Input tensor of shape ``[N, r, in_channels]`` where ``r`` is the
            number of heads.

        Returns
        -------
        Tensor
            Output tensor of shape ``[N, r, out_channels]``.
        """
        Zhat = self.model(Z)

        return Zhat


class FFBlock(nn.Module):
    """Feed-forward block with activations, dropout and a final LayerNorm.

    The block consists of ``n_hidden_layers`` hidden linear layers with an
    activation resolved via ``activation_dict`` followed by an output linear
    layer, with dropout applied after every layer and layer normalization
    applied to the output.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    hidden_dim : int
        Number of hidden units in the intermediate layers.
    n_hidden_layers : int, optional
        Number of hidden layers (must be at least 0). With ``0`` the block
        reduces to a single ``Linear(in_channels, out_channels)`` (default: 1).
    drop : float, optional
        Dropout probability (default: 0.3).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    act : str, optional
        Name of the activation applied after each hidden layer, resolved via
        ``activation_dict`` (default: "gelu").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        n_hidden_layers: int = 1,
        drop: float = 0.3,
        bias: bool = True,
        act: str = "gelu",
    ):
        super().__init__()

        self.dropout = drop
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dimension = hidden_dim
        self.bias = bias
        self.n_hidden_layers = n_hidden_layers
        self.act = act

        assert self.n_hidden_layers >= 0

        els = []

        # Hidden layers (none when n_hidden_layers == 0).
        for layer_index in range(self.n_hidden_layers):
            in_dim = (
                self.in_channels if layer_index == 0 else self.hidden_dimension
            )
            els.append(
                nn.Linear(in_dim, self.hidden_dimension, bias=self.bias)
            )
            els.append(activation_dict[self.act]())
            els.append(nn.Dropout(self.dropout))

        # Output layer. With no hidden layers this is a single linear map from
        # in_channels to out_channels.
        out_in_dim = (
            self.in_channels
            if self.n_hidden_layers == 0
            else self.hidden_dimension
        )
        els.append(nn.Linear(out_in_dim, self.out_channels, bias=self.bias))
        els.append(nn.Dropout(self.dropout))

        self.model = torch.nn.Sequential(*els)
        self.norm = nn.LayerNorm(self.in_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[..., in_channels]``.

        Returns
        -------
        Tensor
            Output tensor of shape ``[..., out_channels]``.
        """

        x = self.norm(x)
        x = self.model(x)

        return x


class LocalCoordinatesLayer(torch.nn.Module):
    """Local coordinate frame layer (equations (2)-(4)).

    For each node this layer projects the node embeddings into ``r`` different
    subspaces, aggregates a smoothed reconstruction over the neighborhood using
    attention weights, and applies a QR decomposition to obtain, per node, an
    orthonormal basis (a local gauge) spanning an ``r``-dimensional subspace of
    the embedding space.

    Parameters
    ----------
    r_subspaces : int
        Number of subspaces (frame vectors) ``r`` learned per node.
    d_embedd : int
        Dimension ``d`` of the node embeddings.
    tau : float, optional
        Temperature used to scale the attention logits (default: 1.0).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    act : str, optional
        Name of the activation used by the feed-forward block, resolved via
        ``activation_dict`` (default: "gelu").
    f_sim_act : str, optional
        Name of the activation used by the per-head similarity network
        ``f_sim``, resolved via ``activation_dict`` (default: "leaky_relu").
    dropout : float, optional
        Dropout probability used by the feed-forward block (default: 0.3).
    f_sim_dropout : float, optional
        Dropout probability used by the per-head similarity network ``f_sim``
        (default: 0.0).
    """

    # Equations (2)-(4).
    def __init__(
        self,
        r_subspaces: int,
        d_embedd: int,
        tau: float = 1.0,
        bias: bool = True,
        act: str = "gelu",
        f_sim_act: str = "leaky_relu",
        dropout: float = 0.3,
        f_sim_dropout: float = 0.0,
    ):
        super().__init__()

        # A d-dimensional space admits at most d orthonormal frame vectors, so
        # r > d would be silently truncated to d by the reduced-mode QR in
        # forward. Reject it explicitly instead of dropping subspaces.
        if r_subspaces > d_embedd:
            raise ValueError(
                f"r_subspaces ({r_subspaces}) cannot exceed d_embedd "
                f"({d_embedd}): a d-dimensional space admits at most d "
                "orthonormal frame vectors."
            )

        self.r = r_subspaces
        self.tau = tau
        self.d = d_embedd
        self.bias = bias
        self.act = act
        self.f_sim_act = f_sim_act
        self.dropout = dropout
        self.f_sim_dropout = f_sim_dropout

        # Combine the per-subspace projectors into a single nn.Linear layer;
        # reshape into r separate projectors afterwards.
        self.initial_projector = torch.nn.Linear(
            self.d, self.d * self.r, bias=self.bias
        )

        # f_sim is the learnable function f of equation (3) that scores the
        # similarity of neighboring node features (one score per subspace).
        #
        # Divergence from the reference implementation: the reference scores
        # each edge with a single linear map on the raw embeddings
        # (score_lin = Sequential(Linear(2*d -> r), LeakyReLU())). Here we use a
        # more expressive per-subspace MLP applied to the projected multi-path
        # embeddings Zh (shape [E, r, 2*d]), yielding the same [E, r] scores.
        # As in the reference, final_activation applies f_sim_act (LeakyReLU by
        # default) to the score itself before the softmax.
        self.f_sim = MultiHeadFF(
            2 * self.d,
            1,
            r=self.r,
            hidden_dims=None,
            act=self.f_sim_act,
            drop=self.f_sim_dropout,
            final_activation=True,
        )

        self.fflayer = FFBlock(
            self.d,
            self.d,
            self.d,
            bias=self.bias,
            act=self.act,
            drop=self.dropout,
        )
        self.preqr_norm = nn.LayerNorm(self.d)

    def forward(self, Z: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass computing per-node local orthonormal frames.

        Implements equations (2)-(4): the neighborhood-smoothed reconstruction
        of the node embeddings (equations (2)-(3)) followed by a QR
        decomposition yielding an orthonormal basis per node (equation (4)).

        Parameters
        ----------
        Z : Tensor
            Node embeddings of shape ``[N, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        """

        N = Z.size(0)  # num_nodes
        src, dst = edge_index[0], edge_index[1]

        # Project the node embeddings into r different subspaces.
        Zh = self.initial_projector(Z)  # [N, r*d]
        Zh = Zh.reshape(N, self.r, self.d)  # [N, r, d]

        # Equation (3): score each edge per subspace; f_vals has shape [E, r, 1].
        # f_sim ends on its activation (final_activation=True), matching the
        # reference score_lin (Linear -> LeakyReLU) before the softmax.
        f_vals = (
            self.f_sim(torch.concat((Zh[src], Zh[dst]), dim=-1)) / self.tau
        )

        f_vals = f_vals.squeeze(-1)  # drop the trailing singleton -> [E, r]
        alphas = torch.softmax(f_vals, dim=-1).unsqueeze(-1)  # [E, r, 1]

        # Equation (2): aggregate the weighted neighbor projections per node.
        out = scatter_add(
            alphas * Zh[src, :, :], index=dst, dim=0, dim_size=N
        )  # [N, r, d]

        # Normalize by the aggregated weights; clamp so that degree-0 nodes
        # (whose scatter_add is 0) do not produce a division by zero. [N, r, 1]
        norm = 1 / (scatter_add(alphas, dst, dim=0, dim_size=N).clamp(1e-6))

        # norm * out broadcasts norm along the last dimension (d) to give
        # [N, r, d]. Z is [N, d], so we insert a subspace axis at -2 before
        # subtracting.
        qhat = Z.unsqueeze(-2) - norm * out

        # Feed-forward followed by a LayerNorm, then QR (equation (4)).
        qhat = self.fflayer(qhat)
        qhat = self.preqr_norm(qhat)

        # Equation (4): qhat is [N, r, d]; transpose to [N, d, r] and apply QR
        # to obtain an orthonormal basis per node.
        Q, _ = torch.linalg.qr(qhat.mT)

        return Q.mT


class GatedFlatteningLayer(nn.Module):
    """Gated flattening layer that smooths local frames (equations (5)-(8)).

    This layer aligns each node's local frame with those of its neighbors. It
    computes gating weights from the overlap between neighboring frames
    (equation (6)), forms a gated aggregate that is blended with the original
    frame (equation (7)), and re-orthonormalizes the result via a QR
    decomposition (equation (8)).

    Parameters
    ----------
    r : int
        Number of subspaces (frame vectors) ``r`` per node.
    gamma : float, optional
        Blending coefficient between the original frame and the aggregated
        neighbor frames (default: 0.01).
    tau : float, optional
        Temperature used to scale the gating logits (default: 1.0).
    """

    # Equations (5)-(8).
    def __init__(self, r: int, gamma: float = 0.01, tau: float = 1.0):
        super().__init__()

        self.r = r
        self.gamma = gamma
        self.tau = tau

    def forward(self, Q: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass smoothing the per-node frames over the graph.

        Implements equations (6)-(8): gating weights from neighboring frame
        overlaps (equation (6)), a gated aggregate blended with the input frame
        (equation (7)), and a final QR re-orthonormalization (equation (8)).

        Parameters
        ----------
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Tensor
            Smoothed per-node orthonormal frames of shape ``[N, r, d]``.
        """
        # Equations (6)-(8).
        N = Q.size(0)  # num_nodes
        src, dst = edge_index[0], edge_index[1]
        k = Q.size(-2)  # with k fixed, the trace of eye(k) = k

        # Equation (6): gating weight from the overlap of neighboring frames.
        # The trace of the k-by-k identity equals k.
        g_vec = scatter_softmax(
            ((Q[src] * Q[dst]).sum((-2, -1)) - k) / self.tau,
            index=dst,
            dim_size=N,
        )

        # Technical note: in principle g_ij is defined for all pairs of nodes,
        # but because we only sum over neighbors, non-neighbor entries never
        # contribute and need not be computed.

        # Equation (7): blend the original frame with the gated neighbor
        # aggregate.
        Qagg = scatter_add(
            g_vec[:, None, None] * Q[src], dim=0, index=dst, dim_size=N
        )
        Qhat = (1 - self.gamma) * Q + self.gamma * Qagg

        # Equation (8): re-orthonormalize the blended frame with another QR.
        Qnew, _ = torch.linalg.qr(Qhat.mT)

        return Qnew.mT


class NodeUpdateLayer(torch.nn.Module):
    """Node feature update layer using the local frames (equations (9)-(10)).

    Each node embedding is projected onto the subspace spanned by its local
    frame and mapped through a learnable matrix (equation (9)). The projected
    features are then aggregated over the neighborhood and, when the residual is
    enabled, combined with a learnable transformation ``phi`` of the original
    embedding (equation (10)). Setting ``phi_hidden_layers`` to ``None`` disables
    the residual and recovers the reference behavior.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    phi_hidden_layers : int or None, optional
        Number of hidden layers of the MLP residual ``phi``. With ``0`` the
        residual is a single linear map (no hidden layer). If ``None`` the
        residual is disabled entirely (matching the reference implementation)
        (default: 1).
    phi_hidden_dim : int or None, optional
        Hidden width of the residual MLP ``phi`` (unused when
        ``phi_hidden_layers`` is ``0``). Defaults to
        ``max(in_channels, out_channels)`` when ``None`` (default: None).
    act : str, optional
        Name of the activation used by the residual MLP ``phi``, resolved via
        ``activation_dict`` (default: "gelu").
    dropout : float, optional
        Dropout probability used by the residual MLP ``phi`` (default: 0.3).
    """

    # Equations (9)-(10).
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        phi_hidden_layers: int | None = 1,
        phi_hidden_dim: int | None = None,
        act: str = "gelu",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.phi = None
        # Learnable function applied to z (only when phi_hidden_layers is not None).
        if phi_hidden_layers is not None:
            self.phi = FFBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=phi_hidden_dim
                if phi_hidden_dim is not None
                else max(in_channels, out_channels),
                n_hidden_layers=phi_hidden_layers,
                act=act,
                drop=dropout,
            )

        # Learnable matrix applied to the frame-projected embedding tilde(z).
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, Z: Tensor, Q: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass updating the node features.

        Implements equations (9)-(10): the frame projection of the node
        embeddings (equation (9)) followed by the neighborhood aggregation and,
        when enabled, a learnable residual connection (equation (10)).

        Parameters
        ----------
        Z : Tensor
            Node embeddings of shape ``[N, d]``.
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Tensor
            Updated node embeddings of shape ``[N, out_channels]``.
        """

        # Step 0: bind commonly used values to local names.
        src, dst = edge_index[0], edge_index[1]
        N = Z.size(0)  # num_nodes, for the scatter ops

        # Step 1: compute the frame-projected embedding tilde(z).
        # Equation (9): Q is [N, r, d] and Z is [N, d]; project each node
        # embedding onto its own frame, batching over the node dimension.
        QtZ = torch.einsum("ijk,ik->ij", Q, Z)
        Z_tilde = torch.einsum("ikj, ik->ij", Q, QtZ)
        Z_tilde = self.W(Z_tilde)

        # Equation (10): aggregate the projected embeddings over the neighborhood.
        #
        # Divergence from the reference implementation: we optionally add a
        # residual connection through self.phi. It is enabled by default and can
        # be disabled (recovering the reference behavior) by passing
        # phi_hidden_layers=None, in which case self.phi is None.
        Znew = scatter_mean(Z_tilde[src], index=dst, dim=0, dim_size=N)

        if self.phi is not None:
            Znew = Znew + self.phi(Z)

        return Znew


class GaugeLayer(torch.nn.Module):
    """A single gauge message-passing layer.

    One layer computes per-node local frames with a
    :class:`LocalCoordinatesLayer` (equations (2)-(4)), smooths them through a
    stack of ``n_gated`` :class:`GatedFlatteningLayer` modules (equations
    (5)-(8)), and updates the node features with a :class:`NodeUpdateLayer`
    (equations (9)-(10)).

    Parameters
    ----------
    d_embedd : int
        Dimension ``d`` of the node embeddings.
    r : int
        Number of subspaces (frame vectors) ``r`` per node.
    n_gated : int, optional
        Number of gated flattening layers applied to the frames (default: 1).
    gamma : float, optional
        Blending coefficient used by the gated flattening layers (default: 0.01).
    tau : float, optional
        Temperature used to scale the attention and gating logits (default: 1.0).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    phi_hidden_layers : int or None, optional
        Number of hidden layers of the MLP residual ``phi`` in the node update.
        With ``0`` the residual is a single linear map (no hidden layer). If
        ``None`` the residual is disabled entirely (matching the reference
        implementation) (default: 1).
    phi_hidden_dim : int or None, optional
        Hidden width of the residual MLP ``phi``. Defaults to ``d_embedd`` when
        ``None`` (default: None).
    act : str, optional
        Name of the activation used by the feed-forward blocks, resolved via
        ``activation_dict`` (default: "gelu").
    f_sim_act : str, optional
        Name of the activation used by the per-head similarity network
        ``f_sim``, resolved via ``activation_dict`` (default: "leaky_relu").
    dropout : float, optional
        Dropout probability used by the feed-forward blocks (default: 0.3).
    f_sim_dropout : float, optional
        Dropout probability used by the per-head similarity network ``f_sim``
        (default: 0.0).
    """

    def __init__(
        self,
        d_embedd: int,
        r: int,
        n_gated: int = 1,
        gamma: float = 0.01,
        tau: float = 1.0,
        bias=True,
        phi_hidden_layers: int | None = 1,
        phi_hidden_dim: int | None = None,
        act: str = "gelu",
        f_sim_act: str = "leaky_relu",
        dropout: float = 0.3,
        f_sim_dropout: float = 0.0,
    ):
        super().__init__()

        self.r = r
        self.bias = bias
        self.tau = tau
        self.n_gated = n_gated
        self.gamma = gamma
        self.d_embedd = d_embedd
        self.act = act
        self.f_sim_act = f_sim_act
        self.dropout = dropout
        self.f_sim_dropout = f_sim_dropout

        self.local_coords_layer = LocalCoordinatesLayer(
            r_subspaces=r,
            d_embedd=d_embedd,
            tau=tau,
            bias=bias,
            act=act,
            f_sim_act=f_sim_act,
            dropout=dropout,
            f_sim_dropout=f_sim_dropout,
        )

        self.gated_flattening_layers = nn.ModuleList(
            [
                GatedFlatteningLayer(self.r, self.gamma, self.tau)
                for _ in range(n_gated)
            ]
        )

        self.node_update_layer = NodeUpdateLayer(
            self.d_embedd,
            self.d_embedd,
            phi_hidden_layers=phi_hidden_layers,
            phi_hidden_dim=phi_hidden_dim,
            act=act,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of a single gauge layer.

        Parameters
        ----------
        x : Tensor
            Node embeddings of shape ``[N, d]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.

        Returns
        -------
        Znew : Tensor
            Updated node embeddings of shape ``[N, d]``.
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d]``.
        """

        Q = self.local_coords_layer(x, edge_index)

        for layer in self.gated_flattening_layers:
            Q = layer(Q, edge_index)

        Znew = self.node_update_layer(x, Q, edge_index)

        return Znew, Q


class GaugeModel(nn.Module):
    """Riemannian graph foundation model with neural vector bundles.

    The model first projects the input features into a ``d_embedd``-dimensional
    embedding space and then applies a stack of :class:`GaugeLayer` modules,
    each of which learns per-node local frames and uses them to update the node
    features.

    Parameters
    ----------
    n_layers : int
        Number of gauge layers.
    in_channels : int
        Number of input features.
    r : int
        Number of subspaces (frame vectors) ``r`` per node.
    d_embedd : int
        Dimension ``d`` of the node embeddings.
    n_gated : int, optional
        Number of gated flattening layers per gauge layer (default: 1).
    gamma : float, optional
        Blending coefficient used by the gated flattening layers (default: 0.01).
    tau : float, optional
        Temperature used to scale the attention and gating logits (default: 1.0).
    bias : bool, optional
        Whether the linear layers use a bias term (default: True).
    phi_hidden_layers : int or None, optional
        Number of hidden layers of the MLP residual ``phi`` in the node update.
        With ``0`` the residual is a single linear map (no hidden layer). If
        ``None`` the residual is disabled entirely (matching the reference
        implementation) (default: 1).
    phi_hidden_dim : int or None, optional
        Hidden width of the residual MLP ``phi``. Defaults to ``d_embedd`` when
        ``None`` (default: None).
    act : str, optional
        Name of the activation used by the feed-forward blocks, resolved via
        ``activation_dict`` (default: "gelu").
    f_sim_act : str, optional
        Name of the activation used by the per-head similarity network
        ``f_sim``, resolved via ``activation_dict`` (default: "leaky_relu").
    dropout : float, optional
        Dropout probability used by the feed-forward blocks (default: 0.3).
    f_sim_dropout : float, optional
        Dropout probability used by the per-head similarity network ``f_sim``
        (default: 0.0).
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        r: int,
        d_embedd: int,
        n_gated: int = 1,
        gamma=0.01,
        tau: float = 1.0,
        bias=True,
        phi_hidden_layers: int | None = 1,
        phi_hidden_dim: int | None = None,
        act: str = "gelu",
        f_sim_act: str = "leaky_relu",
        dropout: float = 0.3,
        f_sim_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.gamma = gamma
        self.tau = tau
        self.bias = bias
        self.in_channels = in_channels
        self.r = r
        self.d_embedd = d_embedd
        self.n_gated = n_gated
        self.act = act
        self.f_sim_act = f_sim_act
        self.dropout = dropout
        self.f_sim_dropout = f_sim_dropout

        self.input_projector = nn.Sequential(
            nn.Linear(in_channels, d_embedd), nn.LayerNorm(d_embedd)
        )

        self.layers = nn.ModuleList(
            [
                GaugeLayer(
                    d_embedd=self.d_embedd,
                    r=self.r,
                    gamma=self.gamma,
                    tau=self.tau,
                    n_gated=self.n_gated,
                    bias=bias,
                    phi_hidden_layers=phi_hidden_layers,
                    phi_hidden_dim=phi_hidden_dim,
                    act=self.act,
                    f_sim_act=self.f_sim_act,
                    dropout=self.dropout,
                    f_sim_dropout=self.f_sim_dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, return_initial: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the full model.

        Parameters
        ----------
        x : Tensor
            Input node features of shape ``[N, in_channels]``.
        edge_index : Tensor
            Edge index tensor of shape ``[2, E]`` with source and destination
            node indices.
        return_initial : bool
            Whether to return the initial projection of x (default: False).

        Returns
        -------
        z : Tensor
            Final node embeddings of shape ``[N, d_embedd]``.
        Q : Tensor
            Per-node orthonormal frames of shape ``[N, r, d_embedd]`` from the
            last gauge layer.
        z0 : Tensor
            The initial projection of ``x``, of shape ``[N, d_embedd]``. Only
            returned when ``return_initial`` is True.
        """
        # Self-loops would make each node its own neighbor in the per-layer
        # aggregation, which is not intended. Removed here (and, consistently,
        # in DirichletLoss) so both aggregate over the same neighborhoods.
        edge_index, _ = remove_self_loops(edge_index)

        z = self.input_projector(x)

        if return_initial:
            z0 = z.clone()

        for layer in self.layers:
            z, Q = layer(z, edge_index)

        if return_initial:
            return z, Q, z0

        return z, Q
