"""Diagonal Sheaf Hypergraph Neural Network backbone.

This module implements the linear, diagonal SheafHyperGNN from:

    Duta, Cassara, Silvestri, and Lio, "Sheaf Hypergraph Networks",
    NeurIPS 2023. https://arxiv.org/abs/2309.17116

The paper defines the linear sheaf Laplacian in Definition 2 / Equation (1)
and a Sheaf Hypergraph Network layer in Section 3.3 as

``Y = sigma((I - Delta) (I x W_1) X_tilde W_2)``.

This implementation follows the diagonal variant of the official
``SheafHyperGNN`` model, obtained by combining ``SheafHyperGNN``,
``SheafBuilderDiag``, and ``HyperDiffusionDiagSheafConv``, while making the
following TopoBench-specific adaptations:

* In the backbone, ``forward`` receives the node features and the node-hyperedge
  incidence matrix as two separate tensors, instead of a complete PyG ``Data``
  object.
* The builder receives the total node and hyperedge counts explicitly.
  ``num_nodes`` determines the stalk-expanded output shape, while the incidence
  matrix width retains isolated hyperedges that are absent from the nonzero
  coordinates.
* Hyperedge features are recomputed for every batch. The original code reuses
  them because it processes one fixed complete hypergraph, whereas TopoBench
  batches may contain different hypergraphs.
* The backbone returns the final ``d * hidden_channels`` node embeddings.
  TopoBench's readout converts them into predictions and replaces the original
  model's final ``lin2`` layer.
* The code uses ELU between diffusion layers because that is what the official
  implementation uses, although Section 3.3 of the paper describes the generic
  activation as ReLU.

Only the diagonal restriction-map family used by the submitted configuration
is implemented. Orthogonal, low-rank, and general restriction maps are separate
model variants in the reference repository and are intentionally out of scope.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class SheafHyperGNN(nn.Module):
    """Linear Sheaf Hypergraph Neural Network.

    The model attaches a ``stalk_dim``-dimensional stalk to each node and
    hyperedge, predicts a restriction map for every non-zero incidence
    ``(node, hyperedge)`` pair, and applies the sheaf diffusion operator from
    Duta et al. (NeurIPS 2023).

    Parameters
    ----------
    in_channels : int
        Number of input node feature channels.
    hidden_channels : int
        Number of feature channels in each stalk coordinate. The backbone
        returns ``stalk_dim * hidden_channels`` channels per node.
    stalk_dim : int, optional
        Stalk dimension ``d``. This corresponds to ``args.heads`` in the
        official implementation.
    num_layers : int, optional
        Number of sheaf diffusion layers.
    sheaf_act : str, optional
        Activation applied to predicted restriction maps. ``"tanh"`` matches the
        submitted configuration and the official README example, and preserves
        signed maps. The original command-line default is ``"sigmoid"``.
        Supported values are ``"tanh"``, ``"sigmoid"``, and ``"none"``.
    sheaf_normtype : str, optional
        Normalization type from the reference implementation. Supported values
        are ``"degree_norm"``, ``"sym_degree_norm"``, ``"block_norm"``, and
        ``"sym_block_norm"``.
    sheaf_pred_block : str, optional
        Hyperedge-feature prediction block. Supported values are
        ``"MLP_var1"``, ``"MLP_var2"``, ``"MLP_var3"``, and ``"cp_decomp"``.
    dropout : float, optional
        Dropout between diffusion layers and on sheaf maps when
        ``sheaf_dropout=True``.
    sheaf_dropout : bool, optional
        Whether to apply dropout to predicted restriction maps.
    dynamic_sheaf : bool, optional
        If ``True``, recompute restriction maps at every layer. Otherwise,
        reuse the first-layer maps, matching the reference static-sheaf option.
    init_hedge : str, optional
        Hyperedge feature initialization: ``"avg"`` or ``"rand"``.
    input_norm : bool, optional
        Whether internal linear predictors use LayerNorm before projection.
    left_proj : bool, optional
        Whether to apply a learnable left projection on the stalk dimension
        inside each diffusion layer.
    residual : bool, optional
        Whether each diffusion layer adds its linearly transformed input.
    sheaf_special_head : bool, optional
        For diagonal sheaves, force the last stalk coordinate to behave like a
        standard hypergraph-convolution head, following the reference option.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stalk_dim: int = 2,
        num_layers: int = 2,
        sheaf_act: str = "tanh",
        sheaf_normtype: str = "sym_degree_norm",
        sheaf_pred_block: str = "cp_decomp",
        dropout: float = 0.0,
        sheaf_dropout: bool = False,
        dynamic_sheaf: bool = False,
        init_hedge: str = "avg",
        input_norm: bool = True,
        left_proj: bool = False,
        residual: bool = False,
        sheaf_special_head: bool = False,
    ) -> None:
        super().__init__()

        if stalk_dim < 1:
            raise ValueError("stalk_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be in [0, 1].")
        if init_hedge not in {"avg", "rand"}:
            raise ValueError("init_hedge must be 'avg' or 'rand'.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.d = stalk_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.dynamic_sheaf = dynamic_sheaf
        self.init_hedge = init_hedge
        self.out_channels = hidden_channels * stalk_dim

        self.lin_in = _MLP(
            in_channels=in_channels,
            out_channels=hidden_channels * stalk_dim,
            input_norm=False,
        )

        num_builders = num_layers if dynamic_sheaf else 1
        self.sheaf_builders = nn.ModuleList(
            [
                _DiagonalSheafBuilder(
                    hidden_channels=hidden_channels,
                    stalk_dim=stalk_dim,
                    apply_dropout=sheaf_dropout,
                    dropout=dropout,
                    sheaf_act=sheaf_act,
                    prediction_type=sheaf_pred_block,
                    special_head=sheaf_special_head,
                    input_norm=input_norm,
                )
                for _ in range(num_builders)
            ]
        )

        self.convs = nn.ModuleList(
            [
                _DiagonalSheafConv(
                    hidden_channels=hidden_channels,
                    stalk_dim=stalk_dim,
                    norm_type=sheaf_normtype,
                    input_norm=input_norm,
                    left_proj=left_proj,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        )

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.lin_in.reset_parameters()
        for builder in self.sheaf_builders:
            builder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x_0: torch.Tensor,
        incidence_hyperedges: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Run SheafHyperGNN on a TopoBench hypergraph batch.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        incidence_hyperedges : torch.Tensor
            Dense or sparse COO incidence matrix with shape
            ``[num_nodes, num_hyperedges]``.

        Returns
        -------
        tuple[torch.Tensor, None]
            Node embeddings of shape
            ``[num_nodes, stalk_dim * hidden_channels]`` and a placeholder for
            compatibility with ``HypergraphWrapper``.
        """
        # Convert TopoBench's incidence matrix to the coordinate format used by
        # the reference implementation.
        hyperedge_index, num_edges = _incidence_to_hyperedge_index(
            incidence_hyperedges
        )
        num_nodes = x_0.size(0)

        hyperedge_attr = self._init_hyperedge_attr(
            x_0, hyperedge_index, num_edges
        )

        x = self.lin_in(x_0).view(num_nodes * self.d, self.hidden_channels)
        e = self.lin_in(hyperedge_attr).view(
            num_edges * self.d, self.hidden_channels
        )

        h_idx, h_val = self.sheaf_builders[0](
            x, e, hyperedge_index, num_nodes, num_edges
        )

        for layer_idx, conv in enumerate(self.convs):
            if self.dynamic_sheaf and layer_idx > 0:
                # Recompute the restriction maps before every later diffusion
                # layer, including the final layer.
                h_idx, h_val = self.sheaf_builders[layer_idx](
                    x,
                    e,
                    hyperedge_index,
                    num_nodes,
                    num_edges,
                )

            # Apply one sheaf Laplacian diffusion layer.
            x = conv(x, h_idx, h_val, num_nodes, num_edges)

            # As in the reference implementation, apply ELU and dropout only
            # between diffusion layers, not after the final layer.
            if layer_idx < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # The reference applies ``lin2: dH -> num_classes`` at this point.
        # TopoBench keeps task-specific classification in its readout, so the
        # backbone exposes the same dH representation used as input by ``lin2``.
        x = x.view(num_nodes, self.out_channels)
        return x, None

    def _init_hyperedge_attr(
        self,
        x_0: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        """Initialize batch-local hyperedge features.

        The reference stores these features on the model after the first
        forward pass because it trains one fixed hypergraph. Recomputing them
        is equivalent for ``avg`` in that setting and is required when
        TopoBench supplies a different graph in the next mini-batch.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        hyperedge_index : torch.Tensor
            Non-zero incidence coordinates of shape ``[2, num_incidences]``.
        num_edges : int
            Number of hyperedges, including isolated hyperedges.

        Returns
        -------
        torch.Tensor
            Hyperedge features of shape ``[num_edges, in_channels]``.
        """
        if self.init_hedge == "rand":
            return torch.randn(
                num_edges,
                self.in_channels,
                device=x_0.device,
                dtype=x_0.dtype,
            )
        if hyperedge_index.numel() == 0:
            return x_0.new_zeros((num_edges, self.in_channels))
        return torch_scatter.scatter(
            x_0[hyperedge_index[0]],
            hyperedge_index[1],
            dim=0,
            dim_size=num_edges,
            reduce="mean",
        )


def _incidence_to_hyperedge_index(
    incidence_hyperedges: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Convert a TopoBench incidence matrix to the reference coordinate format.

    TopoBench stores node-hyperedge membership as a matrix of shape
    ``[num_nodes, num_hyperedges]``. The reference implementation instead uses a
    ``[2, num_incidences]`` tensor whose first row contains node indices and whose
    second row contains hyperedge indices.

    Only positions with nonzero values are used because the model learns its own
    restriction values. The total number of hyperedges is returned separately
    so isolated hyperedges, which have no nonzero coordinates, remain
    represented.

    Parameters
    ----------
    incidence_hyperedges : torch.Tensor
        Dense or sparse COO node-to-hyperedge incidence matrix.

    Returns
    -------
    tuple[torch.Tensor, int]
        Nonzero incidence coordinates and the total number of hyperedges.
    """
    num_edges = incidence_hyperedges.size(1)
    if incidence_hyperedges.layout == torch.sparse_coo:
        incidence_hyperedges = incidence_hyperedges.coalesce()
        nonzero_mask = incidence_hyperedges.values() != 0
        hyperedge_index = incidence_hyperedges.indices()[
            :, nonzero_mask
        ].long()
        return hyperedge_index, num_edges
    return incidence_hyperedges.nonzero(
        as_tuple=False
    ).t().contiguous().long(), num_edges


class _MLP(nn.Module):
    """One-layer predictor used by the official SheafHyperGNN code.

    The reference MLP supports hidden layers, but every MLP used by the
    diagonal SheafHyperGNN is configured with ``num_layers=1``. Its hidden
    width, hidden-activation, and hidden-layer dropout are therefore never
    used. This implementation keeps only the executed behavior: optional
    input LayerNorm followed by one linear projection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    input_norm : bool, optional
        Whether to apply LayerNorm before the linear projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(in_channels) if input_norm else nn.Identity()]
        )
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels)])

    def reset_parameters(self) -> None:
        """Reset parameters (reinitialise the linear layers and learnable normalization parameters)."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.normalizations:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optional input normalization and the linear projection.

        Parameters
        ----------
        x : torch.Tensor
            Input features with the channel dimension last.

        Returns
        -------
        torch.Tensor
            Linearly projected features.
        """
        return self.lins[0](self.normalizations[0](x))


class _DiagonalSheafBuilder(nn.Module):
    """Predict diagonal restriction maps for every non-zero incidence pair.

    This is the TopoBench equivalent of the official ``SheafBuilderDiag``.
    The four restriction-map prediction branches mirror the corresponding
    ``predict_blocks*`` functions in the reference implementation.

    This implementation matches the reference builder's operations, with three
    fixes for TopoBench use. For ``cp_V``, it replaces the hidden-channel count
    mistakenly passed to the normalization option with the configured
    ``input_norm`` boolean. It also handles empty incidence data and uses
    explicit graph dimensions so isolated hyperedges remain represented.

    Parameters
    ----------
    hidden_channels : int
        Number of feature channels per stalk coordinate.
    stalk_dim : int
        Stalk dimension.
    apply_dropout : bool
        Whether to apply dropout to predicted restriction maps.
    dropout : float, optional
        Dropout probability.
    sheaf_act : str, optional
        Activation applied to restriction-map values.
    prediction_type : str, optional
        Method used to construct hyperedge features before predicting the
        restriction maps.
    special_head : bool, optional
        Whether to fix the final diagonal value to one, creating a channel
        similar to standard hypergraph convolution.
    input_norm : bool, optional
        Whether predictor projections use input LayerNorm.
    """

    def __init__(
        self,
        hidden_channels: int,
        stalk_dim: int,
        apply_dropout: bool,
        dropout: float = 0.0,
        sheaf_act: str = "tanh",
        prediction_type: str = "cp_decomp",
        special_head: bool = False,
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        if prediction_type not in {
            "MLP_var1",
            "MLP_var2",
            "MLP_var3",
            "cp_decomp",
        }:
            raise ValueError(
                "prediction_type must be one of 'MLP_var1', 'MLP_var2', "
                "'MLP_var3', or 'cp_decomp'."
            )
        if sheaf_act not in {"tanh", "sigmoid", "none"}:
            raise ValueError("sheaf_act must be 'tanh', 'sigmoid', or 'none'.")

        self.hidden_channels = hidden_channels
        self.d = stalk_dim
        self.apply_dropout = apply_dropout
        self.dropout = dropout
        self.sheaf_act = sheaf_act
        self.prediction_type = prediction_type
        self.special_head = special_head

        self.sheaf_lin = _MLP(
            in_channels=2 * hidden_channels,
            out_channels=stalk_dim,
            input_norm=input_norm,
        )

        if prediction_type == "MLP_var3":
            self.sheaf_lin2 = _MLP(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                input_norm=input_norm,
            )
        elif prediction_type == "cp_decomp":
            self.cp_W = _MLP(
                in_channels=hidden_channels + 1,
                out_channels=hidden_channels,
                input_norm=input_norm,
            )
            self.cp_V = _MLP(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                input_norm=input_norm,
            )

    def reset_parameters(self) -> None:
        """Reset parameters (reinitialise the diagonal restriction-map predictor layers)."""
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == "cp_decomp":
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the expanded sparse sheaf incidence matrix.

        Parameters
        ----------
        x : torch.Tensor
            Stalk-expanded node features.
        e : torch.Tensor
            Stalk-expanded hyperedge features.
        hyperedge_index : torch.Tensor
            Non-zero node-to-hyperedge incidence coordinates.
        num_nodes : int
            Total number of nodes, used to reshape the stalk-expanded features.
        num_edges : int
            Total number of hyperedges, taken from the incidence-matrix width so
            hyperedges with no nonzero incidences are still counted.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Expanded sparse incidence coordinates and restriction-map values.
        """
        if hyperedge_index.numel() == 0:
            empty_index = hyperedge_index.new_empty((2, 0))
            self.last_restriction_maps = x.new_empty((0, self.d))
            return empty_index, self.last_restriction_maps.reshape(-1)

        x_mean = x.view(num_nodes, self.d, -1).mean(dim=1)
        e_mean = e.view(num_edges, self.d, -1).mean(dim=1)

        restriction_diagonals = self._predict_blocks(
            x_mean, e_mean, hyperedge_index, num_edges
        )

        if self.apply_dropout:
            restriction_diagonals = F.dropout(
                restriction_diagonals,
                p=self.dropout,
                training=self.training,
            )

        if self.special_head:
            mask = restriction_diagonals.new_ones(self.d)
            head = restriction_diagonals.new_zeros(self.d)
            mask[-1] = 0.0
            head[-1] = 1.0
            restriction_diagonals = restriction_diagonals * mask + head

        # Store the unexpanded restriction maps only for tests, as in the
        # reference implementation.
        self.last_restriction_maps = restriction_diagonals
        return _expand_diagonal(hyperedge_index, restriction_diagonals, self.d)

    def _predict_blocks(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        """Predict one diagonal restriction vector per incidence.

        All variants concatenate each incident node feature with a hyperedge
        feature, then use ``sheaf_lin`` to predict ``stalk_dim`` diagonal
        restriction values. They differ only in how the hyperedge feature is
        constructed:

        * ``MLP_var1`` uses the hyperedge features passed into the builder.
        * ``MLP_var2`` averages the current features of the incident nodes.
        * ``MLP_var3`` transforms the node features and sums them per
          hyperedge.
        * ``cp_decomp`` combines product-pooled transformed node features with
          sum-pooled node features.

        Parameters
        ----------
        x : torch.Tensor
            Stalk-reduced node features.
        e : torch.Tensor
            Stalk-reduced hyperedge features.
        hyperedge_index : torch.Tensor
            Non-zero node-to-hyperedge incidence coordinates.
        num_edges : int
            Number of hyperedges.

        Returns
        -------
        torch.Tensor
            Restriction vectors of shape ``[num_incidences, stalk_dim]``.
        """
        row, col = hyperedge_index
        x_row = x.index_select(0, row)

        if self.prediction_type == "MLP_var1":
            edge_features = e.index_select(0, col)
        elif self.prediction_type == "MLP_var2":
            # Average the current node features within each hyperedge.
            pooled = torch_scatter.scatter(
                x_row,
                col,
                dim=0,
                dim_size=num_edges,
                reduce="mean",
            )
            edge_features = pooled.index_select(0, col)
        elif self.prediction_type == "MLP_var3":
            # Transform the nodes, then sum them within each hyperedge.
            lifted = self.sheaf_lin2(x)
            pooled = torch_scatter.scatter(
                lifted.index_select(0, row),
                col,
                dim=0,
                dim_size=num_edges,
                reduce="sum",
            )
            edge_features = pooled.index_select(0, col)
        else:
            # Compute a CP-decomposition-inspired product representation.
            ones = x_row.new_ones((x_row.shape[0], 1))
            x_with_bias = torch.cat((x_row, ones), dim=-1)
            product_terms = torch.tanh(self.cp_W(x_with_bias))
            pooled_prod = torch_scatter.scatter(
                product_terms,
                col,
                dim=0,
                dim_size=num_edges,
                reduce="mul",
            )
            pooled_sum = torch_scatter.scatter(
                x_row,
                col,
                dim=0,
                dim_size=num_edges,
                reduce="sum",
            )
            # Add the sum-pooled node features, matching the reference.
            edge_features = torch.relu(self.cp_V(pooled_prod)) + torch.relu(
                pooled_sum
            )
            edge_features = edge_features.index_select(0, col)

        # Predict d diagonal values for every nonzero incidence.
        restriction_diagonals = self.sheaf_lin(
            torch.cat((x_row, edge_features), dim=-1)
        )
        if self.sheaf_act == "tanh":
            return torch.tanh(restriction_diagonals)
        if self.sheaf_act == "sigmoid":
            return torch.sigmoid(restriction_diagonals)
        return restriction_diagonals


def _expand_diagonal(
    hyperedge_index: torch.Tensor,
    restriction_diagonals: torch.Tensor,
    stalk_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand diagonal ``d``-vectors to sparse ``Nd x Ed`` coordinates.

    For each nonzero incidence ``(node, hyperedge)``, the model predicts ``d``
    diagonal restriction values. This function expands that incidence into the
    coordinates ``(node * d + k, hyperedge * d + k)`` for ``k = 0, ..., d - 1``,
    producing the sparse ``Nd x Ed`` sheaf incidence representation. Off-diagonal
    coordinates are omitted because the restriction maps are diagonal.

    The expansion follows the reference implementation's ``repeat``, ``permute``,
    and ``reshape`` structure so its coordinate and value ordering can be
    compared directly.

    Parameters
    ----------
    hyperedge_index : torch.Tensor
        Non-zero incidence coordinates of shape ``[2, num_incidences]``.
    restriction_diagonals : torch.Tensor
        Diagonal restriction vectors.
    stalk_dim : int
        Stalk dimension.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Expanded sparse coordinates and flattened restriction values.
    """
    stalk_offsets = (
        torch.arange(stalk_dim, device=hyperedge_index.device)
        .view(1, -1, 1)
        .repeat(2, 1, 1)
    )
    expanded_index = stalk_dim * hyperedge_index.unsqueeze(1) + stalk_offsets
    expanded_index = expanded_index.permute(0, 2, 1).reshape(2, -1)

    return expanded_index, restriction_diagonals.reshape(-1)


class _DiagonalSheafConv(nn.Module):
    """One diagonal linear-sheaf diffusion layer.

    The official ``HyperDiffusionDiagSheafConv`` explicitly forms the sparse
    operator ``M = D_norm H B^-1 H^T``, where ``D_norm`` is ``D^-1`` or
    ``D^-1/2`` depending on the normalization variant. It then flips the sign
    of every node-diagonal block and adds the identity. This implementation
    applies the same ``I + M - 2 * blockdiag(M)`` operator using native PyTorch
    sparse matrices.

    Parameters
    ----------
    hidden_channels : int
        Number of feature channels per stalk coordinate.
    stalk_dim : int
        Stalk dimension.
    norm_type : str, optional
        Sheaf normalization variant.
    input_norm : bool, optional
        Whether internal projections use input LayerNorm.
    left_proj : bool, optional
        Whether to learn a projection across stalk coordinates.
    residual : bool, optional
        Whether to add the linearly transformed input.
    bias : bool, optional
        Whether to add a learnable output bias.
    """

    def __init__(
        self,
        hidden_channels: int,
        stalk_dim: int,
        norm_type: str = "sym_degree_norm",
        input_norm: bool = False,
        left_proj: bool = False,
        residual: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if norm_type not in {
            "degree_norm",
            "sym_degree_norm",
            "block_norm",
            "sym_block_norm",
        }:
            raise ValueError(
                "norm_type must be 'degree_norm', 'sym_degree_norm', "
                "'block_norm', or 'sym_block_norm'."
            )
        self.hidden_channels = hidden_channels
        self.d = stalk_dim
        self.norm_type = norm_type
        self.left_proj = left_proj
        self.residual = residual

        if left_proj:
            self.lin_left_proj = _MLP(
                in_channels=stalk_dim,
                out_channels=stalk_dim,
                input_norm=input_norm,
            )

        self.lin = _MLP(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            input_norm=input_norm,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_channels))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """Reset parameters (reinitialise the diffusion layer’s projections and bias)."""
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        alpha: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        """Apply sheaf diffusion to stalk-expanded node features.

        Parameters
        ----------
        x : torch.Tensor
            Stalk-expanded node features.
        hyperedge_index : torch.Tensor
            Expanded sparse sheaf-incidence coordinates.
        alpha : torch.Tensor
            Expanded restriction-map values.
        num_nodes : int
            Number of nodes.
        num_edges : int
            Number of hyperedges.

        Returns
        -------
        torch.Tensor
            Updated stalk-expanded node features.
        """
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1, num_nodes * self.d).t()

        x = self.lin(x)
        data_x = x

        # Depending on norm_type, use D^-1 or D^-1/2.
        D_inv, B_inv = _normalisation_vectors(
            x=x,
            node_idx=hyperedge_index[0],
            edge_idx=hyperedge_index[1],
            h_val=alpha,
            num_nodes=num_nodes,
            num_edges=num_edges,
            stalk_dim=self.d,
            norm_type=self.norm_type,
        )

        if self.norm_type in {"sym_degree_norm", "sym_block_norm"}:
            # Compute D^(-1/2) @ x.
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse_coo_tensor(
            hyperedge_index,
            alpha,
            size=(num_nodes * self.d, num_edges * self.d),
        ).coalesce()
        H_t = torch.sparse_coo_tensor(
            hyperedge_index.flip([0]),
            alpha,
            size=(num_edges * self.d, num_nodes * self.d),
        ).coalesce()

        B_inv = _sparse_diagonal(B_inv)
        D_inv = _sparse_diagonal(D_inv)

        # Form minus_L = D_inv @ H @ B_inv @ H_t, following the original.
        minus_L = torch.sparse.mm(B_inv, H_t).coalesce()
        minus_L = torch.sparse.mm(H, minus_L).coalesce()
        minus_L = torch.sparse.mm(D_inv, minus_L).coalesce()

        I_mask = _node_block_mask(num_nodes, self.d, x)
        Id = _sparse_diagonal(x.new_ones(num_nodes * self.d))

        # Negate the diagonal blocks and add the identity matrix.
        minus_L = minus_L - 2 * minus_L.mul(I_mask)
        minus_L = (Id + minus_L).coalesce()

        out = torch.sparse.mm(minus_L, x)

        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x
        return out


def _normalisation_vectors(
    x: torch.Tensor,
    node_idx: torch.Tensor,
    edge_idx: torch.Tensor,
    h_val: torch.Tensor,
    num_nodes: int,
    num_edges: int,
    stalk_dim: int,
    norm_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalization vectors used by the reference implementation.

    Parameters
    ----------
    x : torch.Tensor
        Stalk-expanded node features, used for device and dtype.
    node_idx : torch.Tensor
        Node-coordinate indices of expanded incidences.
    edge_idx : torch.Tensor
        Hyperedge-coordinate indices of expanded incidences.
    h_val : torch.Tensor
        Restriction-map values at expanded incidences.
    num_nodes : int
        Number of nodes.
    num_edges : int
        Number of hyperedges.
    stalk_dim : int
        Stalk dimension.
    norm_type : str
        Sheaf normalization variant.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Node and hyperedge normalization vectors.
    """
    node_size = num_nodes * stalk_dim
    edge_size = num_edges * stalk_dim
    ones = x.new_ones(h_val.shape[0])

    if norm_type in {"degree_norm", "sym_degree_norm"}:
        node_degree = torch_scatter.scatter(
            ones,
            node_idx,
            dim=0,
            dim_size=node_size,
            reduce="sum",
        )
    else:
        node_degree = torch_scatter.scatter(
            h_val * h_val,
            node_idx,
            dim=0,
            dim_size=node_size,
            reduce="sum",
        )

    edge_degree = torch_scatter.scatter(
        ones,
        edge_idx,
        dim=0,
        dim_size=edge_size,
        reduce="sum",
    )

    node_power = (
        -0.5 if norm_type in {"sym_degree_norm", "sym_block_norm"} else -1.0
    )
    node_norm = torch.zeros_like(node_degree)
    node_mask = node_degree > 0
    node_norm[node_mask] = node_degree[node_mask].pow(node_power)

    edge_norm = torch.zeros_like(edge_degree)
    edge_mask = edge_degree > 0
    edge_norm[edge_mask] = edge_degree[edge_mask].reciprocal()

    return node_norm, edge_norm


def _sparse_diagonal(values: torch.Tensor) -> torch.Tensor:
    """Create a square sparse matrix with ``values`` on its diagonal.

    Parameters
    ----------
    values : torch.Tensor
        Values to place on the matrix diagonal.

    Returns
    -------
    torch.Tensor
        Coalesced sparse COO diagonal matrix.
    """
    diagonal = torch.arange(values.numel(), device=values.device)
    indices = torch.stack((diagonal, diagonal))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(values.numel(), values.numel()),
    ).coalesce()


def _node_block_mask(
    num_nodes: int,
    stalk_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a sparse mask containing every node-diagonal stalk block.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    stalk_dim : int
        Stalk dimension.
    reference : torch.Tensor
        Tensor providing the output device and dtype.

    Returns
    -------
    torch.Tensor
        Coalesced sparse COO node-block mask.
    """
    nodes = torch.arange(num_nodes, device=reference.device)
    stalk_coordinates = torch.arange(stalk_dim, device=reference.device)
    block_rows = (
        nodes[:, None, None] * stalk_dim + stalk_coordinates[None, :, None]
    ).expand(-1, -1, stalk_dim)
    block_cols = (
        nodes[:, None, None] * stalk_dim + stalk_coordinates[None, None, :]
    ).expand(-1, stalk_dim, -1)
    indices = torch.stack((block_rows.reshape(-1), block_cols.reshape(-1)))
    size = (num_nodes * stalk_dim, num_nodes * stalk_dim)
    return torch.sparse_coo_tensor(
        indices,
        reference.new_ones(indices.shape[1]),
        size=size,
    ).coalesce()
