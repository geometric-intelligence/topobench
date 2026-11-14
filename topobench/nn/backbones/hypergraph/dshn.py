r"""Directional Sheaf Hypergraph Network (DSHN) backbone for TopoBench.

Implementation of

    E. Mule, S. Fiorini, A. Purificato, F. Siciliano, S. Coniglio and
    F. Silvestri, *"Directional Sheaf Hypergraph Networks: Unifying Learning
    on Directed and Undirected Hypergraphs"*, ICLR 2026,
    https://arxiv.org/abs/2510.04727

with reference to the authors' implementation,
https://github.com/EmaMule/DirectionalSheafHypergraphs, where the model is
named ``SheafGeDi`` (``models/sheafgedi.py``) rather than DSHN.

The model diffuses complex node features with the Directed Sheaf Hypergraph
Laplacian of :mod:`.dshn_utils.laplacian`. Writing
:math:`Q^F_N = I_{nd} - L^F_N`, one layer is (Eq. 9, p. 7)

.. math::
    X_{t+1} = \sigma\big( Q^F_N (I_n \otimes W_1) X_t W_2 \big),
    \qquad X_t \in \mathbb{C}^{nd \times f},

with :math:`W_1 \in \mathbb{R}^{d \times d}`,
:math:`W_2 \in \mathbb{R}^{f \times f}` and :math:`\sigma` the complex ReLU.
The optional residual form (Appendix C.2, p. 25) adds :math:`X_t` inside
:math:`\sigma`. After the last layer the features are mapped back to the
reals with ``unwind`` (p. 8) before the readout.

Two variants are exposed, matching the paper:

``DSHN`` (``light=False``)
    The restriction maps are learned end to end.

``DSHNLight`` (``light=True``)
    The Laplacian is built outside the autograd graph, so :math:`\Phi`'s
    parameters stay fixed: "the parameters of the MLP responsible for
    predicting the restriction maps ... remain fixed throughout the training
    process. The model's adaptability arises from the initial projection
    layer" (p. 8). Cheaper, and the variant that fits where DSHN runs out of
    memory (Table 6).

.. important::
    On undirected hypergraphs the directional machinery does nothing. With
    :math:`H(e) = \emptyset` every phase product in Eq. 4 is :math:`1`, so
    :math:`L^F` is real and its spectrum is independent of :math:`q`; by
    Theorem 6 the operator is then the undirected hypergraph Laplacian of
    Zhou et al. (2006), sheaf-ified. The paper presents this as a
    contribution in its own right (p. 7): it "provides the first definition
    of a Sheaf Hypergraph Laplacian suitable for undirected hypergraphs",
    differing from Duta et al. (2023) only in the diagonal coefficient
    :math:`1 - 1/\delta_e`, which is what makes it positive semidefinite.

    Since TopoBench liftings yield undirected hypergraphs, use
    ``orientation="star"`` to induce the orientation of Appendix D.5 and
    exercise the directional part; see
    :func:`~.dshn_utils.laplacian.derive_orientation`.
"""

import torch
import torch_geometric
from torch import nn
from torch_scatter import scatter_mean

# Absolute imports are required here: ``BackboneExportsManager`` discovers
# backbones by exec'ing each module standalone via ``spec_from_file_location``,
# so relative imports would not resolve.
from topobench.nn.backbones.hypergraph.dshn_utils.complex_ops import (
    ComplexLayerNorm,
    RealLinear,
    complex_dropout,
    complex_relu,
    unwind,
)
from topobench.nn.backbones.hypergraph.dshn_utils.laplacian import (
    charge_phase,
    derive_orientation,
    directed_sheaf_laplacian,
)
from topobench.nn.backbones.hypergraph.dshn_utils.sheaf_builders import (
    SHEAF_BUILDERS,
)


class DSHN(nn.Module):
    r"""Directional Sheaf Hypergraph Network.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Complex feature width :math:`f` of each diffusion layer.
    out_channels : int
        Number of output channels produced for the readout.
    n_layers : int, optional
        Number of diffusion layers (default: ``3``, the reference's
        ``--sheaf_num_layers``). The paper searches :math:`\{1,\dots,5\}` and
        reports accuracy still improving at 7 layers (Fig. 3), i.e. no
        oversmoothing.
    d : int, optional
        Stalk dimension (default: ``2``). The reference's argparse default is
        ``1``; the paper searches :math:`\{1,\dots,6\}` (§D.3) and its
        published best configurations use ``6`` for DSHN and ``4`` for
        DSHNLight. We default to ``2``, the smallest value that leaves
        orthogonal and general restriction maps non-trivial, because the
        operator costs :math:`O(\sum_e \delta_e^2 d^2)` and the challenge
        grid runs on a modest memory budget.
    q : float, optional
        Charge parameter :math:`q` (default: ``0.25``, the reference default
        and the top of the search grid :math:`\{0.00,\dots,0.25\}`).
        ``q = 0.25`` is the Sign-Magnetic point of Theorem 5. This parameter
        does nothing at all unless ``orientation`` induces a direction.
    sheaf_type : str, optional
        Restriction-map family: ``"DiagSheafs"`` (default),
        ``"OrthoSheafs"``, or ``"GeneralSheafs"``. Every DSHN result in the
        paper uses the diagonal family.
    orientation : str, optional
        How to split each hyperedge into head and tail sets: ``"none"``
        (default, undirected) or ``"star"`` (Appendix D.5). See
        :func:`~.dshn_utils.laplacian.derive_orientation`.
    light : bool, optional
        If ``True``, build the Laplacian outside the autograd graph, giving
        the DSHNLight variant (default: ``False``).
    dropout : float, optional
        Dropout probability applied between diffusion layers (default:
        ``0.5``).
    sheaf_act : str, optional
        Nonlinearity inside :math:`\Phi` (default: ``"sigmoid"``).
    sheaf_dropout : bool, optional
        Whether to drop predicted restriction-map parameters (default:
        ``False``).
    sheaf_left_proj : bool, optional
        Whether to apply :math:`I_n \otimes W_1` (default: ``False``,
        following the reference; note Eq. 9 always includes it).
    dynamic_sheaf : bool, optional
        Whether to re-predict the sheaf at every layer rather than reusing the
        first layer's (default: ``False``).
    residual : bool, optional
        Whether to add the residual connection of p. 25 (default: ``False``).
    layer_norm : bool, optional
        Whether to apply :class:`~.dshn_utils.complex_ops.ComplexLayerNorm`
        before each diffusion layer (default: ``False``). Corresponds to the
        reference's ``--AllSet_input_norm`` for the convolutional path.
    input_norm : bool, optional
        Whether to layer-normalize the inputs of :math:`\Phi` (default:
        ``True``).
    init_hedge : str, optional
        How to initialize hyperedge features, ``"avg"`` (default) or
        ``"rand"``.
    add_identity : bool, optional
        Whether to use :math:`D_V + I` in place of :math:`D_V` when
        normalizing (default: ``False``). Not part of Eq. 7; the reference
        enables it by default. See
        :func:`~.dshn_utils.laplacian.directed_sheaf_laplacian`.
    orthogonal_map : str, optional
        Parameterization for ``"OrthoSheafs"``: ``"cayley"`` (default) or
        ``"matrix_exp"``.

    Raises
    ------
    ValueError
        If ``sheaf_type`` is not a known restriction-map family.

    Notes
    -----
    Deliberate departures from the reference implementation, each defaulting
    to the published definition:

    * ``add_identity`` defaults to ``False`` (reference: ``True``); the term
      appears in neither Eq. 2 nor Eq. 7.
    * The charge uses the paper's :math:`e^{-2\pi i q}` on tails; the
      reference uses the conjugate.
    * The imaginary part of :math:`X_0` is initialized to zero. The reference
      copies the real part (``x_img = data.x.clone().detach()``, with
      ``zeros_like`` commented out), which on undirected input makes
      ``unwind`` emit two identical halves.
    * :math:`\sigma` is applied after every layer, per Eq. 9; the reference
      omits it after the last.
    * Only the ``MLP_var1`` restriction-map predictor is implemented, as it
      is the one the paper documents.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 3,
        d: int = 2,
        q: float = 0.25,
        sheaf_type: str = "DiagSheafs",
        orientation: str = "none",
        light: bool = False,
        dropout: float = 0.5,
        sheaf_act: str = "sigmoid",
        sheaf_dropout: bool = False,
        sheaf_left_proj: bool = False,
        dynamic_sheaf: bool = False,
        residual: bool = False,
        layer_norm: bool = False,
        input_norm: bool = True,
        init_hedge: str = "avg",
        add_identity: bool = False,
        orthogonal_map: str = "cayley",
    ) -> None:
        super().__init__()
        if sheaf_type not in SHEAF_BUILDERS:
            raise ValueError(
                f"Unknown sheaf_type {sheaf_type!r}; expected one of "
                f"{tuple(SHEAF_BUILDERS)}."
            )

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.d = d
        self.q = q
        self.sheaf_type = sheaf_type
        self.orientation = orientation
        self.light = light
        self.dropout = dropout
        self.left_proj = sheaf_left_proj
        self.dynamic_sheaf = dynamic_sheaf
        self.residual = residual
        self.init_hedge = init_hedge
        self.add_identity = add_identity

        # X_0: project the input features into d stalks of width f.
        self.lin = nn.Linear(in_channels, hidden_channels * d)

        builder_cls = SHEAF_BUILDERS[sheaf_type]
        builder_kwargs = {
            "sheaf_act": sheaf_act,
            "sheaf_dropout": sheaf_dropout,
            "dropout": dropout,
            "input_norm": input_norm,
        }
        if sheaf_type == "OrthoSheafs":
            builder_kwargs["orthogonal_map"] = orthogonal_map
        n_builders = n_layers if dynamic_sheaf else 1
        self.sheaf_builder = nn.ModuleList(
            builder_cls(d, hidden_channels, **builder_kwargs)
            for _ in range(n_builders)
        )

        # W_2 of Eq. 9, one per layer.
        self.lin_right = nn.ModuleList(
            RealLinear(hidden_channels, hidden_channels)
            for _ in range(n_layers)
        )
        # W_1 of Eq. 9 (the I_n (x) W_1 factor), one per layer.
        self.lin_left = (
            nn.ModuleList(
                RealLinear(d, d, bias=False) for _ in range(n_layers)
            )
            if sheaf_left_proj
            else None
        )
        self.norms = (
            nn.ModuleList(
                ComplexLayerNorm(hidden_channels) for _ in range(n_layers)
            )
            if layer_norm
            else None
        )

        # unwind doubles the width, and the d stalks are flattened into it.
        self.lin_out = nn.Linear(2 * d * hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the network."""
        self.lin.reset_parameters()
        self.lin_out.reset_parameters()
        for builder in self.sheaf_builder:
            builder.reset_parameters()
        for layer in self.lin_right:
            layer.reset_parameters()
        if self.lin_left is not None:
            for layer in self.lin_left:
                layer.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def init_hyperedge_attr(
        self, x: torch.Tensor, edge_index: torch.Tensor, num_hyperedges: int
    ) -> torch.Tensor:
        r"""Initialize hyperedge features :math:`x_e`.

        The paper (§C.1) notes that when hyperedge features are absent "they
        are computed by aggregating the features of the hyperedge's nodes
        using a mean or a sum".

        Parameters
        ----------
        x : torch.Tensor
            Real node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.
        num_hyperedges : int
            Number of hyperedges :math:`m`.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[num_hyperedges, in_channels]``.
        """
        if self.init_hedge == "rand":
            return torch.rand(
                (num_hyperedges, x.size(-1)), device=x.device, dtype=x.dtype
            )
        return scatter_mean(
            x[edge_index[0]],
            edge_index[1],
            dim=0,
            dim_size=num_hyperedges,
        )

    def build_operator(
        self,
        x: torch.Tensor,
        hyperedge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_hyperedges: int,
        layer: int,
    ) -> torch.Tensor:
        r"""Predict a sheaf and assemble :math:`Q^F_N` for one layer.

        Parameters
        ----------
        x : torch.Tensor
            Complex node features of shape ``[num_nodes * d, f]``.
        hyperedge_attr : torch.Tensor
            Complex hyperedge features of shape ``[num_hyperedges * d, f]``.
        edge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.
        num_nodes : int
            Number of nodes :math:`n`.
        num_hyperedges : int
            Number of hyperedges :math:`m`.
        layer : int
            Index of the layer being built, used to select the sheaf builder
            when ``dynamic_sheaf`` is set.

        Returns
        -------
        torch.Tensor
            Sparse complex tensor of shape ``[num_nodes * d, num_nodes * d]``.
        """
        builder = self.sheaf_builder[layer if self.dynamic_sheaf else 0]
        f_blocks = builder(
            x, hyperedge_attr, edge_index, num_nodes, num_hyperedges
        )
        if self.light:
            # DSHNLight: the operator leaves the autograd graph, so Phi gets
            # no gradient. Gradient still reaches the projection layer through
            # the sparse-dense product in `forward`.
            f_blocks = f_blocks.detach()

        is_head = derive_orientation(
            edge_index, num_nodes, num_hyperedges, self.orientation
        )
        phase = charge_phase(is_head, self.q)
        return directed_sheaf_laplacian(
            edge_index,
            f_blocks,
            phase,
            num_nodes,
            num_hyperedges,
            normalized=True,
            add_identity=self.add_identity,
            diagonal_degrees=self.sheaf_type != "GeneralSheafs",
        )

    def forward(
        self, x: torch.Tensor, incidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Diffuse node features with the Directed Sheaf Hypergraph Laplacian.

        Parameters
        ----------
        x : torch.Tensor
            Real node features of shape ``[num_nodes, in_channels]``.
        incidence : torch.Tensor
            Hypergraph incidence, either a sparse COO tensor of shape
            ``[num_nodes, num_hyperedges]`` or a dense ``[2, nnz]`` index.

        Returns
        -------
        x_0 : torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        x_1 : torch.Tensor
            Hyperedge embeddings of shape ``[num_hyperedges, out_channels]``,
            mean-aggregated from the node embeddings over each hyperedge.
        """
        if incidence.layout == torch.sparse_coo:
            edge_index, _ = torch_geometric.utils.to_edge_index(
                incidence.coalesce()
            )
        else:
            edge_index = incidence
        edge_index = edge_index.to(x.device)

        num_nodes = x.size(0)
        num_hyperedges = int(edge_index[1].max().item()) + 1
        d, f = self.d, self.hidden_channels

        hyperedge_attr = self.init_hyperedge_attr(
            x, edge_index, num_hyperedges
        )

        # X_0 in C^{nd x f}; the imaginary part starts at zero (see Notes).
        h = self.lin(x).view(num_nodes * d, f)
        e = self.lin(hyperedge_attr).view(num_hyperedges * d, f)
        h = torch.complex(h, torch.zeros_like(h))
        e = torch.complex(e, torch.zeros_like(e))

        operator = None
        for i in range(self.n_layers):
            if i == 0 or self.dynamic_sheaf:
                operator = self.build_operator(
                    h, e, edge_index, num_nodes, num_hyperedges, i
                )

            residual = h
            if self.norms is not None:
                h = self.norms[i](h)
            if self.lin_left is not None:
                # (I_n (x) W_1) X: act on the stalk axis of each node.
                h = self.lin_left[i](h.view(num_nodes, d, f).transpose(1, 2))
                h = h.transpose(1, 2).reshape(num_nodes * d, f)
            h = self.lin_right[i](h)
            h = torch.sparse.mm(operator, h)
            if self.residual:
                h = h + residual
            h = complex_relu(h)
            if i < self.n_layers - 1:
                h = complex_dropout(h, self.dropout, self.training)

        # unwind: C^{nd x f} -> R^{n x 2df}  (p. 8; see complex_ops.unwind).
        out = self.lin_out(unwind(h.view(num_nodes, d * f)))
        x_1 = scatter_mean(
            out[edge_index[0]],
            edge_index[1],
            dim=0,
            dim_size=num_hyperedges,
        )
        return out, x_1
