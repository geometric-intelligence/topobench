r"""Neural Sheaf Propagation (NSP), the sheaf *wave* equation on graphs.

NSP [1] replaces the parabolic sheaf **diffusion** of Bodnar et al. [2] with the
hyperbolic sheaf **wave** equation, discretised by the leapfrog method.

The continuous dynamics ([1], Eq. (3)) act on 0-cochains
:math:`\mathbf{X}(t) \in C^0(G; \mathcal{F})`:

.. math::
    \ddot{\mathbf{X}}(t) = -\Delta_{\mathcal{F}(t)} \mathbf{X}(t)

Sheaf diffusion ([2], and [1], Eq. (1)) is instead first order in time:
:math:`\dot{\mathbf{X}}(t) = -\Delta_{\mathcal{F}(t)}\mathbf{X}(t)`.

The discrete layer ([1], Section 3, the equation following Proposition 1)
applies the leapfrog stencil to :math:`\ddot{\mathbf{X}}`, keeping the same
learnable channel/stalk mixing that [2], Eq. (2) applies to the Laplacian term:

.. math::
    \mathbf{X}_{t+1} = 2\mathbf{X}_t - \mathbf{X}_{t-1}
        - h\,\sigma\!\left(
            \Delta_{\mathcal{F}(t)}(\mathbf{I} \otimes W_1^t)\mathbf{X}_t W_2^t
          \right)

Because the wave equation is non-dissipative it conserves the sheaf energy
:math:`\mathcal{E}_{\mathcal{F}}(\mathbf{X}) = \tfrac{1}{2}(\lVert
\dot{\mathbf{X}} \rVert^2 + \mathbf{X}^\top \Delta_{\mathcal{F}(t)}\mathbf{X})`
([1], Proposition 1) rather than smoothing it away, which is what makes it
attractive on heterophilic graphs.

The sheaf itself is learned exactly as in [2]: the restriction maps, the
(normalised) sheaf Laplacian :math:`\Delta_{\mathcal{F}} = D^{-1/2}
L_{\mathcal{F}} D^{-1/2}` and the orthogonal parameterisation all come from
:mod:`topobench.nn.backbones.graph.nsd_utils`, shared with the NSD backbone.

[1] Suk, Giusti, Hemo, Lopez, Barmpas, Bodnar. "Surfing on the Neural Sheaf."
    NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations.
    https://openreview.net/pdf?id=xOXFkyRzTlu
[2] Bodnar, Di Giovanni, Chamberlain, Liò, Bronstein. "Neural Sheaf Diffusion:
    A Topological Perspective on Heterophily and Oversmoothing in GNNs."
    NeurIPS 2022. https://arxiv.org/abs/2202.04579
"""

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn
from torch_geometric.utils import remove_self_loops, to_undirected

from topobench.nn.backbones.graph.nsd_utils.laplacian_builders import (
    DiagLaplacianBuilder,
    GeneralLaplacianBuilder,
    NormConnectionLaplacianBuilder,
)
from topobench.nn.backbones.graph.nsd_utils.sheaf_models import (
    LocalConcatSheafLearner,
)

#: The three restriction-map families of [1], Section 4 / Table 1 ``diag``
#: is Diag-NSP, ``bundle`` is O(d)-NSP and ``general`` is Gen-NSP mapped to
#: their Laplacian builder, the per-edge parameter shape the sheaf learner must
#: emit for a stalk dimension ``d``, and any fixed builder options.
#:
#: [1], Section 2 takes :math:`\Delta_{\mathcal{F}}` to be the *normalised*
#: sheaf Laplacian, so ``diag`` asks its builder to normalise and ``bundle``
#: normalises by construction. ``general`` is the one deviation: block
#: normalisation of full :math:`d \times d` maps needs a batched inverse matrix
#: square root, which is numerically fragile, so Gen-NSP propagates the
#: unnormalised :math:`L_{\mathcal{F}}` and wants a smaller ``step_size``.
SHEAF_TYPES = {
    "diag": (DiagLaplacianBuilder, lambda d: (d,), {"normalised": True}),
    "bundle": (
        NormConnectionLaplacianBuilder,
        lambda d: (d * (d + 1) // 2,),
        {},
    ),
    "general": (GeneralLaplacianBuilder, lambda d: (d, d), {}),
}


class NSPEncoder(nn.Module):
    r"""Neural Sheaf Propagation encoder.

    Stacks ``num_layers`` leapfrog steps of the sheaf wave equation ([1],
    Section 3). Each step relearns the sheaf :math:`\mathcal{F}(t)` from the
    current 0-cochain, builds the (normalised) sheaf Laplacian, and advances the
    signal with the second-order stencil :math:`\mathbf{X}_{t+1} =
    2\mathbf{X}_t - \mathbf{X}_{t-1} - h\,\sigma(\Delta_{\mathcal{F}(t)}
    (\mathbf{I} \otimes W_1^t)\mathbf{X}_t W_2^t)`.

    Parameters
    ----------
    input_dim : int
        Dimension of the input node features.
    hidden_dim : int
        Width of the 0-cochain space, i.e. ``d * hidden_channels``. Must be an
        exact multiple of ``d``: a node stalk holds ``d`` vectors of
        ``hidden_dim // d`` channels each, so a remainder would silently shrink
        the requested width.
    num_layers : int, optional
        Number of leapfrog steps ([1], Table 2 sweeps 2-8). Default is 2.
    sheaf_type : str, optional
        Restriction-map family, one of the keys of :data:`SHEAF_TYPES`:
        ``'diag'`` (Diag-NSP), ``'bundle'`` (O(d)-NSP, orthogonal maps) or
        ``'general'`` (Gen-NSP). Default is ``'bundle'``.
    d : int, optional
        Stalk dimension :math:`\dim \mathcal{F}(v)` ([1], Table 2 sweeps 2-5).
        ``'bundle'`` and ``'general'`` need ``d > 1`` to be more than a scalar
        sheaf. Default is 2.
    step_size : float, optional
        Leapfrog step :math:`h` scaling the propagation term ([1], Table 2
        sweeps 0.1-1.0). Default is 1.0, which reproduces the layer equation
        exactly as printed in [1].
    dropout : float, optional
        Dropout rate inside the propagation layers. Default is 0.1.
    input_dropout : float, optional
        Dropout rate on the input features. Default is 0.1.
    sheaf_act : str, optional
        Activation of the sheaf learner: ``'tanh'``, ``'elu'`` or ``'id'``.
        Default is ``'tanh'``.
    orth : str, optional
        Orthogonal parameterisation used by ``sheaf_type='bundle'``:
        ``'cayley'`` or ``'matrix_exp'``. Default is ``'cayley'``.
    **kwargs : dict
        Ignored, for compatibility with the shared model configs.

    Raises
    ------
    ValueError
        If ``sheaf_type`` is unknown, if ``hidden_dim`` is not a multiple of
        ``d``, or if ``d`` is out of range for the chosen ``sheaf_type``.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        sheaf_type="bundle",
        d=2,
        step_size=1.0,
        dropout=0.1,
        input_dropout=0.1,
        sheaf_act="tanh",
        orth="cayley",
        **kwargs,
    ):
        super().__init__()

        if sheaf_type not in SHEAF_TYPES:
            raise ValueError(
                f"Unknown sheaf_type {sheaf_type!r}, "
                f"expected one of {sorted(SHEAF_TYPES)}"
            )
        if d < 1 or (d < 2 and sheaf_type != "diag"):
            raise ValueError(
                f"sheaf_type={sheaf_type!r} needs d > 1, got d={d}"
            )
        if hidden_dim % d != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be a multiple of d={d}, "
                "otherwise the stalk reshape truncates the hidden width"
            )

        builder_cls, out_shape_of, builder_kwargs = SHEAF_TYPES[sheaf_type]
        self.builder_cls = builder_cls
        self.builder_kwargs = dict(builder_kwargs)
        if sheaf_type == "bundle":
            # The builder turns the learner's d(d+1)/2 parameters into an
            # orthogonal matrix, so it needs the parameterisation.
            self.builder_kwargs["orth_map"] = orth

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_channels = hidden_dim // d
        self.sheaf_type = sheaf_type
        self.d = d
        self.num_layers = num_layers
        self.step_size = step_size
        self.dropout = dropout
        self.input_dropout = input_dropout

        # W_1^t of the layer equation: mixes the d stalk coordinates.
        self.lin_left_weights = nn.ModuleList(
            nn.Linear(d, d, bias=False) for _ in range(num_layers)
        )
        # W_2^t of the layer equation: mixes the feature channels.
        self.lin_right_weights = nn.ModuleList(
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            for _ in range(num_layers)
        )
        for left, right in zip(
            self.lin_left_weights, self.lin_right_weights, strict=True
        ):
            nn.init.eye_(left.weight.data)
            nn.init.orthogonal_(right.weight.data)

        # F(t) is relearned at every step, so one learner per layer.
        self.sheaf_learners = nn.ModuleList(
            LocalConcatSheafLearner(
                hidden_dim, out_shape=out_shape_of(d), sheaf_act=sheaf_act
            )
            for _ in range(num_layers)
        )

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def _mix(self, x, layer, num_nodes):
        r"""Apply :math:`(\mathbf{I} \otimes W_1^t)\,\mathbf{X}\,W_2^t`.

        Parameters
        ----------
        x : torch.Tensor
            0-cochain of shape ``[num_nodes * d, hidden_channels]``.
        layer : int
            Index of the propagation layer whose weights to use.
        num_nodes : int
            Number of nodes in the current (possibly batched) graph.

        Returns
        -------
        torch.Tensor
            Mixed 0-cochain, same shape as ``x``.
        """
        # Transpose so that the d stalk coordinates of a (node, channel) pair
        # land on the last axis, apply W_1 there, then restore and apply W_2.
        x = x.t().reshape(-1, self.d)
        x = self.lin_left_weights[layer](x)
        x = x.reshape(-1, num_nodes * self.d).t()
        return self.lin_right_weights[layer](x)

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        edge_weight=None,
        batch=None,
        **kwargs,
    ):
        r"""Propagate node features through the sheaf wave equation.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, input_dim]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``. Symmetrised and stripped
            of self-loops, since a cellular sheaf is defined over an undirected
            graph ([1], Section 2).
        edge_attr : torch.Tensor, optional
            Unused, accepted for wrapper compatibility. Default is None.
        edge_weight : torch.Tensor, optional
            Unused, accepted for wrapper compatibility. Default is None.
        batch : torch.Tensor, optional
            Unused: batched graphs are already a single disconnected graph, on
            which the sheaf Laplacian is block-diagonal. Default is None.
        **kwargs : dict
            Ignored.

        Returns
        -------
        torch.Tensor
            Node features of shape ``[num_nodes, hidden_dim]``.
        """
        num_nodes = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        laplacian_builder = self.builder_cls(
            num_nodes, edge_index, d=self.d, **self.builder_kwargs
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Stack the features into a 0-cochain: row block v holds the d-vector
        # of node v, so X lives in R^{(nd) x f} as in [1], Section 2.
        x = x.view(num_nodes * self.d, -1)

        # Leapfrog needs two states. Starting from X_{-1} = X_0 is the
        # zero-initial-velocity condition, so the first step reduces to the
        # diffusion step X_1 = X_0 - h*sigma(...).
        x_prev, x_curr = x, x
        for layer in range(self.num_layers):
            # Learn F(t) from the current cochain and assemble Delta_{F(t)}.
            x_maps = F.dropout(
                x_curr,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            maps = self.sheaf_learners[layer](
                x_maps.reshape(num_nodes, -1), edge_index
            )
            laplacian, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            # sigma(Delta_{F(t)} (I kron W_1^t) X_t W_2^t)
            x_prop = F.dropout(x_curr, p=self.dropout, training=self.training)
            x_prop = self._mix(x_prop, layer, num_nodes)
            x_prop = torch_sparse.spmm(
                laplacian[0],
                laplacian[1],
                x_prop.size(0),
                x_prop.size(0),
                x_prop,
            )
            x_prop = F.elu(x_prop)

            # X_{t+1} = 2 X_t - X_{t-1} - h * sigma(...)
            x_prev, x_curr = (
                x_curr,
                2 * x_curr - x_prev - self.step_size * x_prop,
            )

        # The wave equation does not dissipate, so a diverging run shows up here
        # rather than being smoothed away silently.
        if not torch.all(torch.isfinite(x_curr)):
            raise RuntimeError(
                "NSP propagation diverged (non-finite cochain); "
                "lower step_size or num_layers"
            )

        return self.lin2(x_curr.reshape(num_nodes, -1))
