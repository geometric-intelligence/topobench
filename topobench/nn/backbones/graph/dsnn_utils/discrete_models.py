r"""Discrete directed sheaf diffusion (Eq. 7 and 8).

The deployed model of the paper is Eq. 8,

.. math::

    X^{(t+1)} = \mathrm{diag}(1 + \varepsilon) X^{(t)} - \sigma\!\Big(
        L_N^{\tilde{\mathcal{F}}(t)}
        \big(I_n \otimes W_1^{(t)}\big) X^{(t)} W_2^{(t)} \Big),

with :math:`X^{(t)} \in \mathbb{C}^{nd \times f}`, real weights
:math:`W_1 \in \mathbb{R}^{d \times d}` and
:math:`W_2 \in \mathbb{R}^{f \times f}` (Eq. 6), a learnable
:math:`\varepsilon \in [-1, 1]^{d}` tiled over the nodes, and the complex
nonlinearity of Sec. 3. Setting :math:`\varepsilon = 0` recovers Eq. 7. The
superscript :math:`\tilde{\mathcal{F}}(t)` matters: a **new sheaf, and hence a
new Laplacian, is learned at every layer**.

The readout is the paper's
:math:`\mathrm{unwind}(X^{(\tau)}) = (\Re(X^{(\tau)}) \Vert \Im(X^{(\tau)}))`
followed by a linear map to the output dimension. The paper does not state that
final linear map explicitly, but ``unwind`` alone cannot change width, and both
Neural Sheaf Diffusion and MagNet have one.

One class covers all three restriction-map families, selected by
``sheaf_type``; the Neural Sheaf Diffusion port in TopoBench instead repeats a
near-identical forward pass in three subclasses.

Notes
-----
Details the paper leaves open, and how they are resolved:
* **What :math:`\Phi` consumes.** Definition 1 types the base restriction maps
  real and Eq. 6 types the weights real, yet :math:`X^{(t)}` is complex from
  the second layer on, and the activations the paper searches over
  (``elu``, ``relu``) are undefined for complex input. We therefore feed
  :math:`\Phi` a real view: ``sheaf_features="unwind"`` (the default) passes
  :math:`(\Re \Vert \Im)`, ``"real"`` passes :math:`\Re` only. The base maps
  stay real either way, which is what keeps the degree blocks real and the
  operator's diagonal free of the phase.
* **Dropout** is not part of Eq. 8. Its placement follows the reference
  implementation whose flag names the paper reuses; masks drop whole complex
  entries rather than the real and imaginary parts independently.
* **The imaginary part of** :math:`X^{(0)}` is zero: inputs are real and the
  paper is silent, and zeros keep the two halves of ``unwind`` from being
  duplicates at the first layer.
"""

import torch
import torch.nn.functional as F
from torch import nn

from topobench.nn.backbones.graph.dsnn_utils.complex_ops import (
    complex_dropout_split,
    complex_relu_split,
    stack_real,
    unwind_split,
)
from topobench.nn.backbones.graph.dsnn_utils.laplace import (
    compute_left_right_map_index,
    spmm,
    symmetrize_support,
)
from topobench.nn.backbones.graph.dsnn_utils.laplacian_builders import (
    LAPLACIAN_BUILDERS,
)
from topobench.nn.backbones.graph.dsnn_utils.orthogonal import (
    ORTHOGONAL_MAPS,
    num_orthogonal_params,
)
from topobench.nn.backbones.graph.dsnn_utils.phase import (
    ORIENTATIONS,
    pair_sign,
    phase_from_sign,
)
from topobench.nn.backbones.graph.dsnn_utils.sheaf_models import (
    LocalConcatSheafLearner,
)

SHEAF_TYPES = ("diag", "bundle", "general")
SHEAF_FEATURES = ("unwind", "real")


class DirectedSheafDiffusion(nn.Module):
    """Stack of directed sheaf diffusion layers implementing Eq. 8.

    Parameters
    ----------
    config : dict
        Configuration with keys ``d``, ``layers``, ``hidden_channels``,
        ``input_dim``, ``output_dim``, ``input_dropout``, ``dropout``,
        ``sheaf_act``, ``sheaf_type``, ``q``, ``orientation``,
        ``phase_sign``, ``normalised``, ``degree_shift``, ``block_norm``,
        ``orth`` and ``sheaf_features``.

    Raises
    ------
    ValueError
        If ``sheaf_type``, ``sheaf_features``, ``orientation`` or ``orth`` is
        unknown, if ``d`` is incompatible with the chosen family, or if
        ``layers`` is not positive.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.sheaf_type = config["sheaf_type"]
        if self.sheaf_type not in SHEAF_TYPES:
            raise ValueError(
                f"Unknown sheaf type: {self.sheaf_type!r}; expected one of "
                f"{SHEAF_TYPES}"
            )
        self.sheaf_features = config["sheaf_features"]
        if self.sheaf_features not in SHEAF_FEATURES:
            raise ValueError(
                f"Unknown sheaf_features {self.sheaf_features!r}; expected "
                f"one of {SHEAF_FEATURES}"
            )
        self.orientation = config["orientation"]
        if self.orientation not in ORIENTATIONS:
            raise ValueError(
                f"Unknown orientation {self.orientation!r}; expected one of "
                f"{ORIENTATIONS}"
            )
        self.orth = config["orth"]
        if self.orth not in ORTHOGONAL_MAPS:
            raise ValueError(
                f"Unknown orth {self.orth!r}; expected one of "
                f"{ORTHOGONAL_MAPS}"
            )

        self.d = config["d"]
        if self.d < 1:
            raise ValueError(f"d must be at least 1, got {self.d}")
        if self.sheaf_type == "bundle" and self.d < 2:
            raise ValueError(
                "The orthogonal family needs d >= 2, since d = 1 leaves "
                "d * (d - 1) // 2 == 0 free parameters"
            )
        self.layers = config["layers"]
        if self.layers < 1:
            raise ValueError(f"layers must be at least 1, got {self.layers}")

        self.hidden_channels = config["hidden_channels"]
        self.hidden_dim = self.hidden_channels * self.d
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.input_dropout = config["input_dropout"]
        self.dropout = config["dropout"]
        self.q = float(config["q"])
        self.phase_sign = config["phase_sign"]
        self.normalised = config["normalised"]
        self.degree_shift = config["degree_shift"]
        self.block_norm = config["block_norm"]

        sheaf_in = self.hidden_dim
        if self.sheaf_features == "unwind":
            sheaf_in *= 2

        self.lin_left_weights = nn.ModuleList()
        self.lin_right_weights = nn.ModuleList()
        self.sheaf_learners = nn.ModuleList()
        self.epsilons = nn.ParameterList()
        for _ in range(self.layers):
            left = nn.Linear(self.d, self.d, bias=False)
            nn.init.eye_(left.weight.data)
            self.lin_left_weights.append(left)
            right = nn.Linear(
                self.hidden_channels, self.hidden_channels, bias=False
            )
            nn.init.orthogonal_(right.weight.data)
            self.lin_right_weights.append(right)
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    sheaf_in,
                    out_shape=self.out_shape(),
                    sheaf_act=config["sheaf_act"],
                )
            )
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        # unwind doubles the width, so this is 2 * hidden_dim wide. The
        # Neural Sheaf Diffusion readout is half as wide, which is why a
        # state dict is not interchangeable between the two models.
        self.lin2 = nn.Linear(2 * self.hidden_dim, self.output_dim)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sheaf_type={self.sheaf_type!r}, "
            f"d={self.d}, layers={self.layers}, "
            f"hidden_channels={self.hidden_channels}, q={self.q}, "
            f"orientation={self.orientation!r})"
        )

    def out_shape(self) -> tuple[int, ...]:
        """Return the shape of the parameters predicted for each arc.

        Returns
        -------
        tuple of int
            ``(d,)`` for diagonal maps, ``(d * (d - 1) // 2,)`` for orthogonal
            maps and ``(d, d)`` for general maps.
        """
        if self.sheaf_type == "diag":
            return (self.d,)
        if self.sheaf_type == "bundle":
            return (num_orthogonal_params(self.d),)
        return (self.d, self.d)

    def build_builder(self, num_nodes: int, edge_index, dtype):
        """Prepare the structural bookkeeping and the Laplacian builder.

        Everything here is integer or constant with respect to the parameters,
        so it runs once per forward pass, outside the autograd graph, and is
        shared by every layer.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the batch.
        edge_index : torch.Tensor
            Raw arc indices of shape ``[2, num_edges]``.
        dtype : torch.dtype
            Floating dtype for the phase.

        Returns
        -------
        builder : DirectedLaplacianBuilder
            Builder configured for this graph.
        support : torch.Tensor
            The symmetrized arc support of shape ``[2, 2 * num_pairs]``.
        """
        with torch.no_grad():
            support = symmetrize_support(edge_index, num_nodes)
            left_right_idx, pair_index = compute_left_right_map_index(
                support, num_nodes
            )
            sign = pair_sign(
                edge_index,
                support,
                pair_index,
                num_nodes,
                self.orientation,
            )
            cos_phase, sin_phase = phase_from_sign(
                sign, self.q, self.phase_sign, dtype=dtype
            )
        kwargs = {
            "normalised": self.normalised,
            "degree_shift": self.degree_shift,
            "block_norm": self.block_norm,
            "training": self.training,
        }
        if self.sheaf_type == "bundle":
            kwargs["orth_map"] = self.orth
        builder = LAPLACIAN_BUILDERS[self.sheaf_type](
            num_nodes,
            support,
            self.d,
            left_right_idx,
            pair_index,
            cos_phase,
            sin_phase,
            **kwargs,
        )
        return builder, support

    def left_right_linear(self, x, left, right):
        r"""Apply :math:`(I_n \otimes W_1) X W_2` of Eq. 8.

        ``W_1`` mixes the ``d`` coordinates within each stalk and ``W_2`` mixes
        the channels. Both are real, so a single application covers the real
        and imaginary halves of the lifted matrix at once.

        Parameters
        ----------
        x : torch.Tensor
            Lifted features of shape ``[rows, f]``, with ``rows`` a multiple
            of ``d``.
        left : torch.nn.Linear
            The ``d x d`` weight :math:`W_1`.
        right : torch.nn.Linear
            The ``f x f`` weight :math:`W_2`.

        Returns
        -------
        torch.Tensor
            Features of shape ``[rows, f]``.
        """
        rows, channels = x.shape
        y = left(x.t().reshape(-1, self.d))
        y = y.reshape(channels, rows).t()
        return right(y)

    def sheaf_input(self, x, num_nodes: int):
        """Build the real node features handed to the sheaf learner.

        Parameters
        ----------
        x : torch.Tensor
            Lifted features of shape ``[2 * num_nodes * d, f]``.
        num_nodes : int
            Number of nodes in the batch.

        Returns
        -------
        torch.Tensor
            Real features of shape ``[num_nodes, 2 * d * f]`` when
            ``sheaf_features="unwind"``, else ``[num_nodes, d * f]``.
        """
        if self.sheaf_features == "unwind":
            return unwind_split(x, num_nodes)
        return x[: num_nodes * self.d].reshape(num_nodes, -1)

    def forward(self, x, edge_index):
        r"""Run directed sheaf diffusion over a graph.

        Iterates the Eq. 8 update

        .. math::

            X^{(t+1)} = \mathrm{diag}(1 + \varepsilon) X^{(t)} - \sigma\!\Big(
                L_N^{\tilde{\mathcal{F}}(t)}
                \big(I_n \otimes W_1^{(t)}\big) X^{(t)} W_2^{(t)} \Big),

        learning a fresh sheaf, and so a fresh Laplacian, at every step, then
        reads out through ``unwind``.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, input_dim]``.
        edge_index : torch.Tensor
            Raw arc indices of shape ``[2, num_edges]``. Directions are read
            from this tensor, so it must **not** be symmetrized by the caller.

        Returns
        -------
        torch.Tensor
            Node features of shape ``[num_nodes, output_dim]``.
        """
        num_nodes = x.size(0)
        size = num_nodes * self.d
        builder, support = self.build_builder(num_nodes, edge_index, x.dtype)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = stack_real(x.view(size, -1))

        x0 = x
        for layer in range(self.layers):
            maps_input = complex_dropout_split(
                x,
                size,
                self.dropout if layer > 0 else 0.0,
                self.training,
            )
            maps = self.sheaf_learners[layer](
                self.sheaf_input(maps_input, num_nodes), support
            )
            operator, transport = builder(maps)
            self.sheaf_learners[layer].set_L(transport)

            x = complex_dropout_split(x, size, self.dropout, self.training)
            x = self.left_right_linear(
                x, self.lin_left_weights[layer], self.lin_right_weights[layer]
            )
            x = spmm(operator[0], operator[1], 2 * size, x)
            x = complex_relu_split(x, size)

            coeff = 1 + torch.tanh(self.epsilons[layer]).tile(2 * num_nodes, 1)
            x0 = coeff * x0 - x
            x = x0

        return self.lin2(unwind_split(x, num_nodes))
