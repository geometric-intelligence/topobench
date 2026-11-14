r"""Directed Sheaf Neural Network (DSNN).

Fiorini, Aktas, Duta, Coniglio, Morerio, Del Bue and Lio,
"Sheaves Reloaded: A Directional Awakening", ICLR 2026.
https://arxiv.org/abs/2506.02842

Sheaf Neural Networks attach a ``d``-dimensional stalk to every node and edge
of a graph and glue them with learned restriction maps, diffusing features with
the sheaf Laplacian :math:`L^{\mathcal{F}} = \delta^{\top}\delta`. That
operator is blind to edge direction, because it is invariant to the orientation
chosen for each edge. DSNN removes that blindness with a single change
(Definition 1): the stalks become complex, and for each edge exactly one of the
two restriction maps is multiplied by a unit-modulus phase read off the
adjacency,

.. math::

    T^{(q)}_{uv} = \exp\!\big(i\,2\pi q\,(A_{uv} - A_{vu})\big), \qquad
    \tilde{\mathcal{F}}_{v \lhd e} =
        \mathcal{F}^{0}_{v \lhd e} T^{(q)}_{uv}.

The resulting directed sheaf Laplacian
:math:`L^{\tilde{\mathcal{F}}} = \tilde{\delta}^{*}\tilde{\delta}` is complex
Hermitian and positive semidefinite (Thm 1), its normalization
:math:`L_N = \tilde{D}^{-1/2} L^{\tilde{\mathcal{F}}} \tilde{D}^{-1/2}` has
spectrum in :math:`[0, 2]` (Thm 2), and it specializes to several known
operators: the real sheaf Laplacian when the graph is undirected or
:math:`q = 0` (Thm 3), and for a trivial sheaf the Magnetic Laplacian, at
:math:`q = 1/4` the Sign-Magnetic Laplacian (Thm 4). Theorem 5 factorizes it
as :math:`\hat{B}\hat{B}^{*}`.

The Thm 4 correspondence carries a scaling the paper states in App. C ("with a
scaling factor of 2, when needed") and is worth being precise about, since the
tests pin it: on a digon-free digraph this operator equals *twice* the
Magnetic Laplacian, because Eq. 3 sums over :math:`\Gamma(u)` while MagNet
gives a one-way arc :math:`A_{s,uv} = 1/2`. The factor disappears on an
all-digon graph. On a graph mixing digons with one-way arcs neither holds, and
the operator instead equals the Magnetic Laplacian built on the *binary*
symmetrized adjacency.

Features are diffused by Eq. 8 with the complex nonlinearity
:math:`\sigma(z) = z` if :math:`\Re(z) \ge 0` and 0 otherwise, and read out as
:math:`(\Re(X) \Vert \Im(X))`. All complex arithmetic is carried out in the
real lifting of Appendix D, so no complex tensor is ever materialized.

Three restriction-map families are available through ``sheaf_type``, matching
the three rows the paper benchmarks: ``"diag"`` (Diag-DSNN), ``"bundle"``
(O(d)-DSNN, orthogonal maps) and ``"general"`` (Gen-DSNN).

.. important::

   On an **undirected** graph :math:`A = A^{\top}`, so
   :math:`T^{(q)} = 1` identically and DSNN reduces exactly to real Neural
   Sheaf Diffusion for *every* value of ``q`` -- this is Theorem 3, and it is
   exact rather than approximate. Every graph dataset in TopoBench is
   undirected, so with the faithful ``orientation="none"`` the directional term
   provably vanishes and ``q`` has no effect. ``orientation`` other than
   ``"none"`` derives a direction from the graph structure so that the complex
   machinery is reachable; that is an extension, not part of the paper, and it
   is flagged as such wherever it appears.

Notes
-----
Deliberate departures from the paper, each discussed where it is implemented:

* Eq. 5 normalization is applied to all three families with the exact
  :math:`\tilde{D}^{-1/2}`; ``degree_shift=1.0`` reproduces the
  :math:`(\tilde{D} + I)^{-1/2}` of the reference implementation, and
  isolated nodes use a Moore-Penrose pseudo-inverse
  (:mod:`.dsnn_utils.laplacian_builders`).
* The sheaf learner consumes a real view of the complex features, since
  Definition 1 types the base maps real while the activations the paper
  searches over are undefined for complex input
  (:mod:`.dsnn_utils.discrete_models`).
* The phase sign follows the formula printed in Definition 1; its worked
  example implies the conjugate, but the two are conjugate, isospectral and
  define the same function class, so the discrepancy is immaterial
  (:mod:`.dsnn_utils.phase`).
* ``add_lp`` and ``add_hp`` are not implemented. They are axes of the paper's
  hyperparameter grid (App. F) rather than part of Eq. 8, they are absent from
  the Neural Sheaf Diffusion port in TopoBench, and they are orthogonal to
  directionality, which is what this model contributes.
* ``edge_weight`` is accepted and ignored: Definition 1 builds the phase from
  the binary adjacency.
* The reference implementation additionally learns a scalar weight per arc for
  the orthogonal family, scaling each restriction map by it and summing the
  squares into the degree block. It is on by default there, but appears
  nowhere in the paper, so ``bundle`` here keeps the plain
  :math:`\deg(u) I_d` of Eq. 3. This is a different mechanism from the
  ``edge_weight`` argument above.
"""

from torch.nn import Module

from topobench.nn.backbones.graph.dsnn_utils.discrete_models import (
    SHEAF_FEATURES,
    SHEAF_TYPES,
    DirectedSheafDiffusion,
)
from topobench.nn.backbones.graph.dsnn_utils.phase import ORIENTATIONS


class DSNNEncoder(Module):
    """Directed Sheaf Neural Network encoder.

    Parameters
    ----------
    input_dim : int
        Dimension of the input node features.
    hidden_dim : int
        Width of the hidden representation. Split into ``d`` stalk coordinates
        of ``hidden_dim // d`` channels each, so a ``hidden_dim`` not divisible
        by ``d`` is rounded down, as in the Neural Sheaf Diffusion port.
    num_layers : int, optional
        Number of diffusion layers, each learning its own sheaf. Default is 2.
    sheaf_type : str, optional
        Restriction-map family: ``"diag"`` (Diag-DSNN), ``"bundle"``
        (O(d)-DSNN) or ``"general"`` (Gen-DSNN). Default is ``"diag"``.
    d : int, optional
        Stalk dimension. The paper searches ``{2, 3, 4, 5}``. Default is 2.
    q : float, optional
        Charge of Definition 1, searched over
        ``[0, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1]``. Since ``q`` and ``1 - q``
        give conjugate operators, ``[0, 0.5]`` is enough. Has no effect on
        undirected input with ``orientation="none"`` (Thm 3). Default is 0.25.
    orientation : str, optional
        How edge directions are obtained: ``"none"`` reads them from the input
        graph, which is the faithful choice; ``"degree"`` and ``"index"``
        derive them and are extensions to the paper. Default is ``"none"``.
    dropout : float, optional
        Dropout rate on the hidden representation. Default is 0.1.
    input_dropout : float, optional
        Dropout rate on the input features. Default is 0.1.
    sheaf_act : str, optional
        Activation of the sheaf learner; the paper searches
        ``{elu, tanh, relu}``. Default is ``"tanh"``.
    sheaf_features : str, optional
        Real view of the complex features passed to the sheaf learner, either
        ``"unwind"`` or ``"real"``. Default is ``"unwind"``.
    orth : str, optional
        Retraction used by the orthogonal family, ``"cayley"`` or
        ``"matrix_exp"``. Default is ``"cayley"``.
    phase_sign : int, optional
        ``1`` for the phase printed in Definition 1, ``-1`` for its conjugate.
        Default is 1.
    normalised : bool, optional
        Whether to use the normalized operator of Eq. 5, which is the one
        Theorems 1 and 2 describe. Default is True.
    degree_shift : float, optional
        Added to the degree blocks before inversion. Default is ``0.0``,
        i.e. exactly Eq. 5.
    block_norm : bool, optional
        Whether the general family normalizes its full degree blocks exactly.
        Default is True; see :mod:`.dsnn_utils.laplacian_builders`.
    device : str, optional
        Accepted for configuration compatibility and unused; tensors follow
        the device of the input. Default is ``"cpu"``.
    **kwargs : dict
        Ignored extra arguments, so a Hydra config may carry keys meant for
        sibling models.

    Raises
    ------
    ValueError
        If ``sheaf_type``, ``orientation`` or ``sheaf_features`` is unknown, if
        ``d`` is not positive or is incompatible with the chosen family, or if
        ``hidden_dim`` is too small to give at least one channel per stalk
        coordinate.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        sheaf_type="diag",
        d=2,
        q=0.25,
        orientation="none",
        dropout=0.1,
        input_dropout=0.1,
        sheaf_act="tanh",
        sheaf_features="unwind",
        orth="cayley",
        phase_sign=1,
        normalised=True,
        degree_shift=0.0,
        block_norm=True,
        device="cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        if sheaf_type not in SHEAF_TYPES:
            raise ValueError(
                f"Unknown sheaf type: {sheaf_type!r}; expected one of "
                f"{SHEAF_TYPES}"
            )
        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"Unknown orientation {orientation!r}; expected one of "
                f"{ORIENTATIONS}"
            )
        if sheaf_features not in SHEAF_FEATURES:
            raise ValueError(
                f"Unknown sheaf_features {sheaf_features!r}; expected one of "
                f"{SHEAF_FEATURES}"
            )
        # Checked before the division below, which would otherwise raise
        # ZeroDivisionError instead of a useful message.
        if d < 1:
            raise ValueError(f"d must be at least 1, got {d}")
        if hidden_dim // d < 1:
            raise ValueError(
                f"hidden_dim={hidden_dim} is too small for d={d}: it must "
                f"leave at least one channel per stalk coordinate"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sheaf_type = sheaf_type
        self.d = d
        self.q = q
        self.orientation = orientation
        self.num_layers = num_layers
        self.device = device

        self.sheaf_config = {
            "d": d,
            "layers": num_layers,
            "hidden_channels": hidden_dim // d,
            "input_dim": input_dim,
            "output_dim": hidden_dim,
            "input_dropout": input_dropout,
            "dropout": dropout,
            "sheaf_act": sheaf_act,
            "sheaf_type": sheaf_type,
            "sheaf_features": sheaf_features,
            "q": q,
            "orientation": orientation,
            "phase_sign": phase_sign,
            "normalised": normalised,
            "degree_shift": degree_shift,
            "block_norm": block_norm,
            "orth": orth,
        }
        self.sheaf_model = DirectedSheafDiffusion(self.sheaf_config)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, sheaf_type={self.sheaf_type!r}, "
            f"d={self.d}, q={self.q}, orientation={self.orientation!r})"
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        edge_weight=None,
        batch=None,
        **kwargs,
    ):
        """Encode node features with directed sheaf diffusion.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, input_dim]``.
        edge_index : torch.Tensor
            Arc indices of shape ``[2, num_edges]``. Passed through
            **without** symmetrization, since Definition 1 reads the edge
            directions from it; the operator is then built on the symmetrized
            support internally.
        edge_attr : torch.Tensor, optional
            Unused. Default is None.
        edge_weight : torch.Tensor, optional
            Unused: the phase of Definition 1 is defined on the binary
            adjacency. Default is None.
        batch : torch.Tensor, optional
            Unused. A batch is handled as a disjoint union, which the operator
            respects because it is block diagonal across components. Default
            is None.
        **kwargs : dict
            Ignored extra arguments.

        Returns
        -------
        torch.Tensor
            Node features of shape ``[num_nodes, hidden_dim]``.
        """
        return self.sheaf_model(x, edge_index)

    def get_sheaf_model(self):
        """Return the underlying diffusion module.

        Returns
        -------
        DirectedSheafDiffusion
            The stack of diffusion layers.
        """
        return self.sheaf_model
