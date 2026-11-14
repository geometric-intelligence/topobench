r"""The directional phase of a directed cellular sheaf (Definition 1).

Definition 1 attaches to a directed graph the Hermitian matrix

.. math::

    T^{(q)} = \exp\!\big(i\,2\pi q\,(A - A^{\top})\big),

applied entrywise to the **binary** adjacency :math:`A`, and multiplies one of
the two restriction maps of each edge by it:
:math:`\tilde{\mathcal{F}}_{v \lhd e} = \mathcal{F}^{0}_{v \lhd e} T^{(q)}_{uv}`.
Since :math:`A_{uv} - A_{vu} \in \{-1, 0, +1\}`, the phase of an edge takes one
of three values, and only its sign per pair is needed:

* a reciprocal pair (a digon, :math:`A_{uv} = A_{vu} = 1`) gives sign 0 and
  phase 1, so it behaves exactly like an undirected edge;
* a one-way arc gives sign :math:`\pm 1` and phase :math:`e^{\pm i 2\pi q}`.

The phase is stored as a real ``(cos, sin)`` pair rather than a complex tensor,
so the whole operator can be assembled in the real lifting of Appendix D.

Notes
-----
**Sign convention.**
Definition 1 prints :math:`\exp(+i\,2\pi q(\cdot))`, but its own worked example
(:math:`q = 1/4` yielding :math:`-i`) requires the conjugate. The choice is
immaterial to the model class: :math:`T^{(q)}` has integer exponents, so
:math:`T^{(1-q)} = \overline{T^{(q)}}`, hence
:math:`L(1-q) = \overline{L(q)}`; and conjugation is absorbed downstream
because the weights of Eq. 6 are real, :math:`\Im(X^{0}) = 0`, the
nonlinearity gates on :math:`\Re(z)` only (so
:math:`\overline{\sigma(z)} = \sigma(\bar{z})`), and the readout is a learned
real map on :math:`(\Re(X) \Vert \Im(X))` that can negate its second half. The
two conventions are therefore conjugate, isospectral, and define the same
function class. ``phase_sign`` exposes both; the default ``+1`` follows the
printed formula and matches MagNet's
:math:`H^{(q)} = A_s \odot \exp(i 2\pi q (A - A^{\top}))`. Two consequences
when searching: :math:`q` and :math:`1 - q` are equivalent, so
:math:`q \in [0, 1/2]` suffices; and :math:`q = 0` gives an exactly real
operator, while :math:`q = 1/2` gives one real only up to rounding, since
:math:`\sin(\pm\pi)` evaluates to :math:`\mp 1.2 \times 10^{-16}` rather than
to zero.

**Induced orientations.**
``orientation`` other than ``"none"`` is **not part of the paper**, which
assumes genuinely directed input. It exists because every graph benchmark in
TopoBench is undirected, and on undirected input :math:`T^{(q)} = 1`
identically for every :math:`q` (Theorem 3), leaving the complex machinery
unreachable. Each mode derives a direction from a node potential
:math:`\pi`, orienting :math:`a \to b` when :math:`\pi(a) < \pi(b)` and
leaving the edge undirected on ties.

A potential that is *exactly* a graph distance is deliberately **not** offered.
BFS layering forces :math:`|\pi(b) - \pi(a)| \le 1` on every edge, so
:math:`T_{ab} = \overline{u_a} u_b` with :math:`u_x = e^{i 2\pi q \pi(x)}` and
therefore :math:`L = U^{*} L^{\mathcal{F}} U` is unitarily equivalent to the
real sheaf Laplacian: a pure gauge transformation, isospectral with plain
Neural Sheaf Diffusion, differing only through the basis dependence of a
nonlinearity that gates on :math:`\Re(z)`. What distinguishes a genuinely
directional phase is non-zero holonomy around cycles. On a tree every
orientation is a gauge, so no induced orientation can help there -- worth
keeping in mind on sparse benchmarks.
"""

import torch

from topobench.nn.backbones.graph.dsnn_utils.laplace import (
    has_directed_edge,
    node_degree,
)

ORIENTATIONS = ("none", "degree", "index")


def directed_pair_sign(edge_index, pair_index, num_nodes: int):
    r"""Read :math:`A_{ab} - A_{ba}` off the binary adjacency, per pair.

    This is the faithful path of Definition 1: the sign comes from the raw
    input graph, before any symmetrization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Raw arc indices of shape ``[2, num_edges]``, as given by the dataset.
    pair_index : torch.Tensor
        Node pairs ``(a, b)`` with ``a < b``, of shape ``[2, num_pairs]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Signs in ``{-1, 0, +1}`` of shape ``[num_pairs]``, dtype
        ``torch.int64``. Zero marks a reciprocal or absent direction.
    """
    forward = has_directed_edge(edge_index, num_nodes, pair_index)
    backward = has_directed_edge(edge_index, num_nodes, pair_index.flip(0))
    return forward.long() - backward.long()


def node_potential(edge_index, num_nodes: int, mode: str):
    """Build the lexicographic node potential driving an induced orientation.

    Parameters
    ----------
    edge_index : torch.Tensor
        A symmetric arc support of shape ``[2, num_edges]``.
    num_nodes : int
        Number of nodes in the graph.
    mode : str
        Either ``"degree"`` or ``"index"``.

    Returns
    -------
    torch.Tensor
        Potential of shape ``[num_nodes, num_keys]``, compared column by
        column from the left.

    Raises
    ------
    ValueError
        If ``mode`` is not an induced orientation mode.
    """
    device = edge_index.device
    if mode == "index":
        keys = [torch.arange(num_nodes, device=device)]
    elif mode == "degree":
        # Degree alone leaves many ties on near-regular graphs, and breaking
        # them by node index would forfeit permutation equivariance. The
        # neighbourhood degree sum is a second isomorphism-invariant key that
        # separates many of those ties; whatever still ties stays undirected.
        degree = node_degree(edge_index, num_nodes, dtype=torch.long)
        neighbour_sum = torch.zeros_like(degree).index_add(
            0, edge_index[0].long(), degree[edge_index[1].long()]
        )
        keys = [degree, neighbour_sum]
    else:
        raise ValueError(
            f"Unknown induced orientation {mode!r}; expected one of "
            f"{tuple(m for m in ORIENTATIONS if m != 'none')}"
        )
    return torch.stack(keys, dim=1)


def induced_pair_sign(edge_index, pair_index, num_nodes: int, mode: str):
    """Derive a per-pair direction from a node potential.

    Parameters
    ----------
    edge_index : torch.Tensor
        A symmetric arc support of shape ``[2, num_edges]``.
    pair_index : torch.Tensor
        Node pairs ``(a, b)`` with ``a < b``, of shape ``[2, num_pairs]``.
    num_nodes : int
        Number of nodes in the graph.
    mode : str
        Either ``"degree"`` or ``"index"``.

    Returns
    -------
    torch.Tensor
        Signs in ``{-1, 0, +1}`` of shape ``[num_pairs]``, dtype
        ``torch.int64``. Zero marks a tie, leaving that edge undirected.
    """
    potential = node_potential(edge_index, num_nodes, mode)
    if pair_index.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=pair_index.device)
    head, tail = pair_index[0].long(), pair_index[1].long()
    sign = torch.zeros(
        pair_index.size(1), dtype=torch.long, device=pair_index.device
    )
    for key in range(potential.size(1)):
        column = potential[:, key]
        step = torch.sign(column[tail] - column[head]).long()
        sign = torch.where(sign == 0, step, sign)
    return sign


def pair_sign(
    raw_edge_index,
    support_edge_index,
    pair_index,
    num_nodes: int,
    orientation: str = "none",
):
    """Dispatch to the faithful or an induced per-pair direction.

    Parameters
    ----------
    raw_edge_index : torch.Tensor
        Raw arc indices of shape ``[2, num_edges]`` as given by the dataset,
        used only by ``orientation="none"``.
    support_edge_index : torch.Tensor
        The symmetrized support of shape ``[2, 2 * num_pairs]``.
    pair_index : torch.Tensor
        Node pairs ``(a, b)`` with ``a < b``, of shape ``[2, num_pairs]``.
    num_nodes : int
        Number of nodes in the graph.
    orientation : str, optional
        One of ``ORIENTATIONS``. Default is ``"none"``, the faithful path.

    Returns
    -------
    torch.Tensor
        Signs in ``{-1, 0, +1}`` of shape ``[num_pairs]``.

    Raises
    ------
    ValueError
        If ``orientation`` is not one of ``ORIENTATIONS``.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(
            f"Unknown orientation {orientation!r}; expected one of "
            f"{ORIENTATIONS}"
        )
    if orientation == "none":
        return directed_pair_sign(raw_edge_index, pair_index, num_nodes)
    return induced_pair_sign(
        support_edge_index, pair_index, num_nodes, orientation
    )


def phase_from_sign(sign, q: float, phase_sign: int = 1, dtype=torch.float32):
    r"""Turn per-pair signs into the real and imaginary parts of the phase.

    Computes :math:`T^{(q)}_{ab} = \exp(i\,s\,\mathrm{phase\_sign}\,2\pi q)`
    for :math:`s \in \{-1, 0, +1\}`.

    Parameters
    ----------
    sign : torch.Tensor
        Per-pair signs of shape ``[num_pairs]``.
    q : float
        The charge of Definition 1.
    phase_sign : int, optional
        ``+1`` for the convention printed in Definition 1, ``-1`` for its
        conjugate. Default is 1.
    dtype : torch.dtype, optional
        Floating dtype of the result. Default is ``torch.float32``.

    Returns
    -------
    cos_phase : torch.Tensor
        Real part, shape ``[num_pairs]``.
    sin_phase : torch.Tensor
        Imaginary part, shape ``[num_pairs]``.

    Raises
    ------
    ValueError
        If ``phase_sign`` is neither ``1`` nor ``-1``.
    """
    if phase_sign not in (1, -1):
        raise ValueError(f"phase_sign must be 1 or -1, got {phase_sign}")
    angle = sign.to(dtype) * (phase_sign * 2.0 * torch.pi * q)
    return torch.cos(angle), torch.sin(angle)
