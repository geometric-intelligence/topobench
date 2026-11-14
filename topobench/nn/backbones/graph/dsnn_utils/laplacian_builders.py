r"""Builders for the normalized directed sheaf Laplacian (Eq. 2, 3 and 5).

The directed sheaf Laplacian is :math:`L^{\tilde{\mathcal{F}}} =
\tilde{\delta}^{*}\tilde{\delta}`, with blocks (Eq. 2 and 3)

.. math::

    L_{uv} = -\mathcal{F}^{0\top}_{u \lhd e}\mathcal{F}^{0}_{v \lhd e}
             T^{(q)}_{uv}, \qquad
    L_{uu} = \sum_{e \in \Gamma(u)} \tilde{\mathcal{F}}^{*}_{u \lhd e}
             \tilde{\mathcal{F}}_{u \lhd e},

and its normalization (Eq. 5) is :math:`L_N = \tilde{D}^{-1/2}
L^{\tilde{\mathcal{F}}} \tilde{D}^{-1/2}` with :math:`\tilde{D}` the block
diagonal of :math:`L^{\tilde{\mathcal{F}}}`.

The implementation exploits a consequence of Definition 1 that makes the whole
operator cheap. :math:`T^{(q)}_{uv}` is a **scalar** of unit modulus and
:math:`T^{(q)}` is Hermitian, so

.. math::

    L^{\tilde{\mathcal{F}}} = L^{\mathcal{F}} \odot
        \big(T^{(q)} \otimes \mathbf{1}_{d \times d}\big), \qquad
    L^{\tilde{\mathcal{F}}}_{uu} = L^{\mathcal{F}}_{uu},

because :math:`\tilde{\mathcal{F}}^{*}_{u \lhd e}
\tilde{\mathcal{F}}_{u \lhd e} = |T^{(q)}_{uv}|^{2}
\mathcal{F}^{0\top}_{u \lhd e}\mathcal{F}^{0}_{u \lhd e}` leaves the diagonal
real. In other words the operator is the *real* sheaf Laplacian with each
off-diagonal block multiplied by its edge's phase. Two practical consequences:

* the real restriction-map algebra of Neural Sheaf Diffusion is reused
  unchanged, and only two things differ: the lower blocks are scaled by the
  phase, and the upper blocks become the **conjugate** of the lower rather than
  a copy (:func:`~.laplace.flip_index` already supplies the transpose);
* the phase commutes with Eq. 5, since :math:`\tilde{D}` is real and
  block diagonal, so normalization can run on the real blocks first.

Setting :math:`q = 0`, or supplying an undirected graph, makes every phase 1
and recovers the real sheaf Laplacian exactly, which is Theorem 3.

Notes
-----
Deviations from the paper, and from the Neural Sheaf Diffusion port:
* Eq. 5 is applied to **all three** restriction-map families. That port
  normalizes only the orthogonal family and returns the unnormalized operator
  for the diagonal and general ones, so ``normalised=False`` is what reproduces
  its behaviour.
* ``degree_shift`` defaults to ``0.0``, i.e. exactly Eq. 5. That port uses
  :math:`(\tilde{D} + I)^{-1/2}`, reproduced by ``degree_shift=1.0``.
* Exact :math:`\tilde{D}^{-1/2}` is undefined for an isolated node. We use the
  Moore-Penrose convention, inverting only entries above ``eps``. The paper
  does not discuss it; sparse benchmarks make it unavoidable.
* For the general family :math:`\tilde{D}_u` is a full :math:`d \times d`
  block, so Eq. 5 needs a matrix inverse square root. ``block_norm=True`` (the
  default) computes it exactly via ``eigh``. ``block_norm=False`` substitutes
  the block's diagonal, a Jacobi approximation that is still a congruence and
  therefore preserves Hermitian-ness and positive semidefiniteness, **but does
  not satisfy Theorem 2**: measured on a small directed graph its largest
  eigenvalue reaches 2.45 at :math:`d = 2` and 4.50 at :math:`d = 4`, against
  the exact normalization's 2.000000. It is offered only as a fallback, since
  the backward pass of ``eigh`` divides by eigenvalue gaps and is therefore
  not finite on a block with repeated eigenvalues. Blocks that are entirely
  zero -- isolated nodes -- are special-cased so that the common degenerate
  case never reaches ``eigh``.

  The diagonal and orthogonal families are unaffected: their degree blocks are
  diagonal by construction (:math:`\deg(u) I_d` for the orthogonal family,
  since :math:`\mathcal{F}^{\top}\mathcal{F} = I`), so their normalization is
  exact either way.
"""

import torch
from torch import nn

from topobench.nn.backbones.graph.dsnn_utils.complex_ops import (
    lifted_index,
    lifted_value,
)
from topobench.nn.backbones.graph.dsnn_utils.laplace import (
    block_indices,
    flip_index,
    node_degree,
)
from topobench.nn.backbones.graph.dsnn_utils.orthogonal import (
    Orthogonal,
    num_orthogonal_params,
)


def block_inv_sqrt(blocks, eps: float = 1e-8):
    r"""Compute the symmetric inverse square root of each block.

    This is the :math:`\tilde{D}^{-1/2}` of Eq. 5 for a family whose degree
    blocks are full :math:`d \times d` matrices.

    Uses the Moore-Penrose convention: eigenvalues at or below ``eps`` are
    mapped to zero rather than inverted, so a singular block yields a
    pseudo-inverse instead of infinities.

    An all-zero block -- the degree block of an isolated node -- is maximally
    degenerate, and the backward pass of ``eigh`` divides by eigenvalue gaps.
    Such blocks are therefore replaced by the identity before the
    decomposition and their result is zeroed afterwards, so the common
    degenerate case never reaches ``eigh``.

    That special case covers the all-zero block, not every degenerate one: a
    block with repeated non-zero eigenvalues (``2 * I``, say) still reaches
    ``eigh``, whose backward pass would return ``NaN``. Following the
    reference implementation, :meth:`~.DirectedLaplacianBuilder.normalise`
    keeps that unreachable by detaching this result and by jittering the
    diagonal during training.

    Parameters
    ----------
    blocks : torch.Tensor
        Symmetric positive semidefinite blocks of shape ``[n, d, d]``.
    eps : float, optional
        Threshold below which an eigenvalue is treated as zero. Default is
        ``1e-8``.

    Returns
    -------
    torch.Tensor
        Blocks of shape ``[n, d, d]`` with :math:`B^{-1/2}` for each input.
    """
    identity = torch.eye(
        blocks.size(-1), dtype=blocks.dtype, device=blocks.device
    )
    empty = (blocks.abs().amax(dim=-1).amax(dim=-1) <= eps).view(-1, 1, 1)
    safe = torch.where(empty, identity.expand_as(blocks), blocks)
    values, vectors = torch.linalg.eigh(safe)
    inv = torch.where(
        values > eps, values.clamp(min=eps).pow(-0.5), torch.zeros_like(values)
    )
    result = vectors @ torch.diag_embed(inv) @ vectors.transpose(-1, -2)
    return torch.where(empty, torch.zeros_like(result), result)


class DirectedLaplacianBuilder(nn.Module):
    r"""Assemble the directed sheaf Laplacian from predicted maps.

    Subclasses supply :meth:`restriction_blocks`, which turns the learner's
    output into the real diagonal and off-diagonal blocks of Eq. 2 and 3. This
    base class applies Eq. 5, attaches the phase, mirrors the result as a
    conjugate transpose, and emits the real lifting of Appendix D.

    Parameters
    ----------
    size : int
        Number of nodes, so the complex operator is ``size * d`` square.
    edge_index : torch.Tensor
        Symmetrized arc support of shape ``[2, 2 * num_pairs]``.
    d : int
        Stalk dimension.
    left_right_idx : torch.Tensor
        Support rows of the arcs ``(a, b)`` and ``(b, a)``, shape
        ``[2, num_pairs]``.
    pair_index : torch.Tensor
        Node pairs ``(a, b)`` with ``a < b``, shape ``[2, num_pairs]``.
    cos_phase : torch.Tensor
        Real part of the per-pair phase, shape ``[num_pairs]``.
    sin_phase : torch.Tensor
        Imaginary part of the per-pair phase, shape ``[num_pairs]``.
    normalised : bool, optional
        Whether to apply Eq. 5. Default is True.
    degree_shift : float, optional
        Added to the degree blocks before inversion. Default is ``0.0``,
        i.e. exactly Eq. 5.
    block_norm : bool, optional
        Whether to normalize full degree blocks exactly, via ``eigh``, rather
        than by their diagonal. Default is True; the Jacobi alternative does
        not satisfy Theorem 2. Only the general family has full degree blocks,
        so this is inert for the other two.
    eps : float, optional
        Threshold below which a degree entry is treated as zero. Default is
        ``1e-8``.
    training : bool, optional
        Whether the owning module is training, which enables the degree-block
        jitter of :func:`block_inv_sqrt`. Default is False, so the operator is
        deterministic unless a caller asks otherwise.
    jitter : float, optional
        Half-width of that jitter. Default is ``0.001``, the value used by the
        reference implementation.

    Attributes
    ----------
    offdiag_diagonal : bool
        Whether off-diagonal blocks carry only their ``d`` diagonal entries.
    nodediag_diagonal : bool
        Whether node-diagonal blocks carry only their ``d`` diagonal entries.
    """

    offdiag_diagonal: bool = False
    nodediag_diagonal: bool = False

    def __init__(
        self,
        size,
        edge_index,
        d,
        left_right_idx,
        pair_index,
        cos_phase,
        sin_phase,
        *,
        normalised: bool = True,
        degree_shift: float = 0.0,
        block_norm: bool = True,
        eps: float = 1e-8,
        training: bool = False,
        jitter: float = 0.001,
    ) -> None:
        super().__init__()
        self.size = size
        self.d = d
        self.edge_index = edge_index
        self.left_right_idx = left_right_idx
        self.pair_index = pair_index
        self.cos_phase = cos_phase
        self.sin_phase = sin_phase
        self.normalised = normalised
        self.degree_shift = degree_shift
        self.block_norm = block_norm
        self.eps = eps
        self.training = training
        self.jitter = jitter
        self.num_pairs = pair_index.size(1)
        self.deg = node_degree(edge_index, size)

        diag_indices, _ = block_indices(
            size, pair_index, d, diagonal=self.nodediag_diagonal
        )
        _, off_indices = block_indices(
            size, pair_index, d, diagonal=self.offdiag_diagonal
        )
        self.diag_indices = diag_indices
        self.tril_indices = off_indices
        self.triu_indices = flip_index(off_indices)
        self.real_index = torch.cat(
            [self.tril_indices, self.triu_indices, self.diag_indices], dim=1
        )

        # The imaginary pattern is chosen from the phase, which carries no
        # gradient, and never from the predicted values. Selecting on
        # ``value != 0`` would drop entries from the sparse pattern whenever a
        # predicted map happens to be zero -- routine with ``sheaf_act="relu"``
        # -- and would then silently zero their gradients even though the
        # derivative with respect to those values is non-zero.
        per_block = d if self.offdiag_diagonal else d * d
        self.per_block = per_block
        active = torch.nonzero(sin_phase != 0, as_tuple=False).flatten()
        self.active_pairs = active
        offsets = torch.arange(per_block, device=pair_index.device)
        selection = (
            active.view(-1, 1) * per_block + offsets.view(1, -1)
        ).reshape(-1)
        self.imag_index = torch.cat(
            [
                self.tril_indices[:, selection],
                self.triu_indices[:, selection],
            ],
            dim=1,
        )
        self.lifted_index = lifted_index(
            self.real_index, self.imag_index, size * d
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size}, d={self.d}, "
            f"num_pairs={self.num_pairs}, normalised={self.normalised})"
        )

    def restriction_blocks(self, maps):
        """Turn predicted parameters into the real blocks of Eq. 2 and 3.

        Parameters
        ----------
        maps : torch.Tensor
            Predicted restriction-map parameters, one row per support arc.

        Returns
        -------
        diag : torch.Tensor
            Node-diagonal blocks, shape ``[size, d]`` or ``[size, d, d]``.
        tril : torch.Tensor
            Off-diagonal blocks for the pairs ``(a, b)``, already carrying the
            minus sign of Eq. 2; shape ``[num_pairs, d]`` or
            ``[num_pairs, d, d]``.

        Raises
        ------
        NotImplementedError
            Always, in this abstract base class.
        """
        raise NotImplementedError

    def normalise(self, diag, tril):
        """Apply the normalization of Eq. 5 to the real blocks.

        Parameters
        ----------
        diag : torch.Tensor
            Node-diagonal blocks, shape ``[size, d]`` or ``[size, d, d]``.
        tril : torch.Tensor
            Off-diagonal blocks, shape ``[num_pairs, d]`` or
            ``[num_pairs, d, d]``.

        Returns
        -------
        diag : torch.Tensor
            Normalized node-diagonal blocks, same shape as the input.
        tril : torch.Tensor
            Normalized off-diagonal blocks, same shape as the input.
        """
        head, tail = self.pair_index[0].long(), self.pair_index[1].long()
        identity = torch.eye(self.d, dtype=diag.dtype, device=diag.device)

        if diag.dim() == 3 and self.block_norm:
            shifted = diag + self.degree_shift * identity
            if self.training:
                # The reference jitters the degree blocks so that a block
                # never carries exactly repeated eigenvalues: the backward
                # pass of ``eigh`` divides by eigenvalue gaps and is not
                # finite on a tie.
                shifted = shifted + self.jitter * identity * torch.empty_like(
                    shifted[..., 0, 0]
                ).uniform_(-1.0, 1.0).view(-1, 1, 1)
            # Detached for the same reason, and likewise following the
            # reference: D^-1/2 is treated as a constant, so no gradient is
            # routed through the eigendecomposition at all.
            inv = block_inv_sqrt(shifted, eps=self.eps).detach()
            tril = inv[head] @ tril @ inv[tail]
            diag = inv @ diag @ inv
            return diag, tril

        scale = (
            diag if diag.dim() == 2 else torch.diagonal(diag, dim1=-2, dim2=-1)
        )
        scale = scale + self.degree_shift
        inv = torch.where(
            scale > self.eps,
            scale.clamp(min=self.eps).pow(-0.5),
            torch.zeros_like(scale),
        )

        if tril.dim() == 2:
            tril = inv[head] * tril * inv[tail]
        else:
            tril = inv[head].unsqueeze(-1) * tril * inv[tail].unsqueeze(-2)

        if diag.dim() == 2:
            diag = inv * diag * inv
        else:
            diag = inv.unsqueeze(-1) * diag * inv.unsqueeze(-2)
        return diag, tril

    def hermitian_parts(self, maps):
        r"""Build the sparse real and imaginary parts of the operator.

        The upper triangle reuses the lower triangle's values at the
        transposed indices, negating the imaginary half, which is exactly the
        conjugate transpose demanded by Eq. 2.

        Parameters
        ----------
        maps : torch.Tensor
            Predicted restriction-map parameters, one row per support arc.

        Returns
        -------
        real_index : torch.Tensor
            Indices of the real part, shape ``[2, nnz_real]``.
        real_value : torch.Tensor
            Values of the real part, shape ``[nnz_real]``.
        imag_index : torch.Tensor
            Indices of the imaginary part, shape ``[2, nnz_imag]``.
        imag_value : torch.Tensor
            Values of the imaginary part, shape ``[nnz_imag]``.
        saved_tril : torch.Tensor
            Detached copy of the unphased off-diagonal blocks.
        """
        diag, tril = self.restriction_blocks(maps)
        saved_tril = tril.detach().clone()
        if self.normalised:
            diag, tril = self.normalise(diag, tril)

        trailing = (1,) * (tril.dim() - 1)
        cos = self.cos_phase.to(tril.dtype).view(-1, *trailing)
        real_block = (tril * cos).reshape(-1)
        real_value = torch.cat([real_block, real_block, diag.reshape(-1)])

        active = self.active_pairs
        sin = self.sin_phase.to(tril.dtype)[active].view(-1, *trailing)
        imag_block = (tril[active] * sin).reshape(-1)
        imag_value = torch.cat([imag_block, -imag_block])

        return (
            self.real_index,
            real_value,
            self.imag_index,
            imag_value,
            saved_tril,
        )

    def forward(self, maps):
        """Assemble the operator in the real lifting of Appendix D.

        Parameters
        ----------
        maps : torch.Tensor
            Predicted restriction-map parameters, one row per support arc.

        Returns
        -------
        operator : tuple of torch.Tensor
            The pair ``(index, value)`` of a real sparse matrix of side
            ``2 * size * d``.
        saved_tril : torch.Tensor
            Detached copy of the unphased off-diagonal blocks.
        """
        _, real_value, _, imag_value, saved_tril = self.hermitian_parts(maps)
        return (
            self.lifted_index,
            lifted_value(real_value, imag_value),
        ), saved_tril


class DirectedDiagLaplacianBuilder(DirectedLaplacianBuilder):
    """Directed sheaf Laplacian with diagonal restriction maps.

    The Diag-DSNN family: the learner predicts ``d`` values per arc, and both
    the off-diagonal and node-diagonal blocks stay diagonal.

    Parameters
    ----------
    *args : tuple
        Positional arguments of :class:`DirectedLaplacianBuilder`.
    **kwargs : dict
        Keyword arguments of :class:`DirectedLaplacianBuilder`.
    """

    offdiag_diagonal = True
    nodediag_diagonal = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def restriction_blocks(self, maps):
        """Build diagonal blocks from ``d`` predicted values per arc.

        Parameters
        ----------
        maps : torch.Tensor
            Predicted parameters of shape ``[2 * num_pairs, d]``.

        Returns
        -------
        diag : torch.Tensor
            Node-diagonal blocks of shape ``[size, d]``.
        tril : torch.Tensor
            Off-diagonal blocks of shape ``[num_pairs, d]``.

        Raises
        ------
        ValueError
            If ``maps`` is not of shape ``[2 * num_pairs, d]``.
        """
        if maps.dim() != 2 or maps.size(1) != self.d:
            raise ValueError(
                f"Diagonal maps must have shape [num_arcs, {self.d}], got "
                f"{tuple(maps.shape)}"
            )
        left, right = self.left_right_idx[0], self.left_right_idx[1]
        tril = -maps.index_select(0, left) * maps.index_select(0, right)
        diag = torch.zeros(
            self.size, self.d, dtype=maps.dtype, device=maps.device
        ).index_add(0, self.edge_index[0].long(), maps**2)
        return diag, tril


class DirectedBundleLaplacianBuilder(DirectedLaplacianBuilder):
    r"""Directed sheaf Laplacian with orthogonal restriction maps.

    The O(d)-DSNN family. Because :math:`\mathcal{F}^{\top}\mathcal{F} = I`,
    the node-diagonal block of Eq. 3 is exactly :math:`\deg(u) I_d`.

    Parameters
    ----------
    *args : tuple
        Positional arguments of :class:`DirectedLaplacianBuilder`.
    orth_map : str, optional
        Retraction used to build the orthogonal maps. Default is
        ``"cayley"``.
    **kwargs : dict
        Keyword arguments of :class:`DirectedLaplacianBuilder`.
    """

    offdiag_diagonal = False
    nodediag_diagonal = True

    def __init__(self, *args, orth_map: str = "cayley", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map

    def restriction_blocks(self, maps):
        """Build blocks from orthogonal restriction maps.

        Parameters
        ----------
        maps : torch.Tensor
            Predicted parameters of shape
            ``[2 * num_pairs, d * (d - 1) // 2]``.

        Returns
        -------
        diag : torch.Tensor
            Node-diagonal blocks of shape ``[size, d]``, equal to the degree.
        tril : torch.Tensor
            Off-diagonal blocks of shape ``[num_pairs, d, d]``.

        Raises
        ------
        ValueError
            If ``maps`` does not have ``d * (d - 1) // 2`` columns.
        """
        expected = num_orthogonal_params(self.d)
        if maps.dim() != 2 or maps.size(1) != expected:
            raise ValueError(
                f"Orthogonal maps must have shape [num_arcs, {expected}], "
                f"got {tuple(maps.shape)}"
            )
        blocks = self.orth_transform(maps)
        left, right = self.left_right_idx[0], self.left_right_idx[1]
        tril = -torch.bmm(
            blocks.index_select(0, left).transpose(-1, -2),
            blocks.index_select(0, right),
        )
        diag = self.deg.to(maps.dtype).unsqueeze(-1).expand(self.size, self.d)
        return diag, tril


class DirectedGeneralLaplacianBuilder(DirectedLaplacianBuilder):
    """Directed sheaf Laplacian with unconstrained restriction maps.

    The Gen-DSNN family: the learner predicts a full ``d x d`` matrix per arc.

    Parameters
    ----------
    *args : tuple
        Positional arguments of :class:`DirectedLaplacianBuilder`.
    **kwargs : dict
        Keyword arguments of :class:`DirectedLaplacianBuilder`.
    """

    offdiag_diagonal = False
    nodediag_diagonal = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def restriction_blocks(self, maps):
        """Build blocks from general ``d x d`` restriction maps.

        Parameters
        ----------
        maps : torch.Tensor
            Predicted parameters of shape ``[2 * num_pairs, d, d]``.

        Returns
        -------
        diag : torch.Tensor
            Node-diagonal blocks of shape ``[size, d, d]``.
        tril : torch.Tensor
            Off-diagonal blocks of shape ``[num_pairs, d, d]``.

        Raises
        ------
        ValueError
            If ``maps`` is not of shape ``[2 * num_pairs, d, d]``.
        """
        if maps.dim() != 3 or maps.shape[1:] != (self.d, self.d):
            raise ValueError(
                f"General maps must have shape [num_arcs, {self.d}, "
                f"{self.d}], got {tuple(maps.shape)}"
            )
        left, right = self.left_right_idx[0], self.left_right_idx[1]
        tril = -torch.bmm(
            maps.index_select(0, left).transpose(-1, -2),
            maps.index_select(0, right),
        )
        outer = torch.bmm(maps.transpose(-1, -2), maps)
        diag = torch.zeros(
            self.size, self.d, self.d, dtype=maps.dtype, device=maps.device
        ).index_add(0, self.edge_index[0].long(), outer)
        return diag, tril


LAPLACIAN_BUILDERS = {
    "diag": DirectedDiagLaplacianBuilder,
    "bundle": DirectedBundleLaplacianBuilder,
    "general": DirectedGeneralLaplacianBuilder,
}
