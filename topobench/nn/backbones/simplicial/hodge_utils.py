r"""Shared Hodge-Laplacian helpers for the simplicial spectral backbones.

Both :class:`PolynomialFilterTNN` and :class:`FilterBankTNN` turn a
(sparse) Hodge Laplacian into a closure ``L_apply : h -> L h`` that the
reused polynomial-filter bases / channels consume. The bases define
their argument relative to ``L_apply`` (Chebyshev evaluates ``T_k`` at
``L_apply - I``), so a ``[0, 2]``-spectrum operator is all they need;
``build_hodge_laplacian_apply(..., 'rescale')`` provides that via
``2 L / bound``, where ``bound`` is a deterministic Gershgorin upper
bound on ``λmax`` (see :func:`spectral_upper_bound`).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def spmm(operator: Tensor, h: Tensor) -> Tensor:
    r"""Multiply ``operator @ h``, dispatching on sparse vs dense layout.

    Parameters
    ----------
    operator : Tensor
        A ``[n, n]`` operator, sparse (COO/CSR) or dense.
    h : Tensor
        A ``[n, F]`` dense signal.

    Returns
    -------
    Tensor
        ``operator @ h``.
    """
    if operator.layout != torch.strided:
        return torch.sparse.mm(operator, h)
    return operator @ h


@torch.no_grad()
def spectral_upper_bound(operator: Tensor, eps: float = 1e-6) -> Tensor:
    r"""Deterministic upper bound on the largest eigenvalue of a symmetric operator.

    The Gershgorin disc bound ``λmax <= max_i Σ_j |L_ij|`` (the maximum
    absolute row sum). Unlike power iteration this is **deterministic** (no
    RNG) and a guaranteed **upper** bound, so rescaling by ``2 / bound``
    keeps the spectrum inside ``[0, 2]`` (hence the reused Chebyshev /
    Jacobi bases inside ``[-1, 1]``), which an under-estimating power
    iteration could not guarantee. It is used only to rescale the Hodge
    spectrum, so no gradient is tracked. The bound can over-estimate
    ``λmax``, giving a looser but always-safe rescale.

    Parameters
    ----------
    operator : Tensor
        A symmetric ``[n, n]`` operator (a Hodge Laplacian), sparse or dense.
    eps : float, optional
        Lower clamp guarding an empty or zero operator against division by
        zero. Defaults to ``1e-6``.

    Returns
    -------
    Tensor
        Scalar upper bound on ``λmax``.
    """
    n = operator.shape[0]
    if n == 0:
        return torch.tensor(eps, device=operator.device)
    if operator.layout == torch.strided:
        row_sums = operator.abs().sum(dim=1)
    else:
        op = operator.coalesce() if operator.is_sparse else operator
        ones = torch.ones(n, 1, dtype=operator.dtype, device=operator.device)
        row_sums = spmm(op.abs(), ones).squeeze(-1)
    return row_sums.max().clamp_min(eps)


def build_hodge_laplacian_apply(
    laplacian: Tensor, laplacian_norm: str = "raw"
) -> Callable[[Tensor], Tensor]:
    r"""Build a closure ``h -> L h`` for a Hodge Laplacian.

    Parameters
    ----------
    laplacian : Tensor
        A (sparse or dense) ``[n, n]`` Hodge Laplacian.
    laplacian_norm : {'raw', 'rescale'}, optional
        ``'raw'`` applies ``L`` directly; ``'rescale'`` applies
        ``(2 / bound) L`` (``bound`` a deterministic Gershgorin upper
        bound on ``λmax``) so the spectrum lands in ``[0, 2]``, the domain
        the Chebyshev / Jacobi bases expect before their internal shift.
        The rescale uses the batch-global bound: a complex's rescale
        depends on its batchmates (a documented limitation; a fully
        batch-invariant rescale would need a normalized Hodge Laplacian).
        Defaults to ``'raw'``.

    Returns
    -------
    Callable[[Tensor], Tensor]
        The closure ``h -> L h`` (or ``h -> (2 / λmax) L h``).
    """
    if laplacian_norm not in {"raw", "rescale"}:
        raise ValueError(
            f"laplacian_norm must be 'raw' or 'rescale'; got {laplacian_norm!r}"
        )
    if laplacian_norm == "rescale":
        lam = spectral_upper_bound(laplacian)

        def apply(h: Tensor) -> Tensor:
            r"""Apply ``(2 / λmax) L`` to ``h``.

            Parameters
            ----------
            h : Tensor
                Signal to filter.

            Returns
            -------
            Tensor
                ``(2 / λmax) L h`` (spectrum in ``[0, 2]``).
            """
            return (2.0 / lam) * spmm(laplacian, h)

        return apply

    def apply(h: Tensor) -> Tensor:
        r"""Apply the raw ``L`` to ``h``.

        Parameters
        ----------
        h : Tensor
            Signal to filter.

        Returns
        -------
        Tensor
            ``L h``.
        """
        return spmm(laplacian, h)

    return apply
