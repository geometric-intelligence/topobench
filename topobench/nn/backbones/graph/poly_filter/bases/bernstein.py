r"""Bernstein polynomial basis.

The Bernstein basis is the one variable basis in Liao Appendix B that is
**not** an orthogonal polynomial family: the Bernstein polynomials form a
partition of unity (the Bezier basis), chosen for non-negativity and
interpretable filter shaping rather than orthogonality. Consequently it
has no three-term recurrence in ``k`` and is given in closed form
(Liao Appendix B, "Bernstein"):

.. math::

    g(\tilde L; \theta) = \sum_{k=0}^{K} \frac{\theta_k}{2^K}\,
        T^{(k)}(\tilde L),
    \qquad
    T^{(k)}(\tilde L) = \binom{K}{k}\,(2I - \tilde L)^{K-k}\,\tilde L^{k} .

**How it fits the basis protocol.**
The protocol passes ``(u_prev, u_prev_prev, L_apply, signal, k)`` to every
basis; signal-independent bases already ignore ``signal``. Bernstein
additionally ignores ``u_prev`` and ``u_prev_prev`` (there is no
recurrence) and builds each basis vector directly from the input signal:

.. math::

    u_k = (2I - \tilde L)^{K-k}\,\tilde L^{k}\, x ,
    \qquad u_0 = (2I - \tilde L)^{K}\, x ,

while the binomial / normalization factor ``\binom{K}{k}/2^K`` is folded
into the coefficients via :meth:`effective_thetas`. The backbone's
accumulator ``y = Σ_k θ_eff[k] u_k`` is then exactly ``g(\tilde L; θ) x``.

Because every ``u_k`` rebuilds two matrix powers, a degree-``K`` Bernstein
filter costs ``O(K^2 m F)`` matrix-vector products, the only basis in the
registry that is not ``O(K m F)`` -- a direct fingerprint of *not* having
the orthogonal-polynomial (tridiagonal Jacobi) structure that gives the
other bases their three-term recurrence.

A useful sanity check: with all ``θ_k = 1`` the partition-of-unity
property makes the filter the identity,
``Σ_k \binom{K}{k}/2^K (2I-\tilde L)^{K-k}\tilde L^k = (1/2^K)(2I)^K = I``.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Bernstein" entry of the
Variable Basis table (complexity ``O(K^2 m F)``).

He et al. (2021) *BernNet: Learning Arbitrary Graph Spectral Filters via
Bernstein Approximation* (NeurIPS, arXiv:2106.10994) -- primary reference
for BernNet. The non-negativity of the Bernstein coefficients yields
interpretable, constraint-respecting spectral filters.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


class Bernstein(Basis):
    r"""Bernstein (Bezier) basis ``T_k(L̃) = C(K,k) (2I - L̃)^(K-k) L̃^k``.

    Closed-form, signal-independent, no learnable parameters of its own
    (the backbone owns ``θ``). The binomial and ``1/2^K`` normalization
    are applied through :meth:`effective_thetas`; the basis vectors
    themselves are ``(2I - L̃)^(K-k) L̃^k x``.

    Parameters
    ----------
    K : int
        Polynomial degree. **Must match the backbone's** ``K`` -- the
        binomial coefficients and the matrix-power exponents are baked in
        at construction. In Hydra configs, set
        ``model.backbone.basis.K = ${model.backbone.K}`` to keep them in
        sync.

    Notes
    -----
    Unlike every other registered basis, Bernstein is not a three-term
    recurrence; it is a closed-form expansion with ``O(K^2 m F)``
    complexity. See the module docstring for why (it is the one
    non-orthogonal basis in Liao Appendix B).
    """

    def __init__(self, K: int):
        super().__init__()
        if K < 0:
            raise ValueError(f"Bernstein K must be >= 0, got {K}")
        self.K = K
        # Coefficient prefactors C(K,k) / 2^K, folded into the effective
        # thetas so the basis vectors stay the bare (2I-L̃)^(K-k) L̃^k.
        coeffs = torch.tensor(
            [math.comb(K, k) / 2.0**K for k in range(K + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("_bern_coeff", coeffs)

    def effective_thetas(self, backbone_theta: Tensor) -> Tensor:
        r"""Scale ``θ`` by the Bernstein prefactors ``C(K,k) / 2^K``.

        The backbone's ``θ`` remain the learnable filter coefficients
        (unlike ChebNetII, which replaces them); they are merely rescaled
        by the binomial / normalization factor that closes the gap
        between the bare basis vectors and Liao's ``T^{(k)}``.

        Parameters
        ----------
        backbone_theta : Tensor, shape ``[K + 1]``
            The backbone's learnable ``θ`` vector.

        Returns
        -------
        Tensor, shape ``[K + 1]``
            ``θ_k · C(K,k) / 2^K``.

        Raises
        ------
        ValueError
            If ``backbone_theta`` has size other than ``K + 1`` (a basis
            ``K`` / backbone ``K`` mismatch).
        """
        if backbone_theta.size(0) != self.K + 1:
            raise ValueError(
                f"Bernstein was constructed with K={self.K}, but the "
                f"backbone passed θ of size {backbone_theta.size(0)}. Set "
                f"model.backbone.basis.K = ${{model.backbone.K}} in the "
                f"Hydra config so they stay in sync."
            )
        return backbone_theta * self._bern_coeff

    def init(self, x: Tensor, L_apply: LaplacianApply) -> Tensor:
        r"""Return ``u_0 = (2I - L̃)^K x``.

        Parameters
        ----------
        x : Tensor, shape ``[N, F]``
            Input features.
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_0``.
        """
        u = x
        for _ in range(self.K):
            u = 2.0 * u - L_apply(u)  # (2I - L̃) u
        return u

    def forward(
        self,
        u_prev: Tensor,  # unused -- no recurrence
        u_prev_prev: Tensor | None,  # unused -- no recurrence
        L_apply: LaplacianApply,
        signal: Tensor,
        k: int,
    ) -> Tensor:
        r"""Build ``u_k = (2I - L̃)^(K-k) L̃^k x`` from the input signal.

        Ignores ``u_prev`` / ``u_prev_prev`` (Bernstein has no recurrence)
        and rebuilds the basis vector directly, hence the ``O(K^2)``
        overall cost.

        Parameters
        ----------
        u_prev : Tensor
            Unused (no recurrence).
        u_prev_prev : Tensor or None
            Unused (no recurrence).
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``.
        signal : Tensor, shape ``[N, F]``
            The input features ``x`` being filtered.
        k : int
            Step index, ``1 <= k <= K``.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k = (2I - L̃)^(K-k) L̃^k x``.
        """
        v = signal
        for _ in range(k):
            v = L_apply(v)  # L̃^k x
        for _ in range(self.K - k):
            v = 2.0 * v - L_apply(v)  # (2I - L̃)^(K-k) ...
        return v
