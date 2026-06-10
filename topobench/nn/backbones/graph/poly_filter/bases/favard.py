r"""FavardGNN polynomial basis.

Three-term recurrence with **learnable** coefficients ``α_k, β_k`` (and
the backbone's ``θ_k`` on top: orthogonal degrees of freedom). The
construction is justified by Favard's theorem: every three-term
recurrence of the form below produces a family of polynomials that
are orthonormal with respect to *some* inner product, so learning the
coefficients amounts to learning the inner product (equivalently,
the basis itself).

Liao Appendix B formulation:

.. math::

    T^{(-1)}(\tilde L) = O, \quad
    T^{(0)}(\tilde L) = \frac{1}{\sqrt{\alpha_0}}\, I,

    T^{(k)}(\tilde L) = \frac{1}{\sqrt{\alpha_k}}\,(I - \tilde L)\, T^{(k-1)}(\tilde L)
                          - \beta_k\, T^{(k-1)}(\tilde L)
                          - \sqrt{\alpha_{k-1}}\, T^{(k-2)}(\tilde L) ,
    \quad k \ge 1 .

**Reparameterization (equivalent; chosen for cleanliness).**
Liao's formula only ever uses ``√α_k`` (as divisor and as multiplier),
never ``α_k`` itself. We therefore learn ``a_k := √α_k`` directly. To
guarantee ``a_k > 0`` (otherwise the recurrence is ill-defined and may
blow up), we parameterize

.. math::

    a_k = \mathrm{softplus}(a^{\text{raw}}_k) \,=\, \log(1 + e^{a^{\text{raw}}_k}) ,

with ``a^{raw}_k`` as the unconstrained ``nn.Parameter``. ``β_k`` is
unconstrained. So the recurrence becomes

.. math::

    u_0 = x / a_0 ,

    u_k = \frac{1}{a_k}\,(u_{k-1} - \tilde L\, u_{k-1})
          - \beta_k\, u_{k-1}
          - a_{k-1}\, u_{k-2}, \quad k \ge 1 ,

with ``u_{-1} = 0`` encoded as ``u_prev_prev = None`` at ``k = 1``.

This is the first basis in the registry where the basis **owns
learnable parameters of its own**: the requirement that ``Basis`` is
an ``nn.Module`` rather than a pure function exists precisely so this
case (and the analogous learnable interpolation in ChebNetII) is
expressible without bifurcating the protocol.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Favard" entry.

Guo & Wei (2023) *Graph Neural Networks with Learnable and Optimal
Polynomial Bases* (ICML, arXiv:2302.12432): primary reference for
FavardGNN. They prove (Theorem 3.2) that *any* polynomial filter can be
expressed in this form, so the learnable α, β recover the entire
spectral filter family, at the cost of harder optimization vs. fixed
bases like Jacobi.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


class FavardGNN(Basis):
    r"""Favard three-term recurrence with learnable ``α``, ``β``.

    Parameters
    ----------
    K : int
        Polynomial degree. Must match the backbone's ``K``: the
        ``α`` and ``β`` parameter vectors are sized ``K + 1`` here.
        In Hydra configs, set
        ``model.backbone.basis.K = ${model.backbone.K}``.
    """

    def __init__(self, K: int):
        super().__init__()
        if K < 0:
            raise ValueError(f"FavardGNN K must be >= 0, got {K}")
        self.K = K

        # a_k = √α_k > 0 via softplus(a_raw_k). Small random init around 0
        # gives a_k ~ softplus(0) = ln(2) ~ 0.693, which produces a roughly
        # Legendre-like initial basis (where the analogous coefficient is 1).
        # Close enough to be a stable starting point; the basis then
        # learns away from there.
        self.a_raw = nn.Parameter(torch.empty(K + 1))
        nn.init.normal_(self.a_raw, mean=0.0, std=0.01)

        # β_k unconstrained; init near 0 so the basis starts close to a
        # symmetric (Legendre-like) configuration.
        self.beta = nn.Parameter(torch.empty(K + 1))
        nn.init.normal_(self.beta, mean=0.0, std=0.01)

    def _a(self) -> Tensor:
        r"""Compute the positive coefficients ``a_k = √α_k`` from ``a_raw``.

        Returns
        -------
        Tensor, shape ``[K + 1]``
            ``softplus(a_raw)``, strictly positive by construction.
        """
        return F.softplus(self.a_raw)

    def init(self, x: Tensor, L_apply: LaplacianApply) -> Tensor:
        r"""Return ``u_0 = x / a_0`` where ``a_0 = √α_0``.

        Parameters
        ----------
        x : Tensor, shape ``[N, F]``
            Input features.
        L_apply : LaplacianApply
            Unused for FavardGNN's ``init``; only the recurrence
            consumes it.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_0``.
        """
        return x / self._a()[0]

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,
        L_apply: LaplacianApply,
        signal: Tensor,  # unused: signal-independent (cf. OptBasis)
        k: int,
    ) -> Tensor:
        r"""Apply the Favard recurrence with learnable ``a_k = √α_k`` and ``β_k``.

        Parameters
        ----------
        u_prev : Tensor, shape ``[N, F]``
            ``u_{k-1}``, the previous basis vector.
        u_prev_prev : Tensor or None, shape ``[N, F]``
            ``u_{k-2}``. ``None`` at ``k = 1`` (encodes ``u_{-1} = 0``).
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``.
        signal : Tensor
            Unused for FavardGNN (signal-independent basis).
        k : int
            Step index, used to look up ``a[k], β[k], a[k - 1]``.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k``.
        """
        a = self._a()
        # (1/a_k) (I - L̃) u_{k-1}  -  β_k u_{k-1}  [  -  a_{k-1} u_{k-2} ]
        z_u = u_prev - L_apply(u_prev)
        out = z_u / a[k] - self.beta[k] * u_prev
        if u_prev_prev is not None:
            out = out - a[k - 1] * u_prev_prev
        return out
