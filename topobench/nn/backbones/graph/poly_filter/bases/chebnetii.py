"""ChebNetII basis: Chebyshev recurrence with interpolation reparameterization.

The basis recurrence is identical to first-kind Chebyshev — what differs
is the **coefficient parameterization**. The K+1 learnable parameters
are interpolation values ``θ_κ`` at the Chebyshev nodes of ``T^{(K+1)}``,
and the effective accumulator coefficient ``θ_k`` is the discrete
Chebyshev reconstruction:

.. math::

    g(\\tilde L; \\theta) \\;=\\; \\frac{2}{K+1}
       \\sum_{k=0}^{K} \\sum_{\\kappa=0}^{K}
       \\theta_\\kappa \\, T^{(k)}(x_\\kappa) \\, T^{(k)}(\\tilde L) ,
    \\qquad
    x_\\kappa = \\cos\\!\\left(\\frac{\\kappa + \\tfrac{1}{2}}{K+1}\\,\\pi\\right) .

Rewriting:

.. math::

    g(\\tilde L; \\theta) = \\sum_{k=0}^{K} \\theta_k^{\\text{eff}} \\, T^{(k)}(\\tilde L) ,
    \\qquad
    \\theta_k^{\\text{eff}}
       = \\tfrac{2}{K+1} \\sum_{\\kappa=0}^{K} \\theta_\\kappa \\, T^{(k)}(x_\\kappa)
       = (M \\cdot \\theta_{\\text{interp}})[k] ,

where ``M[k, κ] = (2/(K+1)) T^{(k)}(x_κ)`` is fixed (depends only on
``K``) and ``θ_interp = (θ_0, …, θ_K)`` is what we actually learn.

This is the only basis in the registry that uses the
:meth:`~topobench.nn.backbones.graph.poly_filter.basis.Basis.effective_thetas`
protocol hook — the basis takes ownership of the accumulator
coefficients, so the backbone's own ``θ`` parameter is still
constructed (it determines the ``K + 1`` shape) but receives no
gradient. Accepted as a small redundancy in exchange for keeping the
basis-protocol signature uniform across bases that do and do not
reparameterize their coefficients.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Chebyshev Interpolation
(ChebInterp)" entry.

He, Wei & Wen (2022) *Convolutional Neural Networks on Graphs with
Chebyshev Approximation, Revisited* (NeurIPS) — primary reference for
ChebNetII. They motivate the interpolation reparameterization as
inducing "generally decaying" coefficients that approximate analytic
spectral filters better than free ``θ_k``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from topobench.nn.backbones.graph.poly_filter.bases.chebyshev import Chebyshev


class ChebNetII(Chebyshev):
    """Chebyshev basis with He-Wei-Wen 2022 interpolation reparameterization.

    Parameters
    ----------
    K : int
        Polynomial degree. **Must match the backbone's** ``K`` — the
        ``θ_interp`` vector and the precomputed interpolation matrix
        ``M`` are sized ``K + 1`` here, and a mismatch with the
        backbone's ``θ`` is detected in :meth:`effective_thetas` and
        raised as a configuration error. In Hydra configs, set
        ``model.backbone.basis.K = ${model.backbone.K}`` to keep them
        in sync.

    Notes
    -----
    The recurrence is inherited verbatim from :class:`Chebyshev` —
    ``ChebNetII`` *is* a Chebyshev basis at the operator level. The
    learnable thing is only the coefficient parameterization.
    """

    def __init__(self, K: int):
        super().__init__()
        if K < 0:
            raise ValueError(f"ChebNetII K must be >= 0, got {K}")
        self.K = K

        # Chebyshev nodes x_κ = cos((κ + 0.5) / (K+1) · π) for κ = 0..K.
        # These are the roots of T^(K+1), the canonical interpolation
        # nodes for first-kind Chebyshev polynomials.
        kappa = torch.arange(K + 1, dtype=torch.float32)
        x_nodes = torch.cos((kappa + 0.5) / (K + 1) * math.pi)

        # Interpolation matrix M[k, κ] = (2/(K+1)) · T^(k)(x_κ).
        # T^(k)(x_κ) computed via the classical first-kind Chebyshev
        # recurrence in scalar form — same equations as Chebyshev.forward,
        # just evaluated at the K+1 nodes instead of the operator.
        T = torch.empty(K + 1, K + 1)
        T[0, :] = 1.0
        if K >= 1:
            T[1, :] = x_nodes
            for k in range(2, K + 1):
                T[k, :] = 2.0 * x_nodes * T[k - 1, :] - T[k - 2, :]
        M = (2.0 / (K + 1)) * T
        # M is a fixed function of K — never trained, never moved
        # between dtypes by anything but the module itself. Register
        # as a buffer so .to(device) / state_dict round-trips work.
        self.register_buffer("M", M)

        # The K+1 learnable interpolation values.
        self.theta_interp = nn.Parameter(torch.empty(K + 1))
        # Init like the backbone's θ — small noise around 1/(K+1) so the
        # initial filter is roughly uniform across orders.
        nn.init.normal_(self.theta_interp, mean=1.0 / (K + 1), std=0.01)

    def effective_thetas(self, backbone_theta: Tensor) -> Tensor:
        """Return ``M @ θ_interp``; ``backbone_theta`` is intentionally ignored.

        The basis takes ownership of the accumulator coefficients here.
        The backbone's ``θ`` still exists — it sets the size signal —
        but it has no gradient attached for ChebNetII (see module
        docstring for the rationale).

        Parameters
        ----------
        backbone_theta : Tensor, shape ``[K + 1]``
            The backbone's ``θ`` vector. Used only to detect a
            ``K`` mismatch between the basis and the backbone; the
            returned coefficients come exclusively from ``θ_interp``.

        Returns
        -------
        Tensor, shape ``[K + 1]``
            ``M @ θ_interp`` — the Chebyshev-interpolation
            reconstruction of the effective coefficients ``θ_k^{eff}``.

        Raises
        ------
        ValueError
            If ``backbone_theta`` has size other than ``K + 1``. This
            catches the configuration mistake where the basis ``K`` and
            backbone ``K`` are out of sync.
        """
        if backbone_theta.size(0) != self.K + 1:
            raise ValueError(
                f"ChebNetII was constructed with K={self.K} (interpolation "
                f"matrix is {self.K + 1}x{self.K + 1}), but the backbone "
                f"passed θ of size {backbone_theta.size(0)}. Set "
                f"model.backbone.basis.K = ${{model.backbone.K}} in the "
                f"Hydra config so they stay in sync."
            )
        return self.M @ self.theta_interp
