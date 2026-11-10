r"""Polynomial basis protocol for :class:`PolynomialFilterGNN`.

A :class:`Basis` produces a sequence of basis vectors
``u_0, u_1, ..., u_K`` where ``u_k`` corresponds to ``T_k(L̃) x`` for some
polynomial sequence ``{T_k}``. The backbone owns the coefficients
``θ_k`` and the accumulation ``y = Σ_k θ_k · u_k``; the basis owns
only the recurrence (and any parameters the recurrence needs, e.g.
FavardGNN's ``α``, ``β``; ChebNetII's interpolation nodes).

The single ``forward`` signature is **uniform** across signal-dependent
bases (OptBasis) and signal-independent bases (Chebyshev, Monomial,
Legendre, Jacobi, ChebNetII, Favard); stateless bases simply ignore
``signal``.

References
----------
The protocol is shaped to fit every variable-basis entry in
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

LaplacianApply = Callable[[Tensor], Tensor]
r"""Closure ``h -> L̃ @ h``.

Built **once per forward pass** by the backbone from
``(edge_index, edge_weight)`` and the chosen Laplacian normalization,
then frozen and handed to the basis. Bases never see ``edge_index`` /
``edge_weight`` directly: that decouples them from how the operator is
stored and pins down the normalization convention at the backbone
level (every registered basis sees the *same* operator).
"""


class Basis(nn.Module):
    r"""Abstract polynomial basis for :class:`PolynomialFilterGNN`.

    Subclasses implement :meth:`forward` returning ``u_k`` from
    ``(u_prev, u_prev_prev, L_apply, signal, k)``. They may override
    :meth:`init` to produce a non-identity ``u_0`` (e.g. OptBasis with
    ``u_0 = x / ‖x‖``).

    Bases are ``nn.Module`` subclasses so they can own learnable
    parameters (FavardGNN's ``α``, ``β``; ChebNetII's interpolation
    coefficients). Stateless bases (Monomial, Chebyshev, Legendre, ...)
    simply register none.

    The backbone treats every basis as opaque and never branches on the
    concrete class. Adding a new basis is a single new file plus a Hydra
    ``_target_`` swap: no backbone change.
    """

    def init(self, x: Tensor, L_apply: LaplacianApply) -> Tensor:
        r"""Return ``u_0``, the zeroth basis vector.

        Default behaviour is ``u_0 = x`` (i.e. ``T_0 = I``), which is
        what every basis in the registry uses **except** OptBasis,
        whose orthonormal recurrence is seeded with ``u_0 = x / ‖x‖``.
        Override in subclasses that need a non-identity initial vector.

        Parameters
        ----------
        x : Tensor, shape ``[N, F]``
            The input features being filtered. In :class:`PolynomialFilterGNN`
            this is the post-pre-MLP signal ``h``, not the raw network input.
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``. Provided in case ``init`` itself needs it;
            the default implementation ignores it.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_0``.
        """
        return x

    def effective_thetas(self, backbone_theta: Tensor) -> Tensor:
        r"""Map the backbone's ``θ`` vector to the effective accumulator coefficients.

        Default: return ``backbone_theta`` unchanged. This is the standard
        case where the backbone owns the learnable ``θ_k``. Override in
        bases that *reparameterize* the coefficients, e.g. ChebNetII,
        where the user-facing learnable parameters are interpolation
        values at Chebyshev nodes and the effective ``θ_k`` is the
        Chebyshev interpolation reconstruction
        ``θ_k = (2/(K+1)) Σ_κ θ_κ T^(k)(x_κ)`` (Liao Appendix B,
        "Chebyshev Interpolation" entry).

        A basis that overrides this method **may ignore**
        ``backbone_theta`` entirely and emit coefficients from its own
        ``nn.Parameter`` instances. The backbone's ``self.theta`` is
        still constructed (it determines the shape signal ``K + 1``),
        but for such bases it stays at its initialization and
        contributes nothing to the loss surface: accepted as a small
        redundancy in exchange for keeping the protocol uniform across
        bases that do and do not own their coefficients.

        Parameters
        ----------
        backbone_theta : Tensor, shape ``[K + 1]``
            The backbone's learnable ``θ`` vector.

        Returns
        -------
        Tensor, shape ``[K + 1]``
            The effective coefficients to use in
            ``y = Σ_k θ_eff[k] · u_k``.
        """
        return backbone_theta

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,
        L_apply: LaplacianApply,
        signal: Tensor,
        k: int,
    ) -> Tensor:
        r"""Produce ``u_k`` from the basis recurrence.

        Parameters
        ----------
        u_prev : Tensor, shape ``[N, F]``
            ``u_{k-1}``, the previous basis vector.
        u_prev_prev : Tensor or None, shape ``[N, F]``
            ``u_{k-2}``. ``None`` at the boundary ``k == 1`` so subclasses
            can encode any single-step-special-case (e.g. Chebyshev's
            ``T_1 = (L̃ - I) T_0``) inside the basis rather than the backbone.
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``. The same closure is reused for every
            ``k`` in a given forward pass.
        signal : Tensor, shape ``[N, F]``
            The input features being filtered (same as the ``x`` argument
            to :meth:`init`). Always passed; stateless bases ignore it.
            OptBasis uses it to define the inner product against which its
            recurrence coefficients are computed.
        k : int
            Step index, ``k >= 1``. Passed explicitly so bases with
            per-``k`` parameterizations (ChebNetII's interpolation; any
            closed-form-per-``k`` family) need no internal state.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k``.
        """
        raise NotImplementedError
