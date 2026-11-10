r"""Filter-bank channel and the shared polynomial-filter operations.

A :class:`Channel` is one polynomial filter in the bank,
``g_q(L̃) h = Σ_k θ_{q,k} T_q^(k)(L̃) h``, built from a
:class:`~topobench.nn.backbones.graph.poly_filter.basis.Basis` and a
coefficient vector ``θ`` that is either **learnable** (FiGURe) or a
**fixed buffer** (ACM-GNN, FAGCN, GNN-LF/HF).

The two free functions :func:`apply_polynomial_filter` and
:func:`build_laplacian_apply` mirror the accumulation loop and the
Laplacian closure of :class:`PolynomialFilterGNN`. They are duplicated
here, rather than imported, to keep the single-filter backbone
decoupled from the filter bank; the logic is intentionally identical (a
small, deliberate duplication).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian, scatter

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


def build_laplacian_apply(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
    laplacian_norm: str = "sym",
    self_loops: bool = True,
) -> LaplacianApply:
    r"""Build a closure ``h -> L̃ @ h`` for the normalized Laplacian.

    Mirrors ``PolynomialFilterGNN._build_laplacian_apply``, kept
    separate to leave the single-filter backbone untouched. For
    ``laplacian_norm='sym'`` the normalized adjacency
    ``Â = D̂^{-1/2}(A + γ I)D̂^{-1/2}`` is built with
    :func:`~torch_geometric.nn.conv.gcn_conv.gcn_norm` (``γ = 1`` when
    ``self_loops`` is set, the GCN renormalization / Liao's ``gen_norm``;
    ``γ = 0`` otherwise) and ``L̃ = I - Â`` is applied as ``h - Â h``. For
    ``'rw'`` / ``'none'`` the plain Laplacian from
    :func:`~torch_geometric.utils.get_laplacian` is used.

    Parameters
    ----------
    edge_index : Tensor, shape ``[2, E]``
        Edge index in COO format.
    edge_weight : Tensor or None, shape ``[E]``
        Edge weights; ``None`` means unit weights.
    num_nodes : int
        Number of nodes ``N``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Normalization scheme. Defaults to ``'sym'``.
    self_loops : bool, optional
        Add the GCN self-loop renormalization (``A + I`` before
        normalizing) for the ``'sym'`` scheme. Ignored for ``'rw'`` /
        ``'none'``. Defaults to ``True``.

    Returns
    -------
    LaplacianApply
        Closure ``h -> L̃ @ h`` mapping ``[N, F]`` to ``[N, F]``.
    """
    if laplacian_norm == "sym":
        ei, ew = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=num_nodes,
            add_self_loops=self_loops,
            improved=False,
        )
        src, dst = ei[0], ei[1]

        def apply(h: Tensor) -> Tensor:
            r"""Apply ``L̃ = I - Â`` to ``h`` via a scatter-add.

            Parameters
            ----------
            h : Tensor, shape ``[N, F]``
                Node features to multiply by ``L̃``.

            Returns
            -------
            Tensor, shape ``[N, F]``
                ``L̃ @ h = h - Â h``.
            """
            a_h = scatter(
                ew.view(-1, 1) * h.index_select(0, src),
                dst,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
            return h - a_h

        return apply

    norm = None if laplacian_norm == "none" else laplacian_norm
    ei, ew = get_laplacian(
        edge_index, edge_weight, normalization=norm, num_nodes=num_nodes
    )
    src, dst = ei[0], ei[1]

    def apply(h: Tensor) -> Tensor:
        r"""Apply ``L̃`` to ``h`` via a scatter-add over the edge list.

        Parameters
        ----------
        h : Tensor, shape ``[N, F]``
            Node features to multiply by ``L̃``.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``L̃ @ h``.
        """
        msg = ew.view(-1, 1) * h.index_select(0, src)
        return scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")

    return apply


def apply_polynomial_filter(
    basis: Basis,
    theta: Tensor,
    L_apply: LaplacianApply,
    h: Tensor,
    K: int,
) -> Tensor:
    r"""Accumulate one polynomial filter ``Σ_{k=0}^K θ_k T_k(L̃) h``.

    Mirrors the accumulation loop of ``PolynomialFilterGNN.forward``,
    with the coefficients ``θ`` passed in rather than owned by a
    backbone, so the same loop serves every channel of a filter bank.

    Parameters
    ----------
    basis : Basis
        The polynomial basis producing the ``T_k(L̃)`` sequence.
    theta : Tensor, shape ``[K + 1]``
        The channel coefficients. Passed through ``basis.effective_thetas``
        before accumulation (identity for every basis except ChebNetII).
    L_apply : LaplacianApply
        Closure ``h -> L̃ @ h``.
    h : Tensor, shape ``[N, F]``
        The signal being filtered.
    K : int
        Polynomial degree.

    Returns
    -------
    Tensor, shape ``[N, F]``
        ``Σ_{k=0}^K θ_eff[k] · T_k(L̃) h``.
    """
    theta_eff = basis.effective_thetas(theta)
    u_prev_prev: Tensor | None = None
    u_prev = basis.init(h, L_apply)
    y = theta_eff[0] * u_prev
    for k in range(1, K + 1):
        u_k = basis(u_prev, u_prev_prev, L_apply, signal=h, k=k)
        y = y + theta_eff[k] * u_k
        u_prev_prev, u_prev = u_prev, u_k
    return y


class Channel(nn.Module):
    r"""One filter-bank channel ``g_q(L̃) h = Σ_k θ_{q,k} T_q^(k)(L̃) h``.

    Wraps a single :class:`Basis` and a coefficient vector ``θ``. If
    ``theta`` is ``None`` the coefficients are a learnable
    ``nn.Parameter`` (FiGURe-style variable filters); if given, they are
    a fixed buffer (the linear LP/HP/ID filters of ACM-GNN, FAGCN, etc.,
    which are degree-≤1 Monomial filters with prescribed coefficients).

    Parameters
    ----------
    basis : Basis
        The basis for this channel (any module from the basis registry).
    K : int
        Polynomial degree of this channel.
    theta : list of float or None, optional
        Fixed coefficients of length ``K + 1``. ``None`` (default) makes
        them learnable.
    """

    def __init__(self, basis: Basis, K: int, theta: list[float] | None = None):
        super().__init__()
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
        self.basis = basis
        self.K = K
        if theta is None:
            self.theta = nn.Parameter(torch.empty(K + 1))
            nn.init.normal_(self.theta, mean=1.0 / (K + 1), std=0.01)
        else:
            t = torch.as_tensor(theta, dtype=torch.float32)
            if t.numel() != K + 1:
                raise ValueError(
                    f"theta must have K+1={K + 1} entries, got {t.numel()}"
                )
            self.register_buffer("theta", t)

    def forward(self, L_apply: LaplacianApply, h: Tensor) -> Tensor:
        r"""Apply this channel's polynomial filter to ``h``.

        Parameters
        ----------
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h`` (built once by the backbone).
        h : Tensor, shape ``[N, F]``
            The signal being filtered.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``g_q(L̃) h``.
        """
        return apply_polynomial_filter(
            self.basis, self.theta, L_apply, h, self.K
        )


class PPRChannel(nn.Module):
    r"""Personalized-PageRank channel with a linear prefactor (GNN-LF/HF).

    Implements one channel of the GNN-LF/HF filter bank (Liao et al.
    2024, "PPR (LP, HP)"):

    .. math::

        g_q(\tilde L)\,h = (I + \beta\,\tilde L)
            \sum_{k=0}^{K} \alpha (1-\alpha)^k\, (I - \tilde L)^k\, h .

    This does not fit the :class:`Channel` shape for two reasons: the
    expansion is in powers of ``(I - L̃) = Ã`` rather than ``L̃``, and the
    sum is pre-multiplied by the linear operator ``(I + β L̃)``. The
    PPR weights ``α (1-α)^k`` and the scaling ``β`` are fixed
    hyperparameters (no learnable parameters here; the channel weights
    ``γ`` are learned at the bank level), matching the decoupled PPR
    propagation of APPNP-style models.

    Parameters
    ----------
    K : int
        Number of propagation steps.
    alpha : float
        PPR teleport probability ``α`` in ``(0, 1)``; the coefficients
        are ``α (1-α)^k``.
    beta : float
        Scaling of the linear prefactor ``(I + β L̃)``. ``β = 0`` recovers
        the plain PPR (low-pass) filter; ``β > 0`` adds high-pass content.

    References
    ----------
    Zhu et al. (2021) *Interpreting and Unifying Graph Neural Networks
    with An Optimization Framework* (WWW) -- primary reference for
    GNN-LF/HF. The spectral form is the "PPR (LP, HP)" filter-bank entry
    of Liao et al. (2024), Table 1.
    """

    def __init__(self, K: int, alpha: float, beta: float):
        super().__init__()
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.K = K
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, L_apply: LaplacianApply, h: Tensor) -> Tensor:
        r"""Apply the PPR-with-prefactor filter to ``h``.

        Parameters
        ----------
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h`` (built once by the backbone).
        h : Tensor, shape ``[N, F]``
            The signal being filtered.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``(I + β L̃) Σ_k α (1-α)^k (I - L̃)^k h``.
        """
        a = self.alpha
        # k = 0 term: α (I - L̃)^0 h = α h; then power-iterate on Ã = I - L̃.
        s = a * h
        u = h
        for k in range(1, self.K + 1):
            u = u - L_apply(u)  # u = (I - L̃) u_{k-1}
            s = s + a * (1.0 - a) ** k * u
        # Linear prefactor (I + β L̃) s = s + β (L̃ s).
        return s + self.beta * L_apply(s)


class GaussianChannel(nn.Module):
    r"""Gaussian (heat-kernel-like) band-pass channel of G²CN.

    Implements one channel of the G²CN filter bank (Liao et al. 2024,
    "Gaussian"):

    .. math::

        g_q(\tilde L)\,h = \sum_{k=0}^{K} \frac{\alpha^k}{k!}
            \big(\mathrm{shift}\cdot I - \tilde L\big)^{2k}\, h ,

    a truncated exponential ``exp(α M)`` of the **squared** shifted
    operator ``M = (shift\cdot I - \tilde L)^2`` (the 2-hop propagation of
    G²CN). The squared operator concentrates the response around a
    frequency set by ``shift``: the two G²CN channels use
    ``shift = 1 + β₁ > 1`` (low-frequency) and ``shift = 1 - β₂ < 1``
    (high-frequency). The decay ``α`` and ``shift`` are fixed
    hyperparameters (no learnable parameters here; the channel weights
    ``γ`` are learned at the bank level).

    Parameters
    ----------
    K : int
        Number of Gaussian terms ``K + 1`` (truncation degree of the
        exponential).
    alpha : float
        Exponential decay ``α`` in the term weights ``α^k / k!``.
    shift : float
        Diagonal shift of the operator ``(shift·I - L̃)``; ``> 1`` for the
        low-frequency channel, ``< 1`` for the high-frequency channel.

    References
    ----------
    Li et al. (2022) *G²CN: Graph Gaussian Convolution Networks with
    Concentrated Graph Filters* (ICML) -- primary reference for G²CN. The
    spectral form is the "Gaussian" filter-bank entry of Liao et al.
    (2024), Table 1.
    """

    def __init__(self, K: int, alpha: float, shift: float):
        super().__init__()
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
        self.K = K
        self.alpha = float(alpha)
        self.shift = float(shift)
        coeffs = [self.alpha**k / math.factorial(k) for k in range(K + 1)]
        self.register_buffer(
            "coeffs", torch.tensor(coeffs, dtype=torch.float32)
        )

    def forward(self, L_apply: LaplacianApply, h: Tensor) -> Tensor:
        r"""Apply the Gaussian band-pass filter to ``h``.

        Parameters
        ----------
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h`` (built once by the backbone).
        h : Tensor, shape ``[N, F]``
            The signal being filtered.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``Σ_k (α^k / k!) (shift·I - L̃)^{2k} h``.
        """

        def shifted(v: Tensor) -> Tensor:
            r"""Apply ``(shift·I - L̃)`` to ``v``.

            Parameters
            ----------
            v : Tensor, shape ``[N, F]``
                Tensor to multiply by ``shift·I - L̃``.

            Returns
            -------
            Tensor, shape ``[N, F]``
                ``shift·v - L̃ v``.
            """
            return self.shift * v - L_apply(v)

        # u_k = M^k h = (shift·I - L̃)^{2k} h; one M is two `shifted` calls.
        u = h
        out = self.coeffs[0] * u
        for k in range(1, self.K + 1):
            u = shifted(shifted(u))
            out = out + self.coeffs[k] * u
        return out
