r"""PolynomialFilterTNN: a single polynomial filter in the Hodge Laplacian.

The topological analogue of ``PolynomialFilterGNN``: the same
``Σ_k θ_k T_k(·) x`` accumulation and the same swappable
:class:`~topobench.nn.backbones.graph.poly_filter.basis.Basis` registry,
but the operator is the ``d``-th Hodge Laplacian ``L_d`` instead of the
graph Laplacian. One polynomial filter is applied to the signal on each
simplicial rank (nodes, edges, triangles).

SNN (Ebli, Defferrard & Spreemann 2020) is this backbone with the
Chebyshev basis: a Chebyshev polynomial in ``L_d`` on each rank.

The basis modules are reused unchanged from the polynomial-filter
registry: a ``Basis`` only ever sees the closure ``L_apply : h -> L h``
and never the underlying domain, so the Monomial / Chebyshev / Jacobi
bases act on the Hodge Laplacian verbatim. The bases define their
argument *relative to* ``L_apply`` (Chebyshev evaluates ``T_k`` at
``L_apply - I``), so a ``[0, 2]``-spectrum operator is all they need;
``laplacian_norm='rescale'`` provides exactly that via ``2 L / λmax``.

References
----------
Ebli, Defferrard & Spreemann (2020) *Simplicial Neural Networks*
(NeurIPS 2020 TDA workshop, arXiv:2010.03633) -- primary reference for SNN.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from topobench.nn.backbones.graph.filter_bank.channel import (
    apply_polynomial_filter,
)
from topobench.nn.backbones.graph.poly_filter.basis import Basis
from topobench.nn.backbones.simplicial.hodge_utils import (
    build_hodge_laplacian_apply,
)


class PolynomialFilterTNN(nn.Module):
    r"""Polynomial filter in the Hodge Laplacian, one per simplicial rank.

    For each rank ``d`` the backbone computes

    .. math::

        y_d = \mathrm{post}_d\!\Big( \sum_{k=0}^{K} \theta_{d,k}\,
              T_k(L_d)\, \mathrm{pre}_d(x_d) \Big),

    where ``L_d`` is the ``d``-th Hodge Laplacian and ``{T_k}`` is a shared
    basis. SNN (Ebli et al. 2020) is the Chebyshev instance.

    Parameters
    ----------
    in_channels : int
        Input feature width on every rank.
    hidden_channels : int
        Width of the filtered representation.
    out_channels : int
        Output feature width on every rank.
    K : int
        Polynomial degree.
    basis : Basis
        A basis module from the polynomial-filter registry, shared across
        ranks. Use a stateless basis (Chebyshev, Monomial, Jacobi,
        Legendre); parameter-owning bases (FavardGNN, ChebNetII) would
        share their parameters across ranks.
    n_ranks : int, optional
        Number of simplicial ranks to filter (``0 .. n_ranks - 1``).
        Defaults to ``3`` (nodes, edges, triangles).
    laplacian_norm : {'raw', 'rescale'}, optional
        ``'raw'`` applies ``L_d`` directly; only the Monomial basis is
        safe on the unbounded Hodge spectrum (a monomial in the full
        ``L_d`` is the SCN / SCCNN power-convolution pattern).
        ``'rescale'`` hands the basis ``2 L_d / bound`` (``bound`` a
        deterministic Gershgorin upper bound on ``λmax``; spectrum in
        ``[0, 2]``) so an orthogonal-interval basis (Chebyshev, Jacobi,
        Legendre) stays inside its ``[-1, 1]`` domain after its internal
        shift, and must be used with those. Defaults to ``'raw'``.
    pre_mlp_layers : int, optional
        Depth of the per-rank pre-filter MLP. Defaults to ``1``.
    post_mlp_layers : int, optional
        Depth of the per-rank post-filter MLP. Defaults to ``1``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int,
        basis: Basis,
        n_ranks: int = 3,
        laplacian_norm: str = "raw",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        super().__init__()
        if laplacian_norm not in {"raw", "rescale"}:
            raise ValueError(
                "laplacian_norm must be 'raw' or 'rescale'; "
                f"got {laplacian_norm!r}"
            )
        self.basis = basis
        self.K = K
        self.n_ranks = n_ranks
        self.laplacian_norm = laplacian_norm

        # One coefficient vector theta and one pre/post MLP per rank.
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.empty(K + 1)) for _ in range(n_ranks)]
        )
        for t in self.theta:
            nn.init.normal_(t, mean=1.0 / (K + 1), std=0.01)
        self.pre = nn.ModuleList(
            _build_mlp(in_channels, hidden_channels, pre_mlp_layers)
            for _ in range(n_ranks)
        )
        self.post = nn.ModuleList(
            _build_mlp(hidden_channels, out_channels, post_mlp_layers)
            for _ in range(n_ranks)
        )

    def forward(
        self,
        x_all: tuple[Tensor, ...],
        laplacian_all: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        r"""Filter each rank's signal with a polynomial in its ``L_d``.

        Parameters
        ----------
        x_all : tuple of Tensor
            Per-rank signals ``(x_0, x_1, ...)``, each shape ``[n_d, F]``.
        laplacian_all : tuple of Tensor
            Per-rank Hodge Laplacians ``(L_0, L_1, ...)``, each a (sparse)
            ``[n_d, n_d]`` operator aligned with ``x_all``.

        Returns
        -------
        tuple of Tensor
            The filtered per-rank signals ``(y_0, y_1, ...)``.
        """
        outs = []
        for d in range(self.n_ranks):
            h = self.pre[d](x_all[d])
            L_apply = self._build_laplacian_apply(laplacian_all[d])
            y = apply_polynomial_filter(
                self.basis, self.theta[d], L_apply, h, self.K
            )
            outs.append(self.post[d](y))
        return tuple(outs)

    def _build_laplacian_apply(self, laplacian: Tensor):
        r"""Build a closure ``h -> L h`` for one rank's Hodge Laplacian.

        Parameters
        ----------
        laplacian : Tensor
            The (sparse or dense) ``[n_d, n_d]`` Hodge Laplacian ``L_d``.

        Returns
        -------
        Callable[[Tensor], Tensor]
            ``h -> L_d h`` (``'raw'``) or ``h -> (2 / λmax) L_d h``
            (``'rescale'``, spectrum mapped to ``[0, 2]``).
        """
        return build_hodge_laplacian_apply(laplacian, self.laplacian_norm)


def _build_mlp(
    in_dim: int, out_dim: int, n_layers: int, dropout: float = 0.0
) -> nn.Module:
    r"""Build a small MLP; ``n_layers == 1`` is a single ``nn.Linear``.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension and hidden width.
    n_layers : int
        Number of ``Linear`` layers.
    dropout : float, optional
        Dropout probability between hidden layers. Defaults to ``0.0``.

    Returns
    -------
    nn.Module
        A single ``nn.Linear`` or an ``nn.Sequential``.
    """
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(cur, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        cur = out_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)
