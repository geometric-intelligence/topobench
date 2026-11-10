r"""FilterBankTNN: SCNN as a filter bank over the Hodge decomposition.

The topological analogue of ``FilterBankGNN``: separate polynomial filters
on the **down** and **up** Hodge Laplacians of each simplicial rank, fused
with learnable channel weights ``γ``:

.. math::

    y_d = \mathrm{post}_d\!\Big(
        \gamma_{d,\downarrow} \sum_t \theta^{\downarrow}_t (L^{\downarrow}_d)^t\, h
      + \gamma_{d,\uparrow}   \sum_s \theta^{\uparrow}_s   (L^{\uparrow}_d)^s\, h
        \Big),\qquad h = \mathrm{pre}_d(x_d).

Because ``L^down L^up = 0`` (the boundary-of-a-boundary identity), the two
channels tune the *gradient* and *curl* components of the Hodge
decomposition independently -- the topological analogue of the low-/
high-pass filter bank. Rank 0 (nodes) has no lower simplices, so only the
up channel (the graph Laplacian ``L_0``) is used there.

The channels reuse the graph
:class:`~topobench.nn.backbones.graph.filter_bank.channel.Channel` with a
Monomial basis, matching TopoModelX's SCNN power convolution; swap the
basis or set ``laplacian_norm='rescale'`` for higher-order / Chebyshev
variants. SCNN (Yang et al. 2022) is the Monomial instance.

Each channel's Monomial expansion includes the ``k = 0`` identity term,
so the fused output carries a learnable identity contribution from
*both* the down and up channels (versus SCNN's single ``H_0 I`` term).
With learnable ``θ`` / ``γ`` this spans the same function space -- a
harmless over-parameterization that also avoids materializing a dense
identity matrix.

References
----------
Yang, Isufi & Leus (2022) *Simplicial Convolutional Neural Networks*
(ICASSP 2022, arXiv:2110.02585) -- primary reference for SCNN.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from topobench.nn.backbones.graph.filter_bank.channel import Channel
from topobench.nn.backbones.graph.poly_filter.bases.monomial import Monomial
from topobench.nn.backbones.simplicial.hodge_utils import (
    build_hodge_laplacian_apply,
)


class FilterBankTNN(nn.Module):
    r"""SCNN: down/up polynomial-filter bank on the Hodge Laplacians.

    Parameters
    ----------
    in_channels : int
        Input feature width on every rank.
    hidden_channels : int
        Width of the filtered representation.
    out_channels : int
        Output feature width on every rank.
    K : int
        Polynomial degree of each channel.
    n_ranks : int, optional
        Number of simplicial ranks to filter. Defaults to ``3``.
    laplacian_norm : {'raw', 'rescale'}, optional
        ``'raw'`` applies the down/up operators directly (matching
        TopoModelX's SCNN); ``'rescale'`` maps them to ``[0, 2]`` for a
        Chebyshev-basis variant. Defaults to ``'raw'``.
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
        self.n_ranks = n_ranks
        self.laplacian_norm = laplacian_norm

        # Per rank: a down channel, an up channel, and their two weights.
        # (rank 0's down channel is unused -- nodes have no lower simplices.)
        self.down = nn.ModuleList(
            Channel(Monomial(), K=K, theta=None) for _ in range(n_ranks)
        )
        self.up = nn.ModuleList(
            Channel(Monomial(), K=K, theta=None) for _ in range(n_ranks)
        )
        self.gamma = nn.Parameter(torch.ones(n_ranks, 2))
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
        down_all: tuple[Tensor | None, ...],
        up_all: tuple[Tensor | None, ...],
    ) -> tuple[Tensor, ...]:
        r"""Fuse the down/up polynomial filters on each rank.

        Parameters
        ----------
        x_all : tuple of Tensor
            Per-rank signals ``(x_0, x_1, ...)``, each shape ``[n_d, F]``.
        down_all : tuple of Tensor or None
            Per-rank lower Hodge Laplacians ``L^down_d`` (``None`` for the
            ranks without lower simplices, e.g. rank 0).
        up_all : tuple of Tensor or None
            Per-rank upper Hodge Laplacians ``L^up_d`` (rank 0 uses the
            full ``L_0``; ``None`` for ranks without upper simplices).

        Returns
        -------
        tuple of Tensor
            The filtered per-rank signals ``(y_0, y_1, ...)``.
        """
        outs = []
        for d in range(self.n_ranks):
            h = self.pre[d](x_all[d])
            y = torch.zeros_like(h)
            if down_all[d] is not None:
                l_apply = build_hodge_laplacian_apply(
                    down_all[d], self.laplacian_norm
                )
                y = y + self.gamma[d, 0] * self.down[d](l_apply, h)
            if up_all[d] is not None:
                l_apply = build_hodge_laplacian_apply(
                    up_all[d], self.laplacian_norm
                )
                y = y + self.gamma[d, 1] * self.up[d](l_apply, h)
            outs.append(self.post[d](y))
        return tuple(outs)


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
