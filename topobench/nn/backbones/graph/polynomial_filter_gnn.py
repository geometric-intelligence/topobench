r"""PolynomialFilterGNN: a single polynomial filter on the normalized Laplacian.

Implements the propagation

.. math::

    y \;=\; \mathrm{post}\!\left(\sum_{k=0}^{K} \theta_k \, u_k\right),
    \qquad u_k = T_k(\tilde L)\,\mathrm{pre}(x)

where ``{T_k}`` is a polynomial sequence produced by a swappable
:class:`~topobench.nn.backbones.graph.poly_filter.basis.Basis`.

This is the single-polynomial-filter pattern. Filter banks (multiple
parallel polynomial filters fused with learnable mixing) are a
structurally different forward pass and live in a separate backbone
(``FilterBankGNN``, planned for a follow-up PR).

Notes
-----
The backbone owns: the propagation loop, the coefficients ``θ_k``, the
accumulation, the pre- and post-polynomial MLPs, the Laplacian
normalization convention, and the ``(x, edge_index, batch, edge_weight)``
interface expected by
:class:`~topobench.nn.wrappers.graph.GNNWrapper`.

The basis owns: the recurrence
``T_k = f(T_{k-1}, T_{k-2}, L̃, signal, k)`` and any parameters the
recurrence needs.

The backbone treats the basis as opaque and never branches on its
concrete class. Adding a new basis is a single new file plus a Hydra
``_target_`` swap.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675): survey unifying every variable-basis
spectral GNN under the same recurrence-in-``L̃`` template. The
single-polynomial-filter pattern implemented here corresponds to
``g(L̃; θ) = Σ_k θ_k T^{(k)}(L̃)`` (Liao Appendix B, Variable Basis
block); the choice of ``T^{(k)}`` is delegated to the basis registry.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import get_laplacian, scatter

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


class PolynomialFilterGNN(nn.Module):
    r"""Single polynomial filter on ``L̃`` with a swappable basis.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Width of the polynomial filter (and the pre-MLP output).
    out_channels : int
        Number of output node features (post-MLP output).
    K : int
        Polynomial degree. ``K = 0`` reduces the filter to ``θ_0 · I``.
    basis : Basis
        The basis module producing ``u_k = T_k(L̃) · pre(x)``. Instantiated
        by Hydra via the ``model.backbone.basis._target_`` config entry.
    dropout : float, optional
        Dropout applied after the pre-MLP and inside multi-layer MLPs.
        Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Normalization used to build ``L̃`` via
        :func:`torch_geometric.utils.get_laplacian`. ``'sym'`` is the
        symmetric normalization ``I - D^{-1/2} A D^{-1/2}`` used by Liao
        Appendix B; defaults to ``'sym'``.
    pre_mlp_layers : int, optional
        Number of layers in the pre-polynomial MLP. ``1`` is a single
        ``nn.Linear``; defaults to ``1``.
    post_mlp_layers : int, optional
        Number of layers in the post-polynomial MLP. Defaults to ``1``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int,
        basis: Basis,
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        super().__init__()
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
        if laplacian_norm not in {"sym", "rw", "none"}:
            raise ValueError(
                "laplacian_norm must be one of 'sym', 'rw', 'none'; "
                f"got {laplacian_norm!r}"
            )

        self.K = K
        self.basis = basis
        self.laplacian_norm = laplacian_norm

        # Polynomial coefficients θ_k: one scalar per order, shared across
        # channels. This matches Liao Appendix B's "shared θ" parameterization
        # for the variable-basis block. If a future registered basis needs
        # per-channel θ (some ChebNetII variants do), promote this to
        # ``Parameter(K + 1, hidden_channels)`` here: backbone-internal, no
        # change to the Basis protocol.
        self.theta = nn.Parameter(torch.empty(K + 1))
        nn.init.normal_(self.theta, mean=1.0 / (K + 1), std=0.01)

        self.pre = _build_mlp(
            in_channels, hidden_channels, pre_mlp_layers, dropout
        )
        self.post = _build_mlp(
            hidden_channels, out_channels, post_mlp_layers, dropout
        )
        self.dropout = nn.Dropout(dropout)

        # Public attributes matching the convention of existing backbones
        # (`identity_gnn.py`, `gps.py`, `nsd.py`).
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        edge_weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``[N, in_channels]``
            Node features.
        edge_index : Tensor, shape ``[2, E]``
            Edge index in COO format.
        batch : Tensor, optional
            Batch assignment vector. Not used at the propagation level:
            this backbone produces node-level outputs and the wrapper /
            readout handle pooling.
        edge_weight : Tensor, optional
            Edge weights for the underlying adjacency. Used to build the
            normalized Laplacian ``L̃``.
        **kwargs : dict
            Swallowed for compatibility with the wrapper's call site.

        Returns
        -------
        Tensor, shape ``[N, out_channels]``
            Output node features.
        """
        h = self.pre(x)
        h = self.dropout(h)

        # Build L̃ once per forward pass and freeze it into a closure.
        # The basis never sees edge_index / edge_weight directly: the
        # backbone fixes the Laplacian normalization once so every
        # registered basis sees the same operator. See LaplacianApply.
        L_apply = self._build_laplacian_apply(
            edge_index, edge_weight, num_nodes=h.size(0)
        )

        # Polynomial accumulator: y = Σ_{k=0}^K θ_eff[k] · u_k.
        # `effective_thetas` defaults to identity (most bases); ChebNetII
        # overrides it to substitute its own interpolation-derived
        # coefficients (Liao App. B, "Chebyshev Interpolation").
        theta_eff = self.basis.effective_thetas(self.theta)
        u_prev_prev: Tensor | None = None
        u_prev = self.basis.init(h, L_apply)
        y = theta_eff[0] * u_prev
        for k in range(1, self.K + 1):
            u_k = self.basis(u_prev, u_prev_prev, L_apply, signal=h, k=k)
            y = y + theta_eff[k] * u_k
            u_prev_prev, u_prev = u_prev, u_k

        return self.post(y)

    # ------------------------------------------------------------------ #
    # Laplacian construction is a backbone-level concern. Centralizing it
    # here means every registered basis sees the *same* operator and only
    # the backbone has to think about normalization.
    # ------------------------------------------------------------------ #
    def _build_laplacian_apply(
        self,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        num_nodes: int,
    ) -> LaplacianApply:
        r"""Build a closure that applies ``L̃`` to a node-feature tensor.

        Uses :func:`torch_geometric.utils.get_laplacian` for the
        normalization, matching the operator definition Liao Appendix B
        writes for every variable-basis entry.

        Parameters
        ----------
        edge_index : Tensor, shape ``[2, E]``
            Edge index in COO format.
        edge_weight : Tensor or None, shape ``[E]``
            Edge weights for the underlying adjacency. ``None`` means
            unit weights.
        num_nodes : int
            Number of nodes ``N`` in the graph.

        Returns
        -------
        LaplacianApply
            Closure ``h -> L̃ @ h`` mapping ``[N, F]`` to ``[N, F]``.
        """
        norm = None if self.laplacian_norm == "none" else self.laplacian_norm
        ei, ew = get_laplacian(
            edge_index,
            edge_weight,
            normalization=norm,
            num_nodes=num_nodes,
        )
        src, dst = ei[0], ei[1]

        def apply(h: Tensor) -> Tensor:
            r"""Apply the normalized Laplacian to ``h`` via a scatter-add.

            Parameters
            ----------
            h : Tensor, shape ``[N, F]``
                Node features to multiply by ``L̃``.

            Returns
            -------
            Tensor, shape ``[N, F]``
                ``L̃ @ h``.
            """
            # (L̃ h)_i = Σ_j L̃_ij · h_j, dispatched as a scatter-add over
            # the Laplacian's edge list. The edge weights ew now encode L̃
            # itself (including the diagonal), so this is one matvec.
            msg = ew.view(-1, 1) * h.index_select(0, src)
            return scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")

        return apply


def _build_mlp(
    in_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
) -> nn.Module:
    r"""Build a small MLP from ``Linear`` / ``ReLU`` / ``Dropout`` blocks.

    ``n_layers == 1`` is a single ``nn.Linear`` (no activation, no
    dropout). For ``n_layers > 1`` the head is a plain ``Linear``
    too: activations and dropout sit only between hidden layers.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension; also the hidden width of every
        intermediate layer.
    n_layers : int
        Total number of ``Linear`` layers.
    dropout : float
        Dropout probability applied between hidden layers.

    Returns
    -------
    nn.Module
        The constructed MLP: either a single ``nn.Linear`` or an
        ``nn.Sequential``.
    """
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(cur, out_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        cur = out_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)
