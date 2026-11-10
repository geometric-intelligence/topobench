r"""FilterBankGNN: Q parallel polynomial filters on the Laplacian, fused.

A filter bank is structurally different from a single polynomial filter:
it runs Q channels in parallel and fuses them with channel weights ``γ``
(Liao et al. 2024, Eq. (3)):

.. math::

    y = \mathrm{post}\!\Big( \mathrm{fuse}_q\big( \gamma_q\, g_q(\tilde L)\,
        \mathrm{pre}(x) \big) \Big),
    \qquad
    g_q(\tilde L) = \sum_{k=0}^{K} \theta_{q,k}\, T_q^{(k)}(\tilde L).

Each channel ``g_q`` is a single polynomial filter and reuses the
basis registry; the backbone owns the channel weights ``γ`` and the
pre/post MLPs, and delegates combination to a swappable
:class:`~topobench.nn.backbones.graph.filter_bank.fusion.Fusion`.

The published filter-bank models are thin subclasses that build their
channel list in ``__init__`` (mirroring how ``Legendre`` is a thin
subclass of ``Jacobi`` in the polynomial-filter registry):

- :class:`ACMGNN` -- LP + HP + identity linear filters (Luan et al. 2022).
- :class:`FiGURe` -- Q variable filters with learnable per-channel
  coefficients (Ekbote et al. 2023).
- :class:`FAGCN` -- LP + HP linear filters with a scaling hyperparameter
  (Bo et al. 2021).
- :class:`GNNLFHF` -- two PPR-style channels with a linear prefactor
  (Zhu et al. 2021).

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Section 3.3 / Table 1 (Filter Bank).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from topobench.nn.backbones.graph.filter_bank.channel import (
    Channel,
    GaussianChannel,
    PPRChannel,
    build_laplacian_apply,
)
from topobench.nn.backbones.graph.filter_bank.fusion import Fusion, SumFusion
from topobench.nn.backbones.graph.poly_filter.bases.bernstein import Bernstein
from topobench.nn.backbones.graph.poly_filter.bases.chebyshev import Chebyshev
from topobench.nn.backbones.graph.poly_filter.bases.monomial import Monomial
from topobench.nn.backbones.graph.poly_filter.basis import Basis


class FilterBankGNN(nn.Module):
    r"""Filter bank of Q polynomial-filter channels with a fusion step.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Width of the channels (and the pre-MLP output).
    out_channels : int
        Number of output node features (post-MLP output).
    channels : list of Channel
        The Q channels; each is a polynomial filter ``g_q(L̃)`` built from
        a basis and its coefficients.
    fusion : Fusion, optional
        How to combine the channel outputs. Defaults to
        :class:`~topobench.nn.backbones.graph.filter_bank.fusion.SumFusion`.
    dropout : float, optional
        Dropout after the pre-MLP and inside multi-layer MLPs. Defaults
        to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Normalization used to build ``L̃``. Defaults to ``'sym'``.
    self_loops : bool, optional
        GCN self-loop renormalization for the ``'sym'`` Laplacian
        (Liao's convention). Defaults to ``True``.
    pre_mlp_layers : int, optional
        Layers in the pre-filter MLP. Defaults to ``1``.
    post_mlp_layers : int, optional
        Layers in the post-fusion MLP. Defaults to ``1``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        channels: list[Channel],
        fusion: Fusion | None = None,
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        self_loops: bool = True,
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("FilterBankGNN needs at least one channel")
        if laplacian_norm not in {"sym", "rw", "none"}:
            raise ValueError(
                "laplacian_norm must be one of 'sym', 'rw', 'none'; "
                f"got {laplacian_norm!r}"
            )

        self.channels = nn.ModuleList(channels)
        self.fusion = fusion if fusion is not None else SumFusion()
        self.laplacian_norm = laplacian_norm
        self.self_loops = self_loops

        # Channel weights γ_q, one learnable scalar per channel, init to 1
        # so every channel contributes equally at the start.
        self.gamma = nn.Parameter(torch.ones(len(channels)))

        self.pre = _build_mlp(
            in_channels, hidden_channels, pre_mlp_layers, dropout
        )
        self.post = _build_mlp(
            hidden_channels, out_channels, post_mlp_layers, dropout
        )
        self.dropout = nn.Dropout(dropout)

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
            Batch assignment vector; unused at the propagation level.
        edge_weight : Tensor, optional
            Edge weights used to build ``L̃``.
        **kwargs : dict
            Swallowed for compatibility with the wrapper call site.

        Returns
        -------
        Tensor, shape ``[N, out_channels]``
            Output node features.
        """
        h = self.pre(x)
        h = self.dropout(h)

        # One Laplacian closure per forward pass, shared by every channel.
        L_apply = build_laplacian_apply(
            edge_index,
            edge_weight,
            h.size(0),
            self.laplacian_norm,
            self.self_loops,
        )

        channel_outs = [channel(L_apply, h) for channel in self.channels]
        y = self.fusion(channel_outs, self.gamma)
        return self.post(y)


class ACMGNN(FilterBankGNN):
    r"""ACM-GNN: low-pass + high-pass + identity linear filters, fused.

    Three channels, each a degree-1 Monomial filter with fixed
    coefficients, combined by a learnable weighted sum (Liao et al. 2024,
    "Linear (LP, HP, ID)"):

    .. math::

        g(\tilde L; \gamma) = \gamma_1 (I - \tilde L) + \gamma_2 \tilde L
                              + \gamma_3 I .

    Concretely the three channels are Monomial filters with ``θ`` equal to
    ``[1, -1]`` (LP, ``I - L̃``), ``[0, 1]`` (HP, ``L̃``), and ``[1, 0]``
    (ID, ``I``); only the channel weights ``γ`` are learned.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature dimensions (see :class:`FilterBankGNN`).
    dropout : float, optional
        Dropout. Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Laplacian normalization. Defaults to ``'sym'``.
    pre_mlp_layers, post_mlp_layers : int, optional
        MLP depths. Default ``1``.

    References
    ----------
    Luan et al. (2022) *Revisiting Heterophily For Graph Neural Networks*
    (NeurIPS) -- primary reference for ACM-GNN. The spectral form is the
    "Linear (LP, HP, ID)" filter-bank entry of Liao et al. (2024),
    Table 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        channels = [
            Channel(Monomial(), K=1, theta=[1.0, -1.0]),  # LP: I - L̃
            Channel(Monomial(), K=1, theta=[0.0, 1.0]),  # HP: L̃
            Channel(Monomial(), K=1, theta=[1.0, 0.0]),  # ID: I
        ]
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            channels=channels,
            fusion=SumFusion(),
            dropout=dropout,
            laplacian_norm=laplacian_norm,
            pre_mlp_layers=pre_mlp_layers,
            post_mlp_layers=post_mlp_layers,
        )


class FBGNN(FilterBankGNN):
    r"""FBGNN: low-pass + high-pass linear filters, fused.

    Two channels, each a degree-1 Monomial filter with fixed
    coefficients, combined by a learnable weighted sum (Liao et al. 2024,
    "Linear (LP, HP)"):

    .. math::

        g(\tilde L; \gamma) = \gamma_1 (I - \tilde L) + \gamma_2 \tilde L .

    The two channels are Monomial filters with ``θ`` equal to ``[1, -1]``
    (low-pass, ``I - L̃ = Â``) and ``[0, 1]`` (high-pass, ``L̃``); only the
    channel weights ``γ`` are learned. FBGNN introduced the filter-bank
    idea for graph heterophily; :class:`ACMGNN` extends it with a third
    identity channel.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature dimensions (see :class:`FilterBankGNN`).
    dropout : float, optional
        Dropout. Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Laplacian normalization. Defaults to ``'sym'``.
    pre_mlp_layers, post_mlp_layers : int, optional
        MLP depths. Default ``1``.

    Notes
    -----
    The original FBGNN mixes the low- and high-pass channels with a
    node-wise weight; this implementation ships the channel-weighted-sum
    spectral form of Liao et al. (2024), matching the scoping of
    :class:`ACMGNN`.

    References
    ----------
    Luan et al. (2022) *Complete the Missing Half: Augmenting Aggregation
    Filtering with Diversification for Graph Convolutional Networks* --
    primary reference for FBGNN. The spectral form is the "Linear (LP,
    HP)" filter-bank entry of Liao et al. (2024), Table 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        channels = [
            Channel(Monomial(), K=1, theta=[1.0, -1.0]),  # LP: I - L̃
            Channel(Monomial(), K=1, theta=[0.0, 1.0]),  # HP: L̃
        ]
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            channels=channels,
            fusion=SumFusion(),
            dropout=dropout,
            laplacian_norm=laplacian_norm,
            pre_mlp_layers=pre_mlp_layers,
            post_mlp_layers=post_mlp_layers,
        )


class FiGURe(FilterBankGNN):
    r"""FiGURe: Q variable filters with learnable per-channel coefficients.

    Each channel is a *full* polynomial filter of degree ``K`` with its
    own basis and its own **learnable** coefficients ``θ_{q,k}``; the
    channels are combined by a learnable weighted sum (Liao et al. 2024,
    Eq. (3) with summation fusion):

    .. math::

        g(\tilde L; \gamma, \theta) = \sum_{q=1}^{Q} \gamma_q
            \sum_{k=0}^{K} \theta_{q,k}\, T_q^{(k)}(\tilde L) .

    This is the variant with *variable* channels: unlike the fixed
    linear filters of ACM-GNN/FAGCN, every channel carries learnable
    ``θ`` and may use a different basis from the registry.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature dimensions (see :class:`FilterBankGNN`).
    K : int
        Polynomial degree shared by every channel.
    bases : list of Basis or None, optional
        One basis per channel. Defaults to the canonical FiGURe set
        ``[Monomial(), Chebyshev(), Bernstein(K)]``. Pass distinct
        instances; bases that own parameters (FavardGNN, ChebNetII,
        Bernstein) must not be shared across channels.
    dropout : float, optional
        Dropout. Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Laplacian normalization. Defaults to ``'sym'``.
    pre_mlp_layers, post_mlp_layers : int, optional
        MLP depths. Default ``1``.

    Notes
    -----
    The original FiGURe (Ekbote et al. 2023) (i) composes Monomial,
    Chebyshev, and Bernstein channels and (ii) pretrains the channel
    embeddings unsupervised. This implementation uses that canonical
    Monomial/Chebyshev/Bernstein channel set and scopes to the
    supervised filter-bank backbone; the unsupervised channel-embedding
    pretraining is out of scope. Liao's official implementation likewise
    composes ``AdjConv, ChebConv, BernConv`` with no Identity channel.

    References
    ----------
    Ekbote et al. (2023) *FiGURe: Simple and Efficient Unsupervised Node
    Representations with Filter Augmentations* (NeurIPS) -- primary
    reference. The spectral form is the "Mono, Cheb, Bern, ID"
    filter-bank entry of Liao et al. (2024), Table 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int,
        bases: list[Basis] | None = None,
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        if bases is None:
            bases = [Monomial(), Chebyshev(), Bernstein(K=K)]
        # Each channel is a variable filter: learnable theta (theta=None).
        channels = [Channel(basis, K=K, theta=None) for basis in bases]
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            channels=channels,
            fusion=SumFusion(),
            dropout=dropout,
            laplacian_norm=laplacian_norm,
            pre_mlp_layers=pre_mlp_layers,
            post_mlp_layers=post_mlp_layers,
        )


class FAGCN(FilterBankGNN):
    r"""FAGCN: low-pass + high-pass linear filters with a scaling term.

    Two degree-1 Monomial channels with fixed coefficients set by a
    scaling hyperparameter ``β``, combined by a learnable weighted sum
    (Liao et al. 2024, "Linear (LP, HP)" with parameter ``β``):

    .. math::

        g(\tilde L; \gamma) = \gamma_1\big((\beta + 1) I - \tilde L\big)
                            + \gamma_2\big((\beta - 1) I + \tilde L\big) .

    The two channels are Monomial filters with ``θ`` equal to
    ``[β + 1, -1]`` (low-pass) and ``[β - 1, +1]`` (high-pass); only the
    channel weights ``γ`` are learned.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature dimensions (see :class:`FilterBankGNN`).
    beta : float, optional
        Scaling coefficient ``β`` (the residual strength; ``β`` in
        ``[0, 1]`` in the original work). Defaults to ``0.3``.
    dropout : float, optional
        Dropout. Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Laplacian normalization. Defaults to ``'sym'``.
    pre_mlp_layers, post_mlp_layers : int, optional
        MLP depths. Default ``1``.

    Notes
    -----
    The original FAGCN gates the two channels with an **edge-level
    attention** mechanism. This implementation ships the equivalent
    spectral form of Liao et al. (2024), a learnable channel-wise
    weighted sum (``γ_1, γ_2``), and scopes out the edge-attention
    gating; an attention-based ``Fusion`` could be added later without
    changing the backbone.

    References
    ----------
    Bo et al. (2021) *Beyond Low-Frequency Information in Graph
    Convolutional Networks* (AAAI) -- primary reference for FAGCN. The
    spectral form is the "Linear (LP, HP)" filter-bank entry (with
    parameter ``β``) of Liao et al. (2024), Table 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        beta: float = 0.3,
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        b = float(beta)
        channels = [
            Channel(Monomial(), K=1, theta=[b + 1.0, -1.0]),  # LP: (β+1)I - L̃
            Channel(Monomial(), K=1, theta=[b - 1.0, 1.0]),  # HP: (β-1)I + L̃
        ]
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            channels=channels,
            fusion=SumFusion(),
            dropout=dropout,
            laplacian_norm=laplacian_norm,
            pre_mlp_layers=pre_mlp_layers,
            post_mlp_layers=post_mlp_layers,
        )


class GNNLFHF(FilterBankGNN):
    r"""GNN-LF/HF: two PPR channels with linear prefactors, fused.

    A bank of two :class:`PPRChannel` filters, a low-frequency and a
    high-frequency channel, combined by a learnable weighted sum
    (Liao et al. 2024, "PPR (LP, HP)"):

    .. math::

        g(\tilde L; \gamma) = \sum_{q \in \{LF, HF\}} \gamma_q\,
            (I + \beta_q \tilde L) \sum_{k=0}^{K}
            \alpha_q (1 - \alpha_q)^k (I - \tilde L)^k .

    Each channel's ``(α_q, β_q)`` are fixed hyperparameters; only the
    channel weights ``γ`` are learned. The defaults give one low-pass
    channel (``β = 0``, plain PPR) and one channel with added high-pass
    content (``β > 0``).

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature dimensions (see :class:`FilterBankGNN`).
    K : int, optional
        PPR propagation steps per channel. Defaults to ``10``.
    alphas : tuple of float, optional
        PPR teleport probabilities ``(α_LF, α_HF)``. Defaults to
        ``(0.1, 0.1)``.
    betas : tuple of float, optional
        Prefactor scalings ``(β_LF, β_HF)`` in the ``(I + β L̃)`` factor.
        Matching Liao's GNN-LF/HF, the low-frequency channel takes a
        positive ``β`` and the high-frequency channel a negative ``β``.
        Defaults to ``(0.2, -0.5)``.
    dropout : float, optional
        Dropout. Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Laplacian normalization. Defaults to ``'sym'``.
    pre_mlp_layers, post_mlp_layers : int, optional
        MLP depths. Default ``1``.

    Notes
    -----
    This implements Liao et al. (2024)'s uniform "PPR (LP, HP)" spectral
    form. Liao's released code uses a positive prefactor ``β`` for the
    low-frequency channel and a negative ``β`` for the high-frequency
    channel (``compose_param`` ranges ``(0.01, 1.00)`` and
    ``(-1.00, -0.01)``); the defaults follow that convention and are
    meant to be swept.

    References
    ----------
    Zhu et al. (2021) *Interpreting and Unifying Graph Neural Networks
    with An Optimization Framework* (WWW) -- primary reference for
    GNN-LF/HF, the "PPR (LP, HP)" filter-bank entry of Liao et al.
    (2024), Table 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alphas: tuple[float, float] = (0.1, 0.1),
        betas: tuple[float, float] = (0.2, -0.5),
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        channels = [
            PPRChannel(K=K, alpha=alphas[0], beta=betas[0]),  # LF
            PPRChannel(K=K, alpha=alphas[1], beta=betas[1]),  # HF
        ]
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            channels=channels,
            fusion=SumFusion(),
            dropout=dropout,
            laplacian_norm=laplacian_norm,
            pre_mlp_layers=pre_mlp_layers,
            post_mlp_layers=post_mlp_layers,
        )


class G2CN(FilterBankGNN):
    r"""G²CN: two Gaussian band-pass channels, fused.

    A bank of two :class:`GaussianChannel` filters, a low-frequency and a
    high-frequency channel, combined by a learnable weighted sum
    (Liao et al. 2024, "Gaussian"):

    .. math::

        g(\tilde L; \gamma) = \gamma_1 \sum_{k=0}^{K} \frac{\alpha^k}{k!}
            \big((1 + \beta_1) I - \tilde L\big)^{2k}
          + \gamma_2 \sum_{k=0}^{K} \frac{\alpha^k}{k!}
            \big((1 - \beta_2) I - \tilde L\big)^{2k} .

    Each channel is a truncated Gaussian (heat-kernel-like) filter
    concentrated at a frequency set by its shift; the decay ``α`` and the
    shifts ``β₁, β₂`` are fixed hyperparameters, only the channel weights
    ``γ`` are learned.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature dimensions (see :class:`FilterBankGNN`).
    K : int, optional
        Number of Gaussian terms per channel. Defaults to ``5``.
    alpha : float, optional
        Exponential decay ``α`` of the term weights. Defaults to ``1.0``.
    betas : tuple of float, optional
        Shifts ``(β₁, β₂)``: the low-frequency channel uses
        ``(1 + β₁) I - L̃`` and the high-frequency channel
        ``(1 - β₂) I - L̃``. Defaults to ``(0.5, 0.5)`` (``β₂ in (0, 1)``
        keeps the high-frequency shift positive, matching Liao's range).
    dropout : float, optional
        Dropout. Defaults to ``0.0``.
    laplacian_norm : {'sym', 'rw', 'none'}, optional
        Laplacian normalization. Defaults to ``'sym'``.
    pre_mlp_layers, post_mlp_layers : int, optional
        MLP depths. Default ``1``.

    Notes
    -----
    This implements Liao et al. (2024)'s "Gaussian" spectral form: each
    term is a *squared* shifted operator ``((1 ± β) I - L̃)^2`` (the 2-hop
    propagation of G²CN) weighted by ``α^k / k!``, summed over ``K + 1``
    terms. The ``(α, β)`` defaults are illustrative and meant to be swept.

    References
    ----------
    Li et al. (2022) *G²CN: Graph Gaussian Convolution Networks with
    Concentrated Graph Filters* (ICML) -- primary reference for G²CN, the
    "Gaussian" filter-bank entry of Liao et al. (2024), Table 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 5,
        alpha: float = 1.0,
        betas: tuple[float, float] = (0.5, 0.5),
        dropout: float = 0.0,
        laplacian_norm: str = "sym",
        pre_mlp_layers: int = 1,
        post_mlp_layers: int = 1,
    ):
        channels = [
            GaussianChannel(K=K, alpha=alpha, shift=1.0 + betas[0]),  # LF
            GaussianChannel(K=K, alpha=alpha, shift=1.0 - betas[1]),  # HF
        ]
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            channels=channels,
            fusion=SumFusion(),
            dropout=dropout,
            laplacian_norm=laplacian_norm,
            pre_mlp_layers=pre_mlp_layers,
            post_mlp_layers=post_mlp_layers,
        )


def _build_mlp(
    in_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
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
    dropout : float
        Dropout probability between hidden layers.

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
        layers.append(nn.Linear(cur, out_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        cur = out_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)
