"""TRAWL: topological random-walk sequence backbone.

TRAWL samples random walks over the cells of a lifted complex and encodes the
walk sequences with configurable sequence layers (Mamba, SISA, hybrid,
transformer, GRU, MLP or any Hydra-instantiated module).
"""

import copy
import math

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from topobench.data.utils.trawl.sampling import WalkSampler, walk_seed

__all__ = ["TRAWL"]


# ---------------------------------------------------------------------------
# Sequence layers
# ---------------------------------------------------------------------------


class SequenceBlock(nn.Module):
    """Pre-normalized residual adapter for sequence-shaped modules.

    Parameters
    ----------
    module : torch.nn.Module
        Module mapping ``[walk, time, hidden]`` tensors to the same shape. A
        tuple output is reduced to its first element.
    d_model : int
        Hidden dimension used by the pre-normalization layer.
    """

    def __init__(self, module, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.module = module

    def residual(self, x):
        """Compute the module update on the normalized input.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Residual update with the same shape as ``x``.
        """
        result = self.module(self.norm(x))
        if isinstance(result, tuple):
            result = result[0]
        if result.shape != x.shape:
            raise ValueError(
                "Custom sequence layers must preserve [walk, time, hidden] shape"
            )
        return result

    def forward(self, x, residual_only=False):
        """Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[walk, time, hidden]``.
        residual_only : bool, optional
            If True, return only the update instead of ``x + update``
            (default: False).

        Returns
        -------
        torch.Tensor
            Updated sequences, or the update alone, shaped like ``x``.
        """
        update = self.residual(x)
        return update if residual_only else x + update


class WalkGraphLayer(nn.Module):
    """Apply a PyG-compatible graph module to disjoint walk path graphs.

    Parameters
    ----------
    module : torch.nn.Module
        Graph module called as ``module(x, edge_index)``.
    bidirectional : bool, optional
        Whether path edges are added in both directions (default: True).
    """

    def __init__(self, module, bidirectional=True):
        super().__init__()
        self.module, self.bidirectional = module, bidirectional

    def forward(self, x):
        """Run the graph module on each walk viewed as a path graph.

        Parameters
        ----------
        x : torch.Tensor
            Walk states of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Updated walk states with the same shape as ``x``.
        """
        walks, length, hidden = x.shape
        sources = (
            torch.arange(walks * length, device=x.device)
            .reshape(walks, length)[:, :-1]
            .flatten()
        )
        edges = torch.stack((sources, sources + 1))
        if self.bidirectional:
            edges = torch.cat((edges, edges.flip(0)), dim=1)
        return self.module(x.reshape(-1, hidden), edges).reshape(
            walks, length, hidden
        )


def feed_forward(d_model, expansion=2, dropout=0.0):
    """Two-layer GELU feed-forward network.

    Parameters
    ----------
    d_model : int
        Input and output width.
    expansion : int, optional
        Hidden width multiplier (default: 2).
    dropout : float, optional
        Dropout between the two layers (default: 0.0).

    Returns
    -------
    torch.nn.Sequential
        The network.
    """
    return nn.Sequential(
        nn.Linear(d_model, expansion * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * d_model, d_model),
    )


def make_layer(config, d_model):
    """Construct a layer from a short name or a Hydra target.

    Parameters
    ----------
    config : str or dict
        Layer kind (``"sisa"``, ``"mamba"``, ``"gru"``, ``"mlp"``,
        ``"transformer"`` or ``"graph"``), or a mapping with a ``kind`` key
        plus layer options, or a Hydra config with a ``_target_`` key.
    d_model : int
        Hidden dimension of the layer.

    Returns
    -------
    torch.nn.Module
        Sequence layer preserving ``[walk, time, hidden]`` shapes.
    """
    config = {"kind": config} if isinstance(config, str) else dict(config)
    if "_target_" in config:
        from hydra.utils import instantiate

        return SequenceBlock(instantiate(config), d_model)
    kind = config.pop("kind")
    if kind == "sisa":
        return SISABlock(d_model=d_model, **config)
    if kind == "mamba":
        backend = config.pop("backend", "torch")
        if backend == "torch":
            module = PureTorchMambaBlock(d_model=d_model, **config)
        elif backend == "mamba_ssm":
            from mamba_ssm import Mamba

            # Base Hydra configs retain the pure-PyTorch scan setting when
            # switching backends. Official Mamba manages its own CUDA scan.
            config.pop("scan", None)
            module = Mamba(d_model=d_model, **config)
        else:
            raise ValueError(f"Unknown Mamba backend {backend}")
    elif kind == "gru":
        module = nn.GRU(d_model, d_model, batch_first=True, **config)
    elif kind == "mlp":
        module = feed_forward(d_model, **config)
    elif kind == "transformer":
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.pop("n_heads", 8),
            batch_first=True,
            norm_first=True,
            **config,
        )
    elif kind == "graph":
        from hydra.utils import instantiate

        module = WalkGraphLayer(instantiate(config.pop("module")), **config)
    else:
        raise ValueError(f"Unknown TRAWL sequence layer {kind}")
    return SequenceBlock(module, d_model)


class PureTorchMambaBlock(nn.Module):
    """Selective SSM block for when mamba-ssm is unavailable.

    Parameters
    ----------
    d_model : int
        Input and output hidden dimension.
    d_state : int, optional
        SSM state dimension (default: 16).
    d_conv : int, optional
        Kernel size of the causal depthwise convolution (default: 4).
    expand : int, optional
        Expansion factor of the inner dimension (default: 2).
    scan : str, optional
        Scan implementation, ``"parallel"`` or ``"sequential"``
        (default: "parallel").
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scan: str = "parallel",
    ):
        super().__init__()
        if scan not in {"parallel", "sequential"}:
            raise ValueError("scan must be parallel or sequential")
        self.scan = scan
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selective SSM block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[batch, length, d_model]``.

        Returns
        -------
        torch.Tensor
            Output sequences of shape ``[batch, length, d_model]``.
        """
        _, seqlen, _ = x.shape
        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :seqlen]
        x_conv = F.silu(x_conv.transpose(1, 2))

        x_dbl = self.x_proj(x_conv)
        B = x_dbl[..., : self.d_state]
        C = x_dbl[..., self.d_state : 2 * self.d_state]
        dt = F.softplus(self.dt_proj(x_dbl[..., -1:]))
        A = -torch.exp(self.A_log.float())

        y = self._selective_scan(x_conv, dt, A, B, C)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C):
        """Run the selective scan, sequentially or as a parallel prefix scan.

        Parameters
        ----------
        u : torch.Tensor
            Inputs, ``[batch, length, d_inner]``.
        dt : torch.Tensor
            Positive step sizes, ``[batch, length, d_inner]``.
        A : torch.Tensor
            Transition rates, ``[d_inner, d_state]``.
        B : torch.Tensor
            Input projections, ``[batch, length, d_state]``.
        C : torch.Tensor
            Output projections, ``[batch, length, d_state]``.

        Returns
        -------
        torch.Tensor
            Outputs, ``[batch, length, d_inner]``.
        """
        bsz, seqlen, d_inner = u.shape
        deltaA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        use_seq = self.scan == "sequential"
        if use_seq:
            # Preallocate output to avoid Python list + torch.stack overhead.
            n = A.shape[1]
            h = u.new_zeros(bsz, d_inner, n)
            ys = u.new_zeros(bsz, seqlen, d_inner)
            for t in range(seqlen):
                h = deltaA[:, t] * h + deltaB_u[:, t]
                ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)
            return ys

        # Parallel associative scan (log-depth)
        n_pad = 1 << ((seqlen - 1).bit_length())
        if n_pad != seqlen:
            deltaA = F.pad(deltaA, (0, 0, 0, 0, 0, n_pad - seqlen), value=1.0)
            deltaB_u = F.pad(
                deltaB_u, (0, 0, 0, 0, 0, n_pad - seqlen), value=0.0
            )

        k = 1
        while k < n_pad:
            A_prev = F.pad(deltaA[:, :-k], (0, 0, 0, 0, k, 0), value=1.0)
            X_prev = F.pad(deltaB_u[:, :-k], (0, 0, 0, 0, k, 0), value=0.0)
            deltaB_u = deltaA * X_prev + deltaB_u
            deltaA = deltaA * A_prev
            k *= 2

        h = deltaB_u[:, :seqlen]
        return (h * C.unsqueeze(2)).sum(-1)


def sequence_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Cumulative sum that stays deterministic on CUDA when requested.

    CUDA ``cumsum`` has no deterministic kernel. Under deterministic mode the
    short walk dimension is summed with a float32 lower-triangular matmul.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which to accumulate.

    Returns
    -------
    torch.Tensor
        Cumulative sum of ``x`` along ``dim`` with the dtype of ``x``.
    """
    if not (x.is_cuda and torch.are_deterministic_algorithms_enabled()):
        return torch.cumsum(x, dim=dim)
    length = x.shape[dim]
    with torch.autocast(x.device.type, enabled=False):
        lower = torch.ones(
            length, length, device=x.device, dtype=torch.float32
        ).tril()
        moved = x.movedim(dim, -1).float() @ lower.T
    return moved.movedim(-1, dim).to(x.dtype)


def build_rope_cache(seq_len: int, dim: int, device, dtype):
    """Build rotary position embedding cosine and sine tables.

    Parameters
    ----------
    seq_len : int
        Number of positions.
    dim : int
        Rotary dimension; ``dim // 2`` frequencies are used.
    device : torch.device
        Device of the returned tables.
    dtype : torch.dtype
        Dtype of the returned tables.

    Returns
    -------
    tuple of torch.Tensor
        Cosine and sine tables, each of shape ``[1, 1, seq_len, dim // 2]``.
    """
    half = dim // 2
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(half, device=device, dtype=torch.float32) / half)
    )
    freqs = torch.outer(
        torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq
    )
    return freqs.cos().to(dtype)[None, None], freqs.sin().to(dtype)[None, None]


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate interleaved feature pairs by the given phases.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``[..., dim]``.
    cos : torch.Tensor
        Phase cosines broadcastable to ``[..., dim // 2]``.
    sin : torch.Tensor
        Phase sines broadcastable to ``[..., dim // 2]``.

    Returns
    -------
    torch.Tensor
        Rotated tensor with the same shape as ``x``.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
    ).flatten(-2)


class SISALayer(nn.Module):
    """Multi-head state-space augmented causal attention.

    Parameters
    ----------
    d_model : int
        Input and output hidden dimension; must be divisible by ``n_heads``.
    n_heads : int, optional
        Number of attention heads (default: 8).
    d_ssm : int, optional
        Per-head state-space key dimension; must be even (default: 16).
    attention_dropout : float, optional
        Dropout probability on attention weights during training
        (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ssm: int = 16,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.attention_dropout = attention_dropout
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ssm = d_ssm
        if self.d_head % 2 or d_ssm % 2:
            raise ValueError("SISA RoPE dimensions must be even")
        h, dh, ds = n_heads, self.d_head, d_ssm
        self.qkv = nn.Linear(d_model, 3 * h * dh)
        self.b_proj = nn.Linear(d_model, h * ds)
        self.c_proj = nn.Linear(d_model, h * ds)
        self.alpha_proj = nn.Linear(d_model, h)
        self.theta_proj = nn.Linear(d_model, h * (ds // 2))
        self.lambd = nn.Parameter(torch.full((h,), 0.01))
        self.out_proj = nn.Linear(h * dh, d_model)

        # alpha=exp(-softplus(-5)) ~= .9933: about a 103-step half-life.
        # Unlike the old Zinc clamp formulation, this starts with nonzero gradients.
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -5.0)
        nn.init.normal_(self.theta_proj.weight, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)

    # Beyond this cumulative log-decay span the centered factors exp(+-span/2)
    # and masked products exp(span) leave the safe fp32/bf16 range.
    MAX_FACTORED_SPAN = 60.0

    def _explicit_attention(self, q, k, v, c_ssm, b_ssm, g, scale, dh):
        """Causal attention with the decay formed only for ``j <= i``.

        Stays finite under arbitrarily strong decay. It omits dropout so it
        draws no random numbers, and only replaces heads that would overflow.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Queries, keys and values, ``[batch, heads, length, d_head]``.
        c_ssm, b_ssm : torch.Tensor
            State-space factors, ``[batch, heads, length, d_ssm]``.
        g : torch.Tensor
            Cumulative log-decay, ``[batch, heads, length, 1]``.
        scale : torch.Tensor
            Per-head state-space scale, ``[1, heads, 1, 1]``.
        dh : int
            Head dimension.

        Returns
        -------
        torch.Tensor
            Attention outputs, ``[batch, heads, length, d_head]``.
        """
        length = q.shape[-2]
        causal = torch.ones(
            length, length, dtype=torch.bool, device=q.device
        ).tril()
        difference = (g - g.transpose(-1, -2)).masked_fill(~causal, 0.0)
        decay = torch.exp(difference) * causal
        state = (scale * c_ssm) @ (scale * b_ssm).transpose(-1, -2)
        logits = q @ k.transpose(-1, -2) + state * decay.to(state)
        logits = (logits / math.sqrt(dh)).masked_fill(~causal, -torch.inf)
        return logits.float().softmax(dim=-1).to(v.dtype) @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply state-space augmented causal attention.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[batch, length, d_model]``.

        Returns
        -------
        torch.Tensor
            Output sequences of shape ``[batch, length, d_model]``.
        """
        bsz, length, _ = x.shape
        h, dh, ds = self.n_heads, self.d_head, self.d_ssm
        qkv = self.qkv(x).view(bsz, length, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = build_rope_cache(length, dh, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        b_ssm = self.b_proj(x).view(bsz, length, h, ds).permute(0, 2, 1, 3)
        c_ssm = self.c_proj(x).view(bsz, length, h, ds).permute(0, 2, 1, 3)
        theta = (
            self.theta_proj(x)
            .view(bsz, length, h, ds // 2)
            .permute(0, 2, 1, 3)
        )
        phase = sequence_cumsum(theta, dim=2)
        cos, sin = phase.cos(), phase.sin()
        b_ssm, c_ssm = apply_rope(b_ssm, cos, sin), apply_rope(c_ssm, cos, sin)

        decay = F.softplus(self.alpha_proj(x))
        alpha = torch.exp(-decay).clamp(1e-4, 0.9999)
        alpha = alpha.permute(0, 2, 1).unsqueeze(-1)
        g = sequence_cumsum(torch.log(alpha), dim=2)
        scale = (dh**0.25) * torch.sqrt(F.softplus(self.lambd) + 1e-6)
        scale = scale.view(1, h, 1, 1)
        # Per-head flag, computed without a host sync so compiled graphs stay
        # whole. Wide heads use the explicit path; the factored path sees a
        # neutral decay there so neither branch (nor its gradient) overflows.
        span = g.amax(dim=2, keepdim=True) - g.amin(dim=2, keepdim=True)
        wide = span.detach() > self.MAX_FACTORED_SPAN
        explicit = self._explicit_attention(
            q, k, v, c_ssm, b_ssm, g, scale, dh
        )
        g = torch.where(wide, torch.zeros_like(g), g)
        # Center the two exponential factors to avoid overflow without changing
        # their pairwise product exp(g_i - g_j).
        center = (
            g.detach().amin(dim=2, keepdim=True)
            + g.detach().amax(dim=2, keepdim=True)
        ) / 2
        c_bar = torch.exp(g - center) * c_ssm
        b_bar = torch.exp(-g + center) * b_ssm

        q_aug = torch.cat((q, scale * c_bar), dim=-1)
        k_aug = torch.cat((k, scale * b_bar), dim=-1)
        y = F.scaled_dot_product_attention(
            q_aug,
            k_aug,
            v,
            is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(dh),
        )
        y = torch.where(wide, explicit.to(y.dtype), y)
        return self.out_proj(
            y.transpose(1, 2).contiguous().view(bsz, length, h * dh)
        )


class SISABlock(nn.Module):
    """Pre-normalized SISA attention followed by a feed-forward network.

    Parameters
    ----------
    d_model : int
        Input and output hidden dimension.
    n_heads : int, optional
        Number of attention heads (default: 8).
    d_ssm : int, optional
        Per-head state-space key dimension (default: 16).
    attention_dropout : float, optional
        Dropout probability on attention weights (default: 0.1).
    dropout : float, optional
        Dropout probability inside the feed-forward network (default: 0.3).
    expansion : int, optional
        Hidden expansion factor of the feed-forward network (default: 2).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ssm: int = 16,
        attention_dropout: float = 0.1,
        dropout: float = 0.3,
        expansion: int = 2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.sisa = SISALayer(
            d_model,
            n_heads=n_heads,
            d_ssm=d_ssm,
            attention_dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = feed_forward(d_model, expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SISA block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[batch, length, d_model]``.

        Returns
        -------
        torch.Tensor
            Output sequences of shape ``[batch, length, d_model]``.
        """
        x = x + self.sisa(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# ---------------------------------------------------------------------------
# Base TRAWL backbone
# ---------------------------------------------------------------------------


def upload(array, device):
    """Copy a host array to ``device`` without making the host wait.

    Integer arrays become int64 on every platform. CUDA copies go through
    pinned memory with ``non_blocking`` so they never synchronize.

    Parameters
    ----------
    array : array_like
        Host array to copy.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor
        Tensor on ``device``; integer and boolean arrays become int64.
    """
    array = np.ascontiguousarray(array)
    if array.dtype.kind in "iub":
        array = array.astype(np.int64, copy=False)
    tensor = torch.from_numpy(array)
    if torch.device(device).type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


class NeighborhoodFusion(nn.Module):
    """Fuse available neighborhood embeddings, preserving empty-slot semantics.

    Parameters
    ----------
    dim : int
        Dimension of each neighborhood embedding.
    count : int
        Number of neighborhoods.
    mode : str, optional
        Fusion mode: ``"mean"``, ``"concat"``, ``"learned"`` or
        ``"attention"`` (default: "mean").
    """

    def __init__(self, dim, count, mode="mean"):
        super().__init__()
        if mode not in {"mean", "concat", "learned", "attention"}:
            raise ValueError(f"Unknown neighborhood fusion {mode}")
        self.mode = mode
        self.weights = (
            nn.Parameter(torch.zeros(count)) if mode == "learned" else None
        )
        self.attention = nn.Linear(dim, 1) if mode == "attention" else None
        self.project = (
            nn.Linear(count * (dim + 1), dim) if mode == "concat" else None
        )

    def forward(self, values, available, fallback):
        """Fuse neighborhood embeddings, ignoring unavailable slots.

        Parameters
        ----------
        values : torch.Tensor
            Neighborhood embeddings of shape ``[count, dim]``.
        available : torch.Tensor
            Boolean mask of shape ``[count]`` marking neighborhoods with walks.
        fallback : torch.Tensor
            Embedding of shape ``[dim]`` returned when none is available.

        Returns
        -------
        torch.Tensor
            Fused embedding of shape ``[dim]``.
        """
        if not bool(available.any()):
            return fallback
        if self.mode == "concat":
            slots = torch.cat(
                (values * available[:, None], available[:, None].to(values)),
                dim=1,
            )
            return self.project(slots.flatten())
        if self.mode == "mean":
            scores = values.new_zeros(len(values))
        elif self.mode == "learned":
            scores = self.weights
        else:
            scores = self.attention(values).squeeze(-1)
        weights = scores.masked_fill(~available, -torch.inf).softmax(dim=0)
        return (values * weights[:, None]).sum(dim=0)


class TRAWL(nn.Module):
    """Encode random walks over lifted cells with configurable sequence layers.

    Consumes the output of ``TRAWLTransform`` and returns contextual cell
    features ``x_r`` plus walk and graph embeddings for ``TRAWLReadout``.
    Input widths left as None are inferred by ``initialize``.

    Parameters
    ----------
    hidden_dim : int, optional
        Width of walk states (default: 128).
    max_rank : int, optional
        Highest cell rank (default: 2).
    in_channels : list of int, optional
        Feature width per rank; inferred if None (default: None).
    pe_dim : int, optional
        Positional encoding width; inferred if None (default: None).
    layers : list, optional
        Explicit ``make_layer`` configs; overrides ``architecture`` and
        ``depth`` (default: None).
    architecture : str, optional
        ``"mamba"``, ``"sisa"``, ``"hybrid"`` (alternating), ``"gru"``,
        ``"transformer"`` or ``"mlp"`` (default: "hybrid").
    depth : int, optional
        Number of preset layers (default: 5).
    layer_options : dict, optional
        Options per layer kind for presets (default: None).
    walks : dict, optional
        ``sample_neighbors`` and guidance options
        (default: ``{"k": 32, "length": 32}``).
    walk_scope : str, optional
        ``"union"`` walks all neighborhoods jointly, ``"separate"`` walks
        each one (default: "union").
    num_neighborhoods : int, optional
        Number of relations built by the transform (default: 2).
    encoder_sharing : str, optional
        ``"shared"`` or ``"independent"`` encoders for separate walks
        (default: "shared").
    fusion : str, optional
        ``NeighborhoodFusion`` mode for separate walks (default: "mean").
    pooling : str, optional
        Time pooling: ``"mean"``, ``"max"`` or ``"mean_max"``
        (default: "mean_max").
    rank_embedding : bool, optional
        Add a learned rank embedding (default: True).
    num_colors : int, optional
        Number of color embeddings; 0 disables them (default: 0).
    move_embedding : bool, optional
        Embed rank moves (down, same, up) along walks (default: True).
    dropout : float, optional
        Dropout on walk inputs (default: 0.0).
    seed : int, optional
        Base seed for walk sampling (default: 42).
    eval_views : int, optional
        Walk views averaged at evaluation (default: 1).
    checkpoint_layers : bool, optional
        Activation checkpointing during training (default: False).
    walk_refresh : str, optional
        ``"train"`` resamples walks every step, ``"fixed"`` never does
        (default: "train").
    occurrence_pooling : str, optional
        Pooling of a cell's walk occurrences: ``"mean"`` or
        ``"attention"`` (default: "mean").
    graph_readout : str, optional
        Graph embedding from ``"walks"`` or ``"cells"`` (default: "walks").
    transition_cache_size : int, optional
        Transition distributions cached by the sampler (default: 1024).
    """

    def __init__(
        self,
        hidden_dim=128,
        max_rank=2,
        in_channels=None,
        pe_dim=None,
        layers=None,
        architecture="hybrid",
        depth=5,
        layer_options=None,
        walks=None,
        walk_scope="union",
        num_neighborhoods=2,
        encoder_sharing="shared",
        fusion="mean",
        pooling="mean_max",
        rank_embedding=True,
        num_colors=0,
        move_embedding=True,
        dropout=0.0,
        seed=42,
        eval_views=1,
        checkpoint_layers=False,
        walk_refresh="train",
        occurrence_pooling="mean",
        graph_readout="walks",
        transition_cache_size=1024,
    ):
        super().__init__()
        choices = {
            "walk_scope": (walk_scope, {"union", "separate"}),
            "encoder_sharing": (encoder_sharing, {"shared", "independent"}),
            "pooling": (pooling, {"mean", "max", "mean_max"}),
            "walk_refresh": (walk_refresh, {"train", "fixed"}),
            "occurrence_pooling": (occurrence_pooling, {"mean", "attention"}),
            "graph_readout": (graph_readout, {"walks", "cells"}),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if eval_views < 1 or depth < 1 or num_neighborhoods < 1:
            raise ValueError(
                "Depth, neighborhood count and eval_views must be positive"
            )
        self.hidden_dim, self.max_rank = hidden_dim, max_rank
        self.walk_sampler = WalkSampler(transition_cache_size)
        self.walk_scope, self.pooling = walk_scope, pooling
        self.num_neighborhoods = num_neighborhoods
        self.walks = dict(walks or {"k": 32, "length": 32})
        self.seed, self.eval_views = seed, eval_views
        self.checkpoint_layers = checkpoint_layers
        self.walk_refresh = walk_refresh
        self.occurrence_pooling = occurrence_pooling
        self.graph_readout = graph_readout
        self.occurrence_attention = (
            nn.Linear(hidden_dim, 1)
            if occurrence_pooling == "attention"
            else None
        )
        self.register_buffer(
            "sampling_step", torch.zeros((), dtype=torch.long)
        )
        # Host mirror of ``sampling_step``; reading the buffer would wait for
        # the device on every forward pass.
        self._host_sampling_step = None
        self.features = nn.ModuleList(
            [
                nn.Linear(in_channels[r], hidden_dim)
                if in_channels is not None
                else nn.LazyLinear(hidden_dim)
                for r in range(max_rank + 1)
            ]
        )
        self.position = (
            nn.LazyLinear(hidden_dim, bias=False)
            if pe_dim is None
            else nn.Linear(pe_dim, hidden_dim, bias=False)
        )
        self.rank = (
            nn.Embedding(max_rank + 1, hidden_dim) if rank_embedding else None
        )
        self.color = (
            nn.Embedding(num_colors, hidden_dim) if num_colors else None
        )
        self.move = nn.Embedding(3, hidden_dim) if move_embedding else None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        options = dict(layer_options or {})
        if layers is None:
            presets = {"mamba", "sisa", "hybrid", "gru", "transformer", "mlp"}
            if architecture not in presets:
                raise ValueError(
                    "Custom architecture requires an explicit layer list"
                )
            kinds = [
                ("mamba" if i % 2 == 0 else "sisa")
                if architecture == "hybrid"
                else architecture
                for i in range(depth)
            ]
            layers = [
                {"kind": kind, **dict(options.get(kind, {}))} for kind in kinds
            ]
        if not layers:
            raise ValueError("Provide at least one sequence layer")
        encoder = nn.ModuleList(
            [make_layer(config, hidden_dim) for config in layers]
        )
        count = (
            num_neighborhoods
            if walk_scope == "separate" and encoder_sharing == "independent"
            else 1
        )
        self.encoders = nn.ModuleList(
            [encoder] + [copy.deepcopy(encoder) for _ in range(count - 1)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        self.fusion = NeighborhoodFusion(
            self.output_dim, num_neighborhoods, fusion
        )

    def initialize(self, data_list):
        """Materialize projections from training data, including absent ranks.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed training graphs used to infer input widths.
        """
        if not data_list:
            raise ValueError(
                "Cannot infer TRAWL input widths from an empty dataset"
            )
        with torch.no_grad():
            for rank, projection in enumerate(self.features):
                if not isinstance(projection, nn.LazyLinear):
                    continue
                key = f"trawl_signal_{rank}"
                widths = {
                    data[key].shape[1] for data in data_list if len(data[key])
                }
                if len(widths) > 1:
                    raise ValueError(
                        f"Inconsistent rank-{rank} feature widths: {widths}"
                    )
                width = next(iter(widths), data_list[0][key].shape[1])
                projection(
                    torch.zeros(1, width, device=projection.weight.device)
                )
            if isinstance(self.position, nn.LazyLinear):
                self.position(
                    torch.zeros(
                        1,
                        data_list[0].trawl_pe.shape[1],
                        device=self.position.weight.device,
                    )
                )

    # Walk metadata read on the host (see ``TBModel.on_before_batch_transfer``).
    host_fields = (
        "trawl_counts",
        "trawl_edges",
        "trawl_weights",
        "trawl_identity",
        "trawl_relations",
        "trawl_colors",
    )

    def _host(self, batch):
        """Return CPU copies of ``host_fields`` without waiting for the device.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch, optionally carrying precomputed ``trawl_host`` arrays.

        Returns
        -------
        dict
            Mapping from field name to NumPy array.
        """
        host = getattr(batch, "trawl_host", None)
        if host is None:
            host = {
                key: batch[key].detach().cpu().numpy()
                for key in self.host_fields
                if key in batch
            }
        return host

    def next_sampling_step(self):
        """Return the training sampling step and advance it.

        Returns
        -------
        int
            Sampling step before the increment.
        """
        if self._host_sampling_step is None:
            self._host_sampling_step = int(self.sampling_step)
        step = self._host_sampling_step
        self._host_sampling_step += 1
        self.sampling_step.add_(1)
        return step

    def reset_sampling_step(self):
        """Restart walk refreshes, e.g. before supervised fitting."""
        self.sampling_step.zero_()
        self._host_sampling_step = 0

    def _load_from_state_dict(self, *args, **kwargs):
        """Load state and invalidate the host sampling-step mirror.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to ``nn.Module``.
        **kwargs : dict
            Keyword arguments forwarded to ``nn.Module``.
        """
        # A restored buffer invalidates the host mirror.
        self._host_sampling_step = None
        super()._load_from_state_dict(*args, **kwargs)

    def _pool(self, x):
        """Pool walk states over time.

        Parameters
        ----------
        x : torch.Tensor
            States of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape ``[walk, output_dim]``.
        """
        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "max":
            return x.amax(dim=1)
        return torch.cat((x.amax(dim=1), x.mean(dim=1)), dim=-1)

    def _encode(self, x, paths, ranks, relation):
        """Encode sampled walks with the sequence layers.

        Parameters
        ----------
        x : torch.Tensor
            Cell states of shape ``[cells, hidden]``.
        paths : torch.Tensor
            Walk cell indices of shape ``[walk, time]``.
        ranks : torch.Tensor
            Rank of each cell, shape ``[cells]``.
        relation : int
            Neighborhood group index selecting the encoder.

        Returns
        -------
        torch.Tensor
            Encoded walk states of shape ``[walk, time, hidden]``.
        """
        tokens = x[paths]
        if self.move is not None:
            path_ranks = ranks[paths]
            movement = torch.zeros_like(path_ranks)
            movement[:, 1:] = torch.sign(
                path_ranks[:, 1:] - path_ranks[:, :-1]
            )
            tokens = tokens + self.move(movement + 1)
        tokens = self.dropout(self.input_norm(tokens))
        encoder = self.encoders[0 if len(self.encoders) == 1 else relation]
        for layer in encoder:
            if self.checkpoint_layers and self.training:
                from torch.utils.checkpoint import checkpoint

                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        return self.output_norm(tokens)

    def forward(self, batch):
        """Sample and encode walks for every graph in the batch.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by ``TRAWLTransform``.

        Returns
        -------
        dict
            Contextual cell features ``x_r`` and ``batch_r``, cell, walk and
            graph embeddings with their batch indices, labels and readout
            metadata.
        """
        host = self._host(batch)
        counts = host["trawl_counts"].reshape(-1, self.max_rank + 1)
        edge_slices = getattr(batch, "_slice_dict", {}).get("trawl_edges")
        node_offsets = np.zeros(self.max_rank + 1, dtype=int)
        state_offset = 0
        (
            graph_embeddings,
            nodes,
            node_batches,
            walk_embeddings,
            walk_batches,
        ) = [], [], [], [], []
        all_cells, all_ranks, walk_views = [], [], []
        rank_cells = [[] for _ in range(self.max_rank + 1)]
        rank_batches = [[] for _ in range(self.max_rank + 1)]
        views = 1 if self.training else self.eval_views
        step = self.next_sampling_step() if self.training else 0
        if self.walk_refresh != "train":
            step = 0
        device = batch.trawl_pe.device

        def index(size, value):
            """Constant index vector.

            Parameters
            ----------
            size : int
                Length.
            value : int
                Fill value.

            Returns
            -------
            torch.Tensor
                Long tensor of ``size`` copies of ``value``.
            """
            return torch.full((size,), value, dtype=torch.long, device=device)

        for graph_id, sizes in enumerate(counts):
            if (
                int(host["trawl_relations"][graph_id])
                != self.num_neighborhoods
            ):
                raise ValueError("Transform/model neighborhood counts differ")
            signals = []
            for rank, size in enumerate(sizes):
                raw = batch[f"trawl_signal_{rank}"][
                    node_offsets[rank] : node_offsets[rank] + size
                ]
                signals.append(
                    self.features[rank](raw)
                    if size
                    else batch.trawl_pe.new_empty(0, self.hidden_dim)
                )
                node_offsets[rank] += size
            features = torch.cat(signals)
            total = int(sizes.sum())
            ranks = upload(
                np.repeat(np.arange(len(sizes)), sizes), features.device
            )
            x = features + self.position(
                batch.trawl_pe[state_offset : state_offset + total]
            )
            if self.rank is not None:
                x = x + self.rank(ranks)
            if self.color is not None:
                colors = batch.trawl_colors[
                    state_offset : state_offset + total
                ]
                host_colors = host["trawl_colors"][
                    state_offset : state_offset + total
                ]
                if (
                    host_colors.size
                    and int(host_colors.max()) >= self.color.num_embeddings
                ):
                    raise ValueError(
                        "num_colors is smaller than a transformed color ID"
                    )
                x = x + self.color(colors)
            if edge_slices is None:
                edges, weights = host["trawl_edges"], host["trawl_weights"]
            else:
                start, stop = (
                    int(edge_slices[graph_id]),
                    int(edge_slices[graph_id + 1]),
                )
                edges, weights = (
                    host["trawl_edges"][start:stop],
                    host["trawl_weights"][start:stop],
                )
            groups = (
                [None]
                if self.walk_scope == "union"
                else list(range(self.num_neighborhoods))
            )
            fallback = self._pool(features[None])[0]
            per_group, availability = [], []
            # Accumulate occurrences in full precision under autocast. Embedding
            # additions can promote encoded states independently of projections.
            accumulation_dtype = (
                torch.float32
                if features.dtype in {torch.float16, torch.bfloat16}
                else features.dtype
            )
            contextual = torch.zeros_like(features, dtype=accumulation_dtype)
            visits = contextual.new_zeros(total, 1)
            occurrence_values, occurrence_ids = [], []
            for group_id, relation in enumerate(groups):
                select = (
                    np.ones(len(edges), dtype=bool)
                    if relation is None
                    else edges[:, 2] == relation
                )
                chosen = edges[select]
                matrix = sp.csr_matrix(
                    (weights[select], (chosen[:, 0], chosen[:, 1])),
                    shape=(total, total),
                )
                pooled_views = []
                for view in range(views):
                    seed = walk_seed(
                        self.seed,
                        int(host["trawl_identity"][graph_id]),
                        step,
                        group_id,
                        view,
                    )
                    paths = self.walk_sampler(matrix, seed=seed, **self.walks)
                    if not len(paths):
                        continue
                    paths = upload(paths, x.device)
                    encoded = self._encode(x, paths, ranks, group_id)
                    pooled = self._pool(encoded)
                    pooled_views.append(pooled.mean(dim=0))
                    walk_embeddings.append(pooled)
                    walk_views.append(index(len(paths), view))
                    walk_batches.append(index(len(paths), graph_id))
                    contextual.index_add_(
                        0,
                        paths.flatten(),
                        encoded.flatten(0, 1).to(contextual),
                    )
                    if self.occurrence_attention is not None:
                        occurrence_values.append(encoded.flatten(0, 1))
                        occurrence_ids.append(paths.flatten())
                    visits.index_add_(
                        0,
                        paths.flatten(),
                        visits.new_ones(paths.numel(), 1),
                    )
                availability.append(bool(pooled_views))
                per_group.append(
                    torch.stack(pooled_views).mean(dim=0)
                    if pooled_views
                    else fallback * 0
                )
            if self.walk_scope == "union":
                graph_embeddings.append(
                    per_group[0] if availability[0] else fallback
                )
            else:
                graph_embeddings.append(
                    self.fusion(
                        torch.stack(per_group),
                        upload(np.array(availability), x.device).bool(),
                        fallback,
                    )
                )
            contextual = torch.where(
                visits > 0, contextual / visits.clamp_min(1), features
            )
            if occurrence_values:
                from torch_geometric.utils import softmax

                values, ids = (
                    torch.cat(occurrence_values),
                    torch.cat(occurrence_ids),
                )
                weights = softmax(
                    self.occurrence_attention(values), ids, num_nodes=total
                )
                attended = torch.zeros_like(contextual).index_add(
                    0, ids, (values * weights).to(contextual)
                )
                contextual = torch.where(visits > 0, attended, features)
            if self.graph_readout == "cells":
                graph_embeddings[-1] = self._pool(contextual[None])[0]
            rank_offset = 0
            for rank, size in enumerate(sizes):
                rank_cells[rank].append(
                    contextual[rank_offset : rank_offset + size]
                )
                rank_batches[rank].append(index(size, graph_id))
                rank_offset += size
            all_cells.append(contextual)
            all_ranks.append(ranks)
            nodes.append(contextual[: sizes[0]])
            node_batches.append(index(sizes[0], graph_id))
            state_offset += total
        result = {
            "x_0": torch.cat(nodes),
            "batch_0": torch.cat(node_batches),
            "cell_embeddings": torch.cat(all_cells),
            "cell_ranks": torch.cat(all_ranks),
            "graph_embedding": torch.stack(graph_embeddings),
            "labels": batch.get("y"),
            "walk_readout_compatible": self.walk_scope == "union"
            and self.graph_readout == "walks",
            "num_views": views,
        }
        result["walk_embedding"] = (
            torch.cat(walk_embeddings)
            if walk_embeddings
            else result["graph_embedding"].new_empty((0, self.output_dim))
        )
        result["walk_batch"] = (
            torch.cat(walk_batches)
            if walk_batches
            else result["batch_0"].new_empty(0)
        )
        result["walk_view"] = (
            torch.cat(walk_views)
            if walk_views
            else result["batch_0"].new_empty(0)
        )
        for rank in range(1, self.max_rank + 1):
            result[f"x_{rank}"] = torch.cat(rank_cells[rank])
            result[f"batch_{rank}"] = torch.cat(rank_batches[rank])
        return result
