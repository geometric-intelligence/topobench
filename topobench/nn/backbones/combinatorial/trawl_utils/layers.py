"""Sequence layers and neighborhood fusion used by the TRAWL backbone.

Every layer maps walk states of shape ``[walk, time, hidden]`` to the same
shape: a pure-PyTorch Mamba block, SISA (state-space augmented attention),
GRU, transformer, MLP and graph layers applied to walks as path graphs.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


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


def attention_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype in which attention kernels will process ``x``.

    Under autocast, ``scaled_dot_product_attention`` casts its inputs to the
    autocast dtype; otherwise it keeps the dtype of ``x``.

    Parameters
    ----------
    x : torch.Tensor
        Attention input.

    Returns
    -------
    torch.dtype
        Effective attention dtype.
    """
    device_type = x.device.type
    if hasattr(torch, "get_autocast_dtype"):  # torch >= 2.4
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    elif device_type == "cuda" and torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    elif device_type == "cpu" and torch.is_autocast_cpu_enabled():
        return torch.get_autocast_cpu_dtype()
    return x.dtype


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
    # fp16 overflows above 65504 (about exp(11)), so float16 attention needs
    # a much smaller span before heads switch to the explicit path.
    MAX_FACTORED_SPAN_FP16 = 8.0

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
        limit = (
            self.MAX_FACTORED_SPAN_FP16
            if attention_dtype(q) == torch.float16
            else self.MAX_FACTORED_SPAN
        )
        wide = span.detach() > limit
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
