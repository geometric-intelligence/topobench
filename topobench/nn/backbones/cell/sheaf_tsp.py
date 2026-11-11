"""SheafTSP backbone: an (n,d)-HilbNet for cell complexes.

Implements polynomial sheaf convolution with learnable restriction maps,
instantiating the (n,d)-HilbNet architecture from:

    Tandon et al. "Consistent Geometric Deep Learning via Hilbert Bundles
    and Cellular Sheaves" (2025), arXiv:2605.06395

Specifically, this module implements:
  - Def. 5 (Network Sheaf): finite-dimensional stalks with learned
    transport maps.
  - Eq. 8 (Sheaf Laplacian): Gaussian-kernel edge weights
    k_ij = exp(-d(x_i, x_j)^2 / 4t), diagonal blocks (sum_j k_ij) I_d,
    off-diagonal blocks -k_ij R_ij; symmetric degree normalization keeps
    the spectrum in [0, 2].
  - Eq. 10 (Polynomial (n,d)-HilbNet): per-order weight matrices,

        S^{l+1} = sigma( sum_{k=0}^{K} (L_F)^k S^l W_{l,k} )

    where the signal S lives on the stalks: features are lifted so each
    stalk coordinate carries distinct channel content (the stalk is the
    signal fiber, Sec. 6), not broadcast copies.
  - Eq. 15 (transport-alignment regularizer): sheaf Dirichlet energy of
    the lifted signal, exposed via ``SheafDirichletLoss`` so TopoBench's
    TBLoss adds lambda * E(s) to the task loss (lambda = 0.01 in the
    paper's experiments, Appendix F).
  - Appendix E: orthogonal transport classes. Tandon et al. use
    Householder products; this implementation projects via the Cayley
    transform (exponential map available as an option).

The Sheaf Laplacian L_F = delta^T K delta replaces the standard Hodge
Laplacian. This avoids oversmoothing because L_F penalizes deviations
from the *learned relational structure*, not raw differences (Theorem 1
in Tandon et al.: convergence of the Hilbert Sheaf Laplacian to the
Connection Laplacian).

When the optional C++ extensions (_tsp_core, _tsp_amg) are installed,
the model uses AMG-preconditioned LOBPCG for O(N log N)
eigendecomposition. Otherwise it falls back to pure PyTorch -- no native
dependency for CI/CD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional C++ accelerators — graceful fallback to pure PyTorch
try:
    import _tsp_amg  # noqa: F401
    import _tsp_core  # noqa: F401

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


# ═══════════════════════════════════════════════════════════════════════
# Sheaf Restriction Map Learner
# ═══════════════════════════════════════════════════════════════════════


class RestrictionMapLearner(nn.Module):
    """Learn restriction maps F_{v<e} : F_v → F_e for each edge.

    For stalk_dim d, each map is a d×d rotation obtained from a
    skew-symmetric generator — learned orthogonal maps on graphs in
    the spirit of BuNN [Bamberger et al., 2024, who use Householder
    reflections] and the orthogonal transport classes of Tandon et
    al., Appendix E; the Cayley projection follows the orthogonal-RNN
    lineage (Helfrich et al., ICML 2018). Two projections
    onto SO(d) are available. ``"cayley"``: R = (I-S)(I+S)^{-1}, whose
    image is the dense subset of SO(d) excluding rotations with a -1
    eigenvalue; reaching near-antipodal maps requires generator norms
    tan(theta/2), which diverge as theta approaches pi. ``"exp"``:
    R = exp(S), surjective onto SO(d) with bounded generators (at
    d = 2, exp(S) is the rotation by the generator's angle parameter),
    removing the antipodal obstruction relevant to heterophilic
    transport. The generator is antisymmetrized over the edge
    direction in both cases, which makes the transports
    orientation-equivariant (R_{vu} = R_{uv}^{-1}) and the model
    invariant to node relabeling.

    Parameters
    ----------
    in_channels : int
        Input feature dimension (used to condition the maps).
    stalk_dim : int
        Dimension of each stalk vector space.
    rotation_param : str
        Projection onto SO(d): ``"cayley"`` or ``"exp"``.
    """

    def __init__(
        self,
        in_channels: int,
        stalk_dim: int,
        rotation_param: str = "cayley",
    ):
        super().__init__()
        if rotation_param not in ("cayley", "exp"):
            raise ValueError("rotation_param must be 'cayley' or 'exp'")
        self.rotation_param = rotation_param
        self.stalk_dim = stalk_dim
        # Number of free parameters in a skew-symmetric d×d matrix
        n_skew = stalk_dim * (stalk_dim - 1) // 2
        # MLP: edge features → skew-symmetric parameters
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, 4 * stalk_dim),
            nn.ReLU(),
            nn.Linear(4 * stalk_dim, max(n_skew, 1)),
        )
        # Upper-triangle index pairs for vectorized skew assembly
        triu = torch.triu_indices(stalk_dim, stalk_dim, offset=1)
        self.register_buffer("_triu", triu, persistent=False)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute restriction maps for each edge.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_channels)
            Node features.
        edge_index : torch.Tensor, shape (2, E)
            Edge connectivity.

        Returns
        -------
        torch.Tensor, shape (E, stalk_dim, stalk_dim)
            Restriction maps as orthogonal matrices.
        """
        d = self.stalk_dim
        E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        # Antisymmetrized generator: params(u,v) = -params(v,u), so the
        # skew matrix flips sign under edge reversal and the Cayley map
        # yields R_{vu} = R_{uv}^{-1} by construction. Transports are
        # therefore orientation-equivariant and independent of node
        # labeling (the u < v canonicalization carries no information).
        feat_uv = torch.cat([x[src], x[dst]], dim=-1)
        feat_vu = torch.cat([x[dst], x[src]], dim=-1)
        params = self.mlp(feat_uv) - self.mlp(feat_vu)  # (E, n_skew)

        if d == 1:
            return torch.ones(E, 1, 1, device=x.device, dtype=x.dtype)

        # Build skew-symmetric matrices (vectorized over edges)
        S = x.new_zeros(E, d, d)
        S[:, self._triu[0], self._triu[1]] = params
        S[:, self._triu[1], self._triu[0]] = -params

        if self.rotation_param == "exp":
            # Exponential map: surjective onto SO(d); at d = 2 this is
            # the rotation by the generator's angle, so antipodal maps
            # are reachable with bounded generators.
            return torch.matrix_exp(S)
        # Cayley transform: R = (I + S)^{-1}(I - S)  →  orthogonal
        # (equals (I - S)(I + S)^{-1} since the factors commute)
        eye = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0)
        R = torch.linalg.solve(eye + S, eye - S)
        return R


# ═══════════════════════════════════════════════════════════════════════
# Sheaf Laplacian Builders (pure PyTorch, fully vectorized)
# ═══════════════════════════════════════════════════════════════════════


def _sheaf_laplacian_coo(
    num_nodes: int,
    edge_index: torch.Tensor,
    restriction_maps: torch.Tensor,
    stalk_dim: int,
    edge_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Assemble COO triplets of L_F = δ^T K δ (vectorized, no loops).

    For each edge e = (u, v) with weight k_e and restriction map R_e
    (paper Eq. 8):

        L[u,u] += k_e R_e^T R_e      L[u,v] += -k_e R_e^T
        L[v,v] += k_e I_d            L[v,u] += -k_e R_e

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_index : torch.Tensor, shape (2, E)
        Edge connectivity (deduplicated, no self-loops).
    restriction_maps : torch.Tensor, shape (E, d, d)
        Restriction maps for the source node of each edge.
    stalk_dim : int
        Stalk dimension d.
    edge_weights : torch.Tensor, shape (E,), optional
        Gaussian kernel weights k_e (default: all ones).

    Returns
    -------
    tuple
        (indices, values, Nd) COO triplets with duplicates unsummed.
    """
    d = stalk_dim
    Nd = num_nodes * d
    E = edge_index.shape[1]
    device = edge_index.device
    dtype = restriction_maps.dtype

    if E == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            Nd,
        )

    if edge_weights is None:
        edge_weights = torch.ones(E, device=device, dtype=dtype)

    src, dst = edge_index[0], edge_index[1]
    R = restriction_maps
    Rt = R.transpose(1, 2)
    RtR = torch.bmm(Rt, R)
    I_blk = (
        torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(E, d, d)
    )
    w = edge_weights.view(E, 1, 1)

    ar = torch.arange(d, device=device)
    ii = ar.view(1, d, 1).expand(E, d, d)  # local row index
    jj = ar.view(1, 1, d).expand(E, d, d)  # local col index
    u_r = (src.view(E, 1, 1) * d + ii).reshape(-1)
    u_c = (src.view(E, 1, 1) * d + jj).reshape(-1)
    v_r = (dst.view(E, 1, 1) * d + ii).reshape(-1)
    v_c = (dst.view(E, 1, 1) * d + jj).reshape(-1)

    rows = torch.cat([u_r, v_r, u_r, v_r])
    cols = torch.cat([u_c, v_c, v_c, u_c])
    vals = torch.cat(
        [
            (w * RtR).reshape(-1),
            (w * I_blk).reshape(-1),
            (-w * Rt).reshape(-1),
            (-w * R).reshape(-1),
        ]
    )
    return torch.stack([rows, cols]), vals, Nd


def build_sheaf_laplacian_torch(
    num_nodes: int,
    edge_index: torch.Tensor,
    restriction_maps: torch.Tensor,
    stalk_dim: int,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the Sheaf Laplacian L_F = δ^T K δ in dense form.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_index : torch.Tensor, shape (2, E)
        Edge connectivity.
    restriction_maps : torch.Tensor, shape (E, d, d)
        Restriction maps for source node of each edge.
    stalk_dim : int
        Stalk dimension d.
    edge_weights : torch.Tensor, shape (E,), optional
        Gaussian kernel weights k_e (paper Eq. 8; default: ones).

    Returns
    -------
    torch.Tensor, shape (N*d, N*d)
        Dense Sheaf Laplacian.
    """
    indices, values, Nd = _sheaf_laplacian_coo(
        num_nodes, edge_index, restriction_maps, stalk_dim, edge_weights
    )
    return torch.sparse_coo_tensor(indices, values, (Nd, Nd)).to_dense()


def build_sheaf_laplacian_sparse(
    num_nodes: int,
    edge_index: torch.Tensor,
    restriction_maps: torch.Tensor,
    stalk_dim: int,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the Sheaf Laplacian as a sparse COO tensor.

    Preferred for large graphs. Same semantics as the dense version.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_index : torch.Tensor, shape (2, E)
        Edge connectivity.
    restriction_maps : torch.Tensor, shape (E, d, d)
        Restriction maps for source node of each edge.
    stalk_dim : int
        Stalk dimension d.
    edge_weights : torch.Tensor, shape (E,), optional
        Gaussian kernel weights k_e (paper Eq. 8; default: ones).

    Returns
    -------
    torch.Tensor
        Sparse COO Sheaf Laplacian of shape (N*d, N*d).
    """
    indices, values, Nd = _sheaf_laplacian_coo(
        num_nodes, edge_index, restriction_maps, stalk_dim, edge_weights
    )
    return torch.sparse_coo_tensor(indices, values, (Nd, Nd)).coalesce()


# ═══════════════════════════════════════════════════════════════════════
# Single Sheaf Convolution Layer
# ═══════════════════════════════════════════════════════════════════════


class SheafConvLayer(nn.Module):
    """One layer of spectral sheaf convolution (paper Eq. 10).

    Applies a K-order polynomial filter with per-order weight matrices
    on the degree-normalized Sheaf Laplacian:

        y = Σ_{k=0}^{K} L̂_F^k · s · W_k,
        L̂_F = D^{-1/2} L_F D^{-1/2}

    where s is the input lifted into stalk space so that each stalk
    coordinate carries distinct channel content (the stalk is the
    signal fiber, paper Sec. 6). Edges are weighted by the Gaussian
    kernel k_ij = exp(-||x_i - x_j||² / 4t) of Eq. 8 with a learnable
    bandwidth t. The layer also records the sheaf Dirichlet energy
    E(s) = sᵀ L̂_F s of the lifted signal in ``last_dirichlet`` — the
    transport-alignment regularizer of Eq. 15 (gradients flow only to
    the transports and kernel, not the signal).

    Parameters
    ----------
    in_channels : int
        Input feature channels per node.
    out_channels : int
        Output feature channels per node.
    stalk_dim : int
        Stalk dimension d.
    filter_order : int
        Polynomial filter order K (powers 0..K are used).
    dropout : float
        Dropout rate on the signal path (input of the stalk lift).
    mlp_dropout : float
        Dropout rate on the input of the restriction-map learner —
        regularizes the learned transports without corrupting the
        filtered signal (surgical regularization).
    filter_basis : str
        Polynomial basis for the spectral filter: ``"monomial"``
        (default; raw powers L̂^k as in Eq. 10) or ``"chebyshev"``
        (Chebyshev T_k(L̂ - I) via the three-term recurrence). The
        degree-normalized spectrum lies in [0, 2], so L̂ - I maps it
        to [-1, 1] where the Chebyshev basis is orthogonal — a
        better-conditioned realization of the paper's Spectral Sheaf
        Filtering (frequency-aware filtering in the sheaf eigenbasis)
        with identical cost and parameter count.
    kernel_distance : str
        ``"feature"`` (raw ‖x_i − x_j‖²) or ``"transport"``
        (‖s_i − R_e s_j‖² under the learned maps) inside the
        Gaussian edge kernel.
    ppr_alpha : float
        Teleport probability for the ``"ppr"`` basis initialization.
    reg_form : str
        Transport regularizer form: ``"dirichlet"`` (Eq. 15 energy
        $\\mathrm{tr}(s^\top L_{\\mathcal F} s)$) or ``"alignment"``
        (negative mean kernel alignment
        $-\\overline{\\exp(-\\lVert s_i - R_e s_j\rVert^2/4t)}$,
        Tandon et al. App. D — bounded per edge).
    rotation_param : str
        Projection onto SO(d) for the restriction maps: ``"cayley"``
        or ``"exp"`` (surjective exponential map).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 2,
        filter_order: int = 3,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        filter_basis: str = "monomial",
        kernel_distance: str = "feature",
        ppr_alpha: float = 0.1,
        reg_form: str = "dirichlet",
        rotation_param: str = "cayley",
    ):
        super().__init__()
        if filter_order < 1:
            raise ValueError("filter_order must be >= 1")
        if filter_basis not in ("monomial", "chebyshev", "ppr"):
            raise ValueError(
                "filter_basis must be 'monomial', 'chebyshev' or 'ppr'"
            )
        if kernel_distance not in ("feature", "transport"):
            raise ValueError(
                "kernel_distance must be 'feature' or 'transport'"
            )
        if reg_form not in ("dirichlet", "alignment"):
            raise ValueError("reg_form must be 'dirichlet' or 'alignment'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stalk_dim = stalk_dim
        self.filter_order = filter_order
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout
        self.filter_basis = filter_basis
        self.kernel_distance = kernel_distance
        self.reg_form = reg_form

        # Learnable restriction maps
        self.map_learner = RestrictionMapLearner(
            in_channels, stalk_dim, rotation_param=rotation_param
        )

        # Lift channels into stalk fibers: (N, C) → (N*d, C).
        # Each stalk coordinate receives its own linear view of the
        # features, so transports act on non-degenerate stalk vectors.
        self.lift = nn.Linear(in_channels, stalk_dim * out_channels)

        # Per-order filter weight matrices W_k (paper Eq. 10).
        # Init: W_0 = I, W_{k>0} = 0 → layer starts near-identity.
        # In "ppr" mode the per-order matrices are replaced by K+1
        # scalars on the sheaf lazy-walk operator P = I - L̂/2
        # (spectrum in [0, 1]), initialized to the personalized-
        # PageRank profile w_k = α(1-α)^k — decoupled long-range
        # low-pass diffusion (HiGCN/APPNP-style) that concentrates
        # on the bottom eigensections of the sheaf Laplacian.
        if filter_basis == "ppr":
            a = ppr_alpha
            w = [a * (1.0 - a) ** k for k in range(filter_order)]
            w.append((1.0 - a) ** filter_order)
            self.filter_weights = nn.Parameter(torch.tensor(w))
        else:
            W = torch.zeros(filter_order + 1, out_channels, out_channels)
            W[0] = torch.eye(out_channels)
            self.filter_weights = nn.Parameter(W)

        # Learned pooling back to node space: (N, d*C) → (N, C)
        self.pool = nn.Linear(stalk_dim * out_channels, out_channels)

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.norm = nn.LayerNorm(out_channels)

        # Gaussian kernel bandwidth t (Eq. 8), softplus(0.5413) ≈ 1
        self.log_bandwidth = nn.Parameter(torch.tensor(0.5413))

        # Sheaf Dirichlet energy of the last forward pass (Eq. 15)
        self.last_dirichlet: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_channels)
            Node features.
        edge_index : torch.Tensor, shape (2, E)
            Edge connectivity (deduplicated, no self-loops).

        Returns
        -------
        torch.Tensor, shape (N, out_channels)
            Filtered node features.
        """
        N = x.shape[0]
        d = self.stalk_dim
        C = self.out_channels
        src, dst = edge_index[0], edge_index[1]

        # Learn restriction maps; MLP-input dropout regularizes the
        # learned transports without touching the signal path.
        R = self.map_learner(
            F.dropout(x, p=self.mlp_dropout, training=self.training),
            edge_index,
        )  # (E, d, d)

        # Lift channels into stalk fibers: (N, C) → (N*d, C)
        s = self.lift(
            F.dropout(x, p=self.dropout, training=self.training)
        ).view(N * d, C)

        # Gaussian kernel edge weights k_ij (paper Eq. 8); the squared
        # distance is a per-channel mean so the scale is C-independent.
        t = F.softplus(self.log_bandwidth) + 1e-6
        if self.kernel_distance == "transport":
            # Transport-aware distance ‖s_u − R_e s_v‖²: weights edges
            # by inconsistency under the learned map instead of raw
            # feature similarity, so transport-consistent (not merely
            # similar) neighbors couple strongly — the sheaf-native
            # form of Eq. 8 (smoothness w.r.t. learned structure).
            s3 = s.view(N, d, C)
            d2 = (s3[src] - torch.bmm(R, s3[dst])).pow(2).mean(dim=(1, 2))
        else:
            d2 = (x[src] - x[dst]).pow(2).mean(dim=-1)
        k = torch.exp(-d2 / (4.0 * t))

        # Build (kernel-weighted) Sheaf Laplacian
        use_sparse = N * d > 2000
        if use_sparse:
            L = build_sheaf_laplacian_sparse(
                N, edge_index, R, d, edge_weights=k
            )
        else:
            L = build_sheaf_laplacian_torch(
                N, edge_index, R, d, edge_weights=k
            )

        # Symmetric degree normalization: L̂ = D^{-1/2} L D^{-1/2}.
        # With orthogonal R the diagonal blocks are (Σ_e k_e) I_d, so
        # D is scalar per node and the spectrum of L̂ lies in [0, 2] —
        # keeps powers L̂^k stable for the polynomial filter.
        deg = x.new_zeros(N)
        deg = deg.index_add_(0, src, k).index_add_(0, dst, k)
        dinv = torch.where(
            deg > 0,
            deg.clamp(min=1e-12).rsqrt(),
            x.new_zeros(N),
        )
        dvec = dinv.repeat_interleave(d)  # (N*d,)
        if use_sparse:
            idx = L.indices()
            val = L.values() * dvec[idx[0]] * dvec[idx[1]]
            L = torch.sparse_coo_tensor(idx, val, L.shape).coalesce()
        else:
            L = L * (dvec.unsqueeze(1) * dvec.unsqueeze(0))

        # Transport regularizer (Eq. 15). Signal detached in both
        # forms so gradients flow only into the transports R and the
        # kernel bandwidth t.
        s_d = s.detach()
        if self.reg_form == "alignment":
            # Kernel-alignment form (Tandon et al., App. D): reward
            # transports that align neighbors. Bounded in (0, 1] per
            # edge, so genuinely dissimilar pairs (e.g. cross-community
            # edges) saturate instead of dominating the gradient the
            # way a quadratic Dirichlet penalty lets them.
            s3d = s_d.view(N, d, C)
            diff2 = (s3d[src] - torch.bmm(R, s3d[dst])).pow(2).mean(dim=(1, 2))
            # Bandwidth detached inside the regularizer: with learnable
            # t the reward saturates as t grows, independent of the
            # transports, so alignment gradients must flow into R only.
            # The forward kernel still trains t via the task loss.
            align = torch.exp(-diff2 / (4.0 * t.detach()))
            self.last_dirichlet = (
                -align.mean() if align.numel() > 0 else s_d.new_zeros(())
            )
        else:
            Ls_d = torch.sparse.mm(L, s_d) if use_sparse else L @ s_d
            self.last_dirichlet = (s_d * Ls_d).sum() / max(s_d.numel(), 1)

        # Polynomial filter with per-order weights (Eq. 10)
        def Lmm(v: torch.Tensor) -> torch.Tensor:
            """Apply the assembled Laplacian to a stalk signal.

            Parameters
            ----------
            v : torch.Tensor
                Stalk signal of shape (N*d, C).

            Returns
            -------
            torch.Tensor
                The product L v, same shape as ``v``.
            """
            return torch.sparse.mm(L, v) if use_sparse else L @ v

        if self.filter_basis == "ppr":
            # Decoupled scalar filter on the sheaf lazy walk
            # P = I - L̂/2 (spectrum ⊂ [0, 1]): y = Σ_k w_k P^k s.
            # Stable at large K; PPR init makes it personalized-
            # PageRank diffusion over the sheaf at epoch 0.
            y = self.filter_weights[0] * s
            v = s
            for kk in range(1, self.filter_order + 1):
                v = v - 0.5 * Lmm(v)
                y = y + self.filter_weights[kk] * v
            y = self.pool(y.view(N, d * C))
            return self.norm(y + self.bias)

        y = s @ self.filter_weights[0]
        if self.filter_basis == "chebyshev":
            # Chebyshev basis on L̃ = L̂ - I (spectrum in [-1, 1]):
            # T_0 = s, T_1 = L̃ s, T_{k+1} = 2 L̃ T_k - T_{k-1}.
            # Numerically better-conditioned than raw monomials
            # (paper: SSF frequency-aware filtering).
            T_prev = s
            T_cur = Lmm(s) - s
            y = y + T_cur @ self.filter_weights[1]
            for kk in range(2, self.filter_order + 1):
                T_next = 2.0 * (Lmm(T_cur) - T_cur) - T_prev
                y = y + T_next @ self.filter_weights[kk]
                T_prev, T_cur = T_cur, T_next
        else:
            Ls = s
            for kk in range(1, self.filter_order + 1):
                Ls = Lmm(Ls)
                y = y + Ls @ self.filter_weights[kk]

        # Learned pooling back to node space: (N*d, C) → (N, C)
        y = self.pool(y.view(N, d * C))

        y = self.norm(y + self.bias)
        return y


# ═══════════════════════════════════════════════════════════════════════
# Full SheafTSP Backbone
# ═══════════════════════════════════════════════════════════════════════


class SheafTSP(nn.Module):
    r"""Sheaf-Topological Signal Processing backbone for cell complexes.

    A multi-layer spectral sheaf convolutional network that operates on
    the 1-cells (edges) of a cell complex using learnable Sheaf
    Laplacians.

    Each layer:
      1. Learns restriction maps R_{v<e} via a conditioned Cayley MLP.
      2. Builds the kernel-weighted Sheaf Laplacian L_F = δ^T K δ and
         degree-normalizes it (paper Eq. 8).
      3. Lifts channels into stalk fibers and applies a K-order
         polynomial filter with per-order weight matrices (Eq. 10).

    The stacked architecture provides a topologically-aware message
    passing scheme that avoids oversmoothing by penalizing deviations
    from *learned* relational structure rather than raw node
    differences. The summed per-layer sheaf Dirichlet energies are
    exposed as ``dirichlet_energy`` for the Eq. 15 regularizer (see
    ``SheafDirichletLoss``).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    n_layers : int, optional
        Number of sheaf convolution layers (default: 2).
    stalk_dim : int, optional
        Dimension of stalk vector spaces (default: 2).
    filter_order : int, optional
        Polynomial filter order K per layer; powers 0..K (default: 3).
    dropout : float, optional
        Dropout rate on the signal path (default: 0.0).
    mlp_dropout : float, optional
        Dropout rate on the restriction-map learner input — surgical
        regularization of the learned transports that leaves the
        filtered signal clean (default: 0.0).
    filter_basis : str, optional
        Basis for the spectral filter in each layer: ``"monomial"``
        (raw powers of L̂, Eq. 10), ``"chebyshev"`` (Chebyshev
        polynomials T_k(L̂ − I), a better-conditioned realization of
        the paper's Spectral Sheaf Filtering with identical cost and
        parameter count) or ``"ppr"`` (decoupled scalar coefficients
        on the sheaf lazy walk P = I − L̂/2, PPR-initialized —
        long-range low-pass diffusion concentrating on the bottom
        eigensections, stable at large ``filter_order``).
        Default: ``"monomial"``.
    kernel_distance : str, optional
        Distance inside the Gaussian edge kernel: ``"feature"``
        (raw ‖x_i − x_j‖², Eq. 8) or ``"transport"`` (inconsistency
        under the learned map ‖s_i − R_e s_j‖² on lifted stalk
        signals — weights edges by transport-consistency instead of
        feature similarity). Default: ``"feature"``.
    ppr_alpha : float, optional
        Teleport probability initializing the ``"ppr"`` filter
        scalars w_k = α(1−α)^k (default: 0.1).
    reg_form : str, optional
        Transport regularizer form: ``"dirichlet"`` (Eq. 15) or
        ``"alignment"`` (bounded kernel-alignment reward, Tandon
        et al. App. D). Default: ``"dirichlet"``.
    rotation_param : str, optional
        Projection onto SO(d) for the restriction maps: ``"cayley"``
        or ``"exp"`` (surjective exponential map). Default:
        ``"cayley"``.
    global_context : bool, optional
        Add a zero-initialized per-layer global mean-context term
        ``x += W_g mean(x)`` (default: False).
    last_act : bool, optional
        Whether to apply activation after the last layer
        (default: False).
    **kwargs : dict, optional
        Additional keyword arguments (absorbs Hydra-injected nodes
        such as the nested ``loss`` regularizer config).

    References
    ----------
    .. [1] Tandon et al. "Consistent Geometric Deep Learning via
       Hilbert Bundles and Cellular Sheaves" (2025), arXiv:2605.06395
       Def. 5 (Network Sheaf), Eq. 8 (Sheaf Laplacian), Eq. 10
       (Polynomial (n,d)-HilbNet), Eq. 15 (transport regularizer),
       Theorem 1 (convergence to Connection Laplacian)
    .. [2] Bodnar et al. "Neural Sheaf Diffusion" (2022),
       arXiv:2202.04579, Sec. 3: Sheaf Diffusion as a generalization
       of graph diffusion
    .. [3] Bamberger et al. "Bundle Neural Networks for message
       diffusion on graphs" (2024), arXiv:2405.15540: learned
       orthogonal bundle maps on graphs (Householder reflections;
       direct SO(2) parameterization at d = 2)
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        stalk_dim: int = 2,
        filter_order: int = 3,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        filter_basis: str = "monomial",
        kernel_distance: str = "feature",
        ppr_alpha: float = 0.1,
        reg_form: str = "dirichlet",
        rotation_param: str = "cayley",
        global_context: bool = False,
        last_act: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # maintain dimension for wrapper
        self.n_layers = n_layers
        self.stalk_dim = stalk_dim
        self.dropout = dropout
        self.filter_basis = filter_basis
        self.last_act = last_act

        # Summed sheaf Dirichlet energy of the last forward (Eq. 15)
        self.dirichlet_energy: torch.Tensor | None = None

        # Optional per-layer global mean-context: x += W_g mean(x).
        # Zero-init keeps epoch-0 behavior identical; gives every cell
        # a whole-complex summary at each layer.
        self.global_context = global_context
        if global_context:
            self.gctx = nn.ModuleList()
            for _ in range(n_layers):
                lin = nn.Linear(in_channels, in_channels, bias=False)
                nn.init.zeros_(lin.weight)
                self.gctx.append(lin)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SheafConvLayer(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    stalk_dim=stalk_dim,
                    filter_order=filter_order,
                    dropout=dropout,
                    mlp_dropout=mlp_dropout,
                    filter_basis=filter_basis,
                    kernel_distance=kernel_distance,
                    ppr_alpha=ppr_alpha,
                    reg_form=reg_form,
                    rotation_param=rotation_param,
                )
            )

    def forward(
        self, x: torch.Tensor, Ld: torch.Tensor, Lu: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass on cell complex 1-skeleton.

        Accepts the same interface as CCCN for compatibility with
        TopoBench wrappers. Ld (down_laplacian_1) and Lu
        (up_laplacian_1) are used to derive the edge connectivity for
        restriction map learning.

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_channels)
            Input 1-cell (edge) features.
        Ld : torch.Tensor
            Down Laplacian of rank 1 (sparse).
        Lu : torch.Tensor
            Up Laplacian of rank 1 (sparse).

        Returns
        -------
        torch.Tensor, shape (N, in_channels)
            Output 1-cell features.
        """
        # Extract edge connectivity from the Hodge Laplacian
        # L_1 = Ld + Lu; nonzero pattern gives adjacency of 1-cells.
        # Keep the canonical orientation u < v only: this strips the
        # diagonal (self-loops from Laplacian degree entries) and
        # dedupes the symmetric (u,v)/(v,u) copies so each undirected
        # pair contributes exactly one coherent restriction map.
        L1 = (Ld + Lu).coalesce()
        idx = L1.indices()
        edge_index = idx[:, idx[0] < idx[1]]  # (2, E)

        reg = x.new_zeros(())
        for i, layer in enumerate(self.layers):
            x_new = layer(x, edge_index)
            if layer.last_dirichlet is not None:
                reg = reg + layer.last_dirichlet
            # Residual connection
            x = x + x_new
            if self.global_context:
                x = x + self.gctx[i](x.mean(dim=0, keepdim=True))
            if i < self.n_layers - 1 or self.last_act:
                x = F.relu(x)

        self.dirichlet_energy = reg
        return x


# ═══════════════════════════════════════════════════════════════════════
# Transport-Alignment Regularizer (paper Eq. 15)
# ═══════════════════════════════════════════════════════════════════════


class SheafDirichletLoss:
    """Transport-alignment regularizer of Tandon et al., Eq. 15.

    Adds lambda * E(s) to the task loss, where E(s) is the summed
    per-layer sheaf Dirichlet energy of the lifted signal under the
    learned transports and kernel weights. Plugs into TopoBench's
    ``TBLoss`` via the ``model.backbone.loss`` Hydra node; the wrapper
    exposes the energy as ``model_out["sheaf_dirichlet"]`` during
    training only (validation/test losses stay pure task losses).

    Parameters
    ----------
    loss_weight : float, optional
        Regularization weight lambda (default: 0.01, per Appendix F).
    """

    def __init__(self, loss_weight: float = 0.01):
        self.loss_weight = loss_weight

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loss_weight={self.loss_weight})"

    def __call__(self, model_out: dict, batch) -> torch.Tensor:
        """Compute the weighted regularizer term.

        Parameters
        ----------
        model_out : dict
            Model output dictionary; reads ``sheaf_dirichlet`` if set.
        batch : torch_geometric.data.Data
            Batch object (used only for device placement).

        Returns
        -------
        torch.Tensor
            Scalar regularization loss (0 when not training).
        """
        reg = model_out.get("sheaf_dirichlet")
        if reg is None:
            return torch.zeros((), device=batch.y.device)
        return self.loss_weight * reg
