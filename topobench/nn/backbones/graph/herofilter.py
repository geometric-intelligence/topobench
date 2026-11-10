"""HeroFilter backbone.

Implementation of *HeroFilter: Adaptive Spectral Graph Filter for Varying
Heterophilic Relations* (NeurIPS 2025, arXiv:2510.10864), reference code
`zshuai8/HeroFilter` @ f0749d6.

The model builds, for every node, a *patch* of the p most spectrally relevant
nodes (Eq. 8-9 / Eq. 15), mixes the patches with an MLP-Mixer (Eq. 10-11) and
aggregates them into node representations (Section 4.2). Two patchers are
provided:

- :class:`FastPatcher` (default) - Fast-HeroFilter, Eq. 12-15. Relevance is a
  truncated Neumann series of the personalized-PageRank closed form; only two
  hyperparameters (``c``, ``K``), hence usable inductively on graphs of any
  size.
- :class:`SpectralPatcher` - Eq. 6-9. Relevance is ``R = U g(Lambda) U^T``
  built from an eigendecomposition of the normalized adjacency and the
  learnable filter of :class:`AdaptivePolynomialFilter`.

Known divergences from the released reference code, stated here once:

- The reference eigendecomposes the row-normalized adjacency ``D^-1 A`` with
  ``np.linalg.eig`` (non-symmetric, so ``U U^T != I``). We use the symmetric
  normalization ``D^-1/2 A D^-1/2`` with ``torch.linalg.eigh``, which is
  similar to ``D^-1 A`` (identical eigenvalues) and makes Eq. 7 exact.
- In the released code the spectral filter bank is a plain tensor - it is not
  registered as a Parameter and is not passed to the optimizer. The
  ``"reference"`` filter mode reproduces that behaviour verbatim (frozen
  weights, ``w_k^k * lambda^k``, identity activation); the ``"binned"`` mode
  implements the paper's learnable ``w_k`` (Eq. 6) in a size-agnostic way.
- The reference weights each patch slot by a softmax of a fixed random,
  non-trainable vector before summing. Section 4.2 and the Appendix
  pseudocode specify a plain aggregation ("mean, sum, or flatten" /
  "summation"), which is what ``aggregation`` implements.
- The patching rule of Eq. 8-9 selects *indices*, which is
  non-differentiable: as written (and as released - the reference softmax
  patch weighting is commented out), the classification loss provides no
  gradient to the filter weights :math:`w_k` of Eq. 6. ``weight_patches``
  multiplies each patch by the softmax of its selected relevance scores,
  restoring a gradient path so the Eq. 6 filter is genuinely learnable. It
  defaults to on for the ``"spectral"`` patcher and off for ``"fast"`` /
  ``"reference"``.
"""

import torch
import torch.nn.functional as F
from torch import nn

_SIGMAS = {
    "identity": lambda t: t,
    "relu": F.relu,
    "tanh": torch.tanh,
}


class AdaptivePolynomialFilter(nn.Module):
    r"""Adaptive polynomial spectral filter, Eq. 6 of arXiv:2510.10864.

    Implements :math:`g(\Lambda) = \sum_{k=1}^{K} \sigma(w_k \odot \Lambda^k)`
    where :math:`w_k` are learnable frequency-specific weights.

    The paper defines :math:`w_k \in \mathbb{R}^n` (one weight per frequency
    of one fixed graph), which is transductive. For inductive use, ``mode
    ="binned"`` parameterizes :math:`w_k` on ``num_bins`` fixed frequency bins
    spanning the eigenvalue range ``[-1, 1]`` of the symmetric normalized
    adjacency and linearly interpolates onto each graph's actual spectrum.
    This follows the paper's own Fig. 2, which analyses learned filters over
    fixed eigenvalue segments (0-0.4, ..., 1.6-2.0 of the Laplacian spectrum).

    ``mode="reference"`` instead reproduces the released reference code
    (`zshuai8/HeroFilter` @ f0749d6, ``src/HeroFilter.py::Patcher``):
    :math:`g(\Lambda) = \sum_k w_k^k \odot \Lambda^k` with identity
    activation, where ``w`` is a uniform random tensor that is *not*
    registered as a Parameter and receives no gradient updates. Weights are
    drawn deterministically per graph size so repeated forward passes agree.

    Parameters
    ----------
    order : int
        Polynomial order :math:`K` (Eq. 6).
    num_bins : int, optional
        Number of frequency bins for ``mode="binned"`` (default: 16).
    sigma : str, optional
        Activation applied to each term. One of ``"identity"``, ``"relu"``,
        ``"tanh"`` (default: ``"identity"``). The released code applies no
        activation; identity also satisfies the :math:`\sigma(0)=0` condition
        of Prop. 2.
    mode : str, optional
        ``"binned"`` (paper's Eq. 6, inductive) or ``"reference"`` (released
        code, frozen) (default: ``"binned"``).
    seed : int, optional
        Seed for the frozen ``"reference"`` weights (default: 0).
    """

    def __init__(
        self, order, num_bins=16, sigma="identity", mode="binned", seed=0
    ):
        super().__init__()
        if sigma not in _SIGMAS:
            raise ValueError(f"Unknown sigma '{sigma}'")
        if mode not in ("binned", "reference"):
            raise ValueError(f"Unknown filter mode '{mode}'")
        self.order = order
        self.num_bins = num_bins
        self.sigma = sigma
        self.mode = mode
        self.seed = seed
        if mode == "binned":
            # Uniform init in [0, 1), matching the reference filter bank.
            self.weight = nn.Parameter(torch.rand(order, num_bins))
        else:
            self.weight = None
            self._frozen_weights: dict[int, torch.Tensor] = {}

    def _reference_weights(self, n, device):
        """Frozen uniform random weights of the reference code, per graph size.

        Parameters
        ----------
        n : int
            Number of eigenvalues (graph size).
        device : torch.device
            Device to place the weights on.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(order, n)``, deterministic in ``(seed, n)``.
        """
        if n not in self._frozen_weights:
            generator = torch.Generator().manual_seed(self.seed + n)
            self._frozen_weights[n] = torch.rand(
                self.order, n, generator=generator
            )
        return self._frozen_weights[n].to(device)

    def interpolate_weights(self, eigvals):
        r"""Interpolate the binned weights onto a graph's eigenvalues.

        Bin centers are spread uniformly over ``[-1, 1]`` (the spectrum of
        the symmetric normalized adjacency); each eigenvalue receives the
        linear interpolation of the two enclosing bins.

        Parameters
        ----------
        eigvals : torch.Tensor
            Eigenvalues :math:`\Lambda` of shape ``(n,)``.

        Returns
        -------
        torch.Tensor
            Weights :math:`w_k(\lambda)` of shape ``(order, n)``.
        """
        pos = (eigvals.clamp(-1.0, 1.0) + 1.0) / 2.0 * (self.num_bins - 1)
        lo = pos.floor().long().clamp(max=self.num_bins - 1)
        hi = (lo + 1).clamp(max=self.num_bins - 1)
        frac = pos - lo.to(pos.dtype)
        return self.weight[:, lo] * (1.0 - frac) + self.weight[:, hi] * frac

    def forward(self, eigvals):
        r"""Evaluate the filter response :math:`g(\Lambda)` (Eq. 6).

        Parameters
        ----------
        eigvals : torch.Tensor
            Eigenvalues of the normalized adjacency, shape ``(n,)``.

        Returns
        -------
        torch.Tensor
            Filter response of shape ``(n,)``.
        """
        sigma = _SIGMAS[self.sigma]
        if self.mode == "binned":
            weights = self.interpolate_weights(eigvals)
        else:
            weights = self._reference_weights(eigvals.shape[0], eigvals.device)
        response = eigvals.new_zeros(eigvals.shape)
        for k in range(1, self.order + 1):
            w_k = weights[k - 1]
            if self.mode == "reference":
                # Released code raises the weight itself to the power k:
                # `filtered += torch.mul(p**(i+1), eigV**(i+1))`.
                w_k = w_k**k
            response = response + sigma(w_k * eigvals**k)
        return response


class SpectralPatcher(nn.Module):
    r"""Spectral patcher, Eq. 6-9 of arXiv:2510.10864.

    Builds the relevance matrix :math:`R = U g(\Lambda) U^\top` (Eq. 7) from
    an eigendecomposition of the normalized adjacency and the filter of
    :class:`AdaptivePolynomialFilter`, then selects for each node ``v`` the
    top-``p-1`` entries of column ``v`` of ``R`` and prepends ``v`` itself
    (Eq. 8), yielding patch indices for the patch tensor
    :math:`P \in \mathbb{R}^{n \times p \times d}` (Eq. 9).

    Self-inclusion follows the reference code exactly: the node is prepended
    and *not* masked out of the top-k, so it may appear twice in its own
    patch. When a graph has fewer than ``p`` nodes, remaining slots are
    filled with the node itself - the reference's own padding rule
    (``train.py:622`` pads with ``neighbor[0]``).

    Parameters
    ----------
    order : int
        Polynomial order :math:`K` of the filter (Eq. 6).
    num_bins : int, optional
        Frequency bins of the ``"binned"`` filter (default: 16).
    sigma : str, optional
        Filter activation (default: ``"identity"``).
    filter_mode : str, optional
        ``"binned"`` or ``"reference"``, see
        :class:`AdaptivePolynomialFilter` (default: ``"binned"``).
    seed : int, optional
        Seed for ``"reference"`` weights (default: 0).
    """

    def __init__(
        self,
        order,
        num_bins=16,
        sigma="identity",
        filter_mode="binned",
        seed=0,
    ):
        super().__init__()
        self.filter = AdaptivePolynomialFilter(
            order, num_bins=num_bins, sigma=sigma, mode=filter_mode, seed=seed
        )

    def relevance(self, eigvals, eigvecs):
        r"""Relevance matrix :math:`R = U\,\mathrm{diag}(g(\Lambda))\,U^\top` (Eq. 7).

        Parameters
        ----------
        eigvals : torch.Tensor
            Eigenvalues, shape ``(n,)``.
        eigvecs : torch.Tensor
            Eigenvectors ``U`` (columns), shape ``(n, n)``.

        Returns
        -------
        torch.Tensor
            Relevance matrix of shape ``(n, n)``.
        """
        return (eigvecs * self.filter(eigvals)) @ eigvecs.T

    def forward(self, adj_norm, patch_size):
        r"""Patch indices via ``top-p_col(R)`` (Eq. 8).

        Parameters
        ----------
        adj_norm : torch.Tensor
            Dense symmetric normalized adjacency of one graph, ``(n, n)``.
        patch_size : int
            Patch size ``p``.

        Returns
        -------
        tuple of torch.Tensor
            Patch indices of shape ``(n, p)`` (row ``v`` starts with ``v``)
            and their relevance scores of shape ``(n, p)``.
        """
        eigvals, eigvecs = torch.linalg.eigh(adj_norm)
        relevance = self.relevance(eigvals, eigvecs)
        return _top_p_col(relevance, patch_size, mask_self=False)


class FastPatcher(nn.Module):
    r"""Fast-HeroFilter patcher, Eq. 12-15 of arXiv:2510.10864.

    The personalized-PageRank objective (Eq. 12) has the closed-form solution
    :math:`r_v = (1-c)(I - c\tilde A)^{-1} e_v` (Eq. 13), approximated by the
    truncated Neumann series
    :math:`r_v \approx (1-c)\sum_{k=0}^{K} c^k \tilde A^k e_v` (Eq. 14).
    Stacking all :math:`r_v` columnwise gives the relevance matrix, and
    patches are :math:`\phi_{fast}(\tilde A) = \mathrm{top\text{-}p_{col}}`
    of it (Eq. 15).

    The reference computes these scores offline with power-iteration
    PageRank (``arxiv_year_ppr.py``, alpha=0.5); here the truncated series is
    evaluated in-model per graph, which is exact w.r.t. Eq. 14 and needs no
    preprocessing. There are no learnable parameters - only ``c`` and ``K``
    (Section 4.4) - so the patcher is inductive by construction. The node
    itself is guaranteed first in its own patch (the reference's PPR ranking
    places the personalization node first).

    Parameters
    ----------
    c : float, optional
        Teleport/attenuation factor of Eq. 12-14, in ``(0, 1)``
        (default: 0.5, the reference's alpha).
    order : int, optional
        Truncation order ``K`` of the Neumann series (default: 10).
    """

    def __init__(self, c=0.5, order=10):
        super().__init__()
        if not 0.0 < c < 1.0:
            raise ValueError(f"c must be in (0, 1), got {c}")
        self.c = c
        self.order = order

    def relevance(self, adj_norm):
        r"""Truncated Neumann-series relevance (Eq. 14), all nodes at once.

        Parameters
        ----------
        adj_norm : torch.Tensor
            Dense normalized adjacency of one graph, ``(n, n)``.

        Returns
        -------
        torch.Tensor
            :math:`(1-c)\sum_{k=0}^{K} c^k \tilde A^k`, shape ``(n, n)``;
            column ``v`` is :math:`r_v`.
        """
        n = adj_norm.shape[0]
        term = torch.eye(n, device=adj_norm.device, dtype=adj_norm.dtype)
        result = term.clone()
        for k in range(1, self.order + 1):
            term = adj_norm @ term
            result = result + self.c**k * term
        return (1.0 - self.c) * result

    def forward(self, adj_norm, patch_size):
        r"""Patch indices via :math:`\phi_{fast}` (Eq. 15).

        Parameters
        ----------
        adj_norm : torch.Tensor
            Dense symmetric normalized adjacency of one graph, ``(n, n)``.
        patch_size : int
            Patch size ``p``.

        Returns
        -------
        tuple of torch.Tensor
            Patch indices of shape ``(n, p)`` (row ``v`` starts with ``v``,
            no duplicates unless padded for ``n < p``) and their relevance
            scores of shape ``(n, p)``.
        """
        return _top_p_col(self.relevance(adj_norm), patch_size, mask_self=True)


def _top_p_col(relevance, patch_size, mask_self):
    """Select top-``p`` patch indices per column of a relevance matrix.

    Row ``v`` of the result indexes node ``v``'s patch: ``v`` itself first,
    then the highest-scoring entries of column ``v``. Graphs with fewer than
    ``patch_size`` nodes are padded by repeating ``v``.

    Parameters
    ----------
    relevance : torch.Tensor
        Relevance matrix ``R`` of shape ``(n, n)``.
    patch_size : int
        Patch size ``p``.
    mask_self : bool
        If True, exclude ``v`` from the top-k of its own column so patches
        contain no duplicates (Fast-HeroFilter ranking semantics). If False,
        keep the reference spectral behaviour where ``v`` may also appear
        inside the top-k.

    Returns
    -------
    tuple of torch.Tensor
        Patch indices of shape ``(n, p)`` and their relevance scores of
        shape ``(n, p)``; ``scores[v, j] = R[patch[v, j], v]``.
    """
    n = relevance.shape[0]
    self_idx = torch.arange(n, device=relevance.device)
    columns = relevance.T
    if mask_self:
        masked = columns.clone()
        masked[self_idx, self_idx] = float("-inf")
    else:
        masked = columns
    k = min(patch_size - 1, n)
    _, top_idx = torch.topk(masked, k, dim=1)
    patch = torch.cat([self_idx.unsqueeze(1), top_idx], dim=1)
    if patch.shape[1] < patch_size:
        pad = self_idx.unsqueeze(1).expand(n, patch_size - patch.shape[1])
        patch = torch.cat([patch, pad], dim=1)
    scores = columns.gather(1, patch)
    return patch, scores


class HeroFilterMLP(nn.Module):
    """Two-layer MLP used inside the mixer (Eq. 10-11 / reference ``MLP``).

    ``Linear -> ReLU -> Dropout -> Linear(bias=False) -> Dropout``, hidden
    width ``num_features * expansion_factor``, mirroring the reference
    implementation.

    Parameters
    ----------
    num_features : int
        Input width.
    expansion_factor : int
        Hidden width multiplier.
    dropout : float
        Dropout probability.
    out_features : int, optional
        Output width; defaults to ``num_features``.
    """

    def __init__(
        self, num_features, expansion_factor, dropout, out_features=None
    ):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(
            num_hidden, out_features or num_features, bias=False
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(..., num_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor ``(..., out_features)``.
        """
        x = self.dropout1(F.relu(self.fc1(x)))
        return self.dropout2(self.fc2(x))


class PatchMixer(nn.Module):
    r"""Patch-mixing layer, Eq. 10: :math:`\hat P_v = \mathrm{MLP}(\mathrm{LayerNorm}(P_v^\top))^\top`.

    Mixes information *across the patch dimension*, independently per
    feature, with a residual connection as in the reference implementation.

    Parameters
    ----------
    num_features : int
        Feature width ``d``.
    num_patches : int
        Patch size ``p``.
    expansion_factor : int
        MLP hidden width multiplier.
    dropout : float
        Dropout probability.
    """

    def __init__(self, num_features, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = HeroFilterMLP(num_patches, expansion_factor, dropout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Patch tensor of shape ``(n, p, d)``.

        Returns
        -------
        torch.Tensor
            Mixed tensor of shape ``(n, p, d)``.
        """
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.mlp(x).transpose(1, 2)
        return x + residual


class FeatureMixer(nn.Module):
    r"""Feature-mixing layer, Eq. 11: :math:`\tilde P_v = \mathrm{MLP}(\mathrm{LayerNorm}(\hat P_v))`.

    Mixes information *across the feature dimension*, independently per
    patch slot, with a residual connection as in the reference
    implementation.

    Parameters
    ----------
    num_features : int
        Feature width ``d``.
    expansion_factor : int
        MLP hidden width multiplier.
    dropout : float
        Dropout probability.
    """

    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = HeroFilterMLP(num_features, expansion_factor, dropout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Patch tensor of shape ``(n, p, d)``.

        Returns
        -------
        torch.Tensor
            Mixed tensor of shape ``(n, p, d)``.
        """
        return self.mlp(self.norm(x)) + x


class MixerLayer(nn.Module):
    """One HeroFilter mixer block: patch-mixing (Eq. 10) + feature-mixing (Eq. 11).

    The ``use_patch_mixing`` / ``use_feature_mixing`` flags enable the
    Table 6 ablations (+PatchMixer / +FeatureMixer / NoMixer).

    Parameters
    ----------
    num_features : int
        Feature width ``d``.
    num_patches : int
        Patch size ``p``.
    expansion_factor : int
        MLP hidden width multiplier.
    dropout : float
        Dropout probability.
    use_patch_mixing : bool, optional
        Apply Eq. 10 (default: True).
    use_feature_mixing : bool, optional
        Apply Eq. 11 (default: True).
    """

    def __init__(
        self,
        num_features,
        num_patches,
        expansion_factor,
        dropout,
        use_patch_mixing=True,
        use_feature_mixing=True,
    ):
        super().__init__()
        self.patch_mixer = (
            PatchMixer(num_features, num_patches, expansion_factor, dropout)
            if use_patch_mixing
            else None
        )
        self.feature_mixer = (
            FeatureMixer(num_features, expansion_factor, dropout)
            if use_feature_mixing
            else None
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Patch tensor of shape ``(n, p, d)``.

        Returns
        -------
        torch.Tensor
            Mixed tensor of shape ``(n, p, d)``.
        """
        if self.patch_mixer is not None:
            x = self.patch_mixer(x)
        if self.feature_mixer is not None:
            x = self.feature_mixer(x)
        return x


class HeroFilter(nn.Module):
    r"""HeroFilter backbone (arXiv:2510.10864), Track 1 / TDL Challenge 2026.

    Pipeline per forward pass: segment the (possibly block-diagonal) batch by
    the ``batch`` vector; per graph, build the dense symmetric normalized
    adjacency and run the selected patcher to get patch indices (Eq. 8 /
    Eq. 15); gather the patch tensor :math:`P \in \mathbb{R}^{n \times p
    \times d}` (Eq. 9); embed patches; apply ``num_layers`` mixer blocks
    (Eq. 10-11); aggregate over the patch dimension (Section 4.2) with an
    optional residual on the embedded input features (reference forward);
    LayerNorm and return node embeddings.

    Patch selection is segmented per graph, so nodes never select
    "neighbors" from other graphs in a block-diagonal batch, and graphs
    smaller than ``patch_size`` are padded with the node itself.

    Parameters
    ----------
    in_channels : int
        Input feature width (output width of the feature encoder).
    hidden_channels : int
        Node embedding width.
    patcher : str, optional
        ``"fast"`` (Eq. 12-15, default), ``"spectral"`` (Eq. 6-9 with binned
        learnable :math:`w_k`) or ``"reference"`` (Eq. 6-9 with the released
        code's frozen :math:`w_k^k` filter).
    patch_size : int, optional
        Patch size ``p`` (default: 16).
    order : int, optional
        Polynomial order ``K``: filter order in Eq. 6 for the spectral
        patchers, Neumann truncation in Eq. 14 for the fast patcher
        (default: 10).
    c : float, optional
        Teleport factor of Eq. 12-14, fast patcher only (default: 0.5).
    num_bins : int, optional
        Frequency bins of the binned filter (default: 16).
    sigma : str, optional
        Filter activation in Eq. 6 (default: ``"identity"``, matching the
        released code; satisfies Prop. 2's sigma(0)=0).
    num_layers : int, optional
        Number of mixer blocks (default: 2).
    expansion_factor : int, optional
        Mixer MLP width multiplier (default: 1).
    dropout : float, optional
        Dropout probability (default: 0.0).
    aggregation : str, optional
        Aggregation over the patch dimension, ``"sum"`` or ``"mean"``
        (Section 4.2; default: ``"sum"``, the pseudocode's summation).
    residual : bool, optional
        Add the embedded input features to the aggregated patches, as in the
        reference spectral forward (default: True).
    weight_patches : bool, optional
        Multiply each patch by the softmax of its selected relevance scores.
        Eq. 8-9 select indices, which is non-differentiable, so without this
        the loss provides no gradient to the filter weights of Eq. 6 (true
        of the released code as well). Defaults to True for the
        ``"spectral"`` patcher and False otherwise.
    use_patch_mixing : bool, optional
        Enable Eq. 10 (default: True). Table 6 ablation flag.
    use_feature_mixing : bool, optional
        Enable Eq. 11 (default: True). Table 6 ablation flag.
    seed : int, optional
        Seed for the frozen ``"reference"`` filter weights (default: 0).
    **kwargs
        Additional arguments (ignored).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        patcher="fast",
        patch_size=16,
        order=10,
        c=0.5,
        num_bins=16,
        sigma="identity",
        num_layers=2,
        expansion_factor=1,
        dropout=0.0,
        aggregation="sum",
        residual=True,
        weight_patches=None,
        use_patch_mixing=True,
        use_feature_mixing=True,
        seed=0,
        **kwargs,
    ):
        super().__init__()
        if aggregation not in ("sum", "mean"):
            raise ValueError(f"Unknown aggregation '{aggregation}'")
        if weight_patches is None:
            weight_patches = patcher == "spectral"
        self.weight_patches = weight_patches
        if patcher == "fast":
            self.patcher = FastPatcher(c=c, order=order)
        elif patcher in ("spectral", "reference"):
            self.patcher = SpectralPatcher(
                order,
                num_bins=num_bins,
                sigma=sigma,
                filter_mode="binned" if patcher == "spectral" else "reference",
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown patcher '{patcher}'")
        self.out_channels = hidden_channels
        self.patch_size = patch_size
        self.aggregation = aggregation
        self.residual = residual
        # Patch embedding: the reference's `dim_red` MLP (expansion 2),
        # applied both to patches and to the residual input features.
        self.dim_red = HeroFilterMLP(
            in_channels, 2, dropout, out_features=hidden_channels
        )
        self.mixers = nn.Sequential(
            *[
                MixerLayer(
                    hidden_channels,
                    patch_size,
                    expansion_factor,
                    dropout,
                    use_patch_mixing=use_patch_mixing,
                    use_feature_mixing=use_feature_mixing,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_channels)

    @staticmethod
    def normalized_adjacency(edge_index, num_nodes, edge_weight=None):
        r"""Dense symmetric normalized adjacency :math:`D^{-1/2} A D^{-1/2}`.

        The adjacency is symmetrized with an elementwise maximum so
        ``torch.linalg.eigh`` is applicable; rows/columns of isolated nodes
        stay zero.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of one graph (local numbering), shape ``(2, e)``.
        num_nodes : int
            Number of nodes ``n``.
        edge_weight : torch.Tensor, optional
            Edge weights of shape ``(e,)``; defaults to ones.

        Returns
        -------
        torch.Tensor
            Dense normalized adjacency of shape ``(n, n)``.
        """
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        weight = (
            edge_weight
            if edge_weight is not None
            else torch.ones(edge_index.shape[1], device=edge_index.device)
        )
        adj[edge_index[0], edge_index[1]] = weight
        adj = torch.maximum(adj, adj.T)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0.0
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    def build_patches(self, x, edge_index, batch, edge_weight=None):
        r"""Patch tensor :math:`P \in \mathbb{R}^{n \times p \times d}` (Eq. 9).

        Runs the patcher independently on every graph segment of the batch
        (block-diagonal batching) and maps local patch indices back to batch
        coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(n, d)``.
        edge_index : torch.Tensor
            Batched edge indices of shape ``(2, e)``.
        batch : torch.Tensor
            Graph assignment vector of shape ``(n,)``.
        edge_weight : torch.Tensor, optional
            Edge weights of shape ``(e,)``.

        Returns
        -------
        tuple of torch.Tensor
            Patch tensor of shape ``(n, patch_size, d)`` and relevance
            scores of shape ``(n, patch_size)``.
        """
        counts = torch.bincount(batch)
        edge_graph = batch[edge_index[0]]
        indices = []
        scores = []
        offset = 0
        for graph, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            mask = edge_graph == graph
            local_edges = edge_index[:, mask] - offset
            local_weight = (
                edge_weight[mask] if edge_weight is not None else None
            )
            adj_norm = self.normalized_adjacency(
                local_edges, count, local_weight
            )
            local_idx, local_scores = self.patcher(adj_norm, self.patch_size)
            indices.append(local_idx + offset)
            scores.append(local_scores)
            offset += count
        return x[torch.cat(indices, dim=0)], torch.cat(scores, dim=0)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        """Forward pass; returns node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(n, in_channels)``.
        edge_index : torch.Tensor
            Edge indices of shape ``(2, e)``.
        batch : torch.Tensor, optional
            Graph assignment vector of shape ``(n,)``; ``None`` means a
            single graph (transductive setting).
        edge_weight : torch.Tensor, optional
            Edge weights of shape ``(e,)``.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``(n, hidden_channels)``.
        """
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        patches, scores = self.build_patches(x, edge_index, batch, edge_weight)
        if self.weight_patches:
            patches = patches * F.softmax(scores, dim=1).unsqueeze(-1)
        patches = self.dim_red(patches)
        patches = self.mixers(patches)
        embedding = patches.sum(dim=1)
        if self.aggregation == "mean":
            embedding = embedding / patches.shape[1]
        if self.residual:
            embedding = embedding + self.dim_red(x)
        return self.norm(embedding)
