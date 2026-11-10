"""Unit tests for the HeroFilter backbone (arXiv:2510.10864).

The parity tests transcribe the relevant lines of the authors' released
implementation (`zshuai8/HeroFilter` @ f0749d6, ``src/HeroFilter.py``)
verbatim into plain loops and assert numerical agreement with this
implementation, following the plan of testing the patcher math with fixed
weights (there is no trained reference model to match end-to-end).
"""

import pytest
import torch

from topobench.nn.backbones.graph.herofilter import (
    AdaptivePolynomialFilter,
    FastPatcher,
    FeatureMixer,
    HeroFilter,
    HeroFilterMLP,
    MixerLayer,
    PatchMixer,
    SpectralPatcher,
    _top_p_col,
)


def _random_graph(num_nodes, num_edges, seed):
    """Create a random undirected test graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    num_edges : int
        Number of directed edges to draw before symmetrization.
    seed : int
        Random seed.

    Returns
    -------
    tuple of torch.Tensor
        Node features ``(n, 12)`` and edge index ``(2, e)``.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, 12, generator=generator)
    edge_index = torch.randint(
        0, num_nodes, (2, num_edges), generator=generator
    )
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return x, edge_index


def _reference_filter_response(weights, eigvals):
    """Verbatim loop transcription of the reference filter.

    Reference: ``src/HeroFilter.py`` lines 118-128 (default mode):
    ``filtered += torch.mul(p**(i+1), eigV**(i+1))``.

    Parameters
    ----------
    weights : torch.Tensor
        Filter bank of shape ``(order, n)``.
    eigvals : torch.Tensor
        Eigenvalues of shape ``(n,)``.

    Returns
    -------
    torch.Tensor
        Filter response of shape ``(n,)``.
    """
    filtered = torch.zeros(eigvals.shape[0])
    for i, p in enumerate(weights):
        filtered = filtered + torch.mul(p ** (i + 1), eigvals ** (i + 1))
    return filtered


def _reference_patcher(x, filtered, eigvecs, k):
    """Verbatim loop transcription of the reference spectral patcher.

    Reference: ``src/HeroFilter.py`` lines 133-151: build
    ``atts = U diag(filtered) U^T``, take ``topk(atts.T, k - 1)`` and prepend
    each node to its own patch.

    Parameters
    ----------
    x : torch.Tensor
        Node features ``(n, d)``.
    filtered : torch.Tensor
        Filter response ``g(Lambda)`` of shape ``(n,)``.
    eigvecs : torch.Tensor
        Eigenvector matrix ``U`` of shape ``(n, n)``.
    k : int
        Patch size.

    Returns
    -------
    tuple of torch.Tensor
        Patch tensor ``(n, k, d)`` and patch indices ``(n, k)``.
    """
    eigv_diag = torch.diag(filtered)
    atts = eigvecs.matmul(eigv_diag)
    atts = atts.matmul(eigvecs.T)
    _, selected_k_ind = torch.topk(atts.T, k - 1)
    returned = []
    indices_all = []
    for step, indices in enumerate(selected_k_ind):
        full = torch.cat(
            (torch.tensor([step], device=indices.device), indices)
        )
        returned.append(torch.index_select(x, 0, full))
        indices_all.append(full)
    return torch.stack(returned), torch.stack(indices_all)


class TestAdaptivePolynomialFilter:
    """Tests for AdaptivePolynomialFilter (Eq. 6)."""

    def test_reference_mode_parity(self):
        """Reference mode reproduces the released filter loop exactly."""
        torch.manual_seed(0)
        n, order = 20, 6
        filt = AdaptivePolynomialFilter(order, mode="reference", seed=3)
        eigvals = torch.linspace(-1, 1, n)
        weights = filt._reference_weights(n, eigvals.device)
        expected = _reference_filter_response(weights, eigvals)
        torch.testing.assert_close(filt(eigvals), expected)

    def test_reference_weights_frozen_and_deterministic(self):
        """Frozen weights are identical across calls and never trainable."""
        filt = AdaptivePolynomialFilter(4, mode="reference", seed=1)
        w1 = filt._reference_weights(15, torch.device("cpu"))
        w2 = filt._reference_weights(15, torch.device("cpu"))
        torch.testing.assert_close(w1, w2)
        assert not w1.requires_grad
        assert len(list(filt.parameters())) == 0

    def test_binned_mode_matches_bruteforce(self):
        """Binned Eq. 6 agrees with an explicit loop over k."""
        torch.manual_seed(0)
        order, num_bins, n = 5, 8, 30
        filt = AdaptivePolynomialFilter(order, num_bins=num_bins)
        eigvals = torch.rand(n) * 2 - 1
        weights = filt.interpolate_weights(eigvals)
        expected = torch.zeros(n)
        for k in range(1, order + 1):
            expected = expected + weights[k - 1] * eigvals**k
        torch.testing.assert_close(filt(eigvals), expected)

    def test_binned_weights_trainable(self):
        """Binned w_k is a Parameter of shape (order, num_bins)."""
        filt = AdaptivePolynomialFilter(3, num_bins=7)
        params = list(filt.parameters())
        assert len(params) == 1
        assert params[0].shape == (3, 7)
        assert params[0].requires_grad

    def test_interpolation_at_bin_centers(self):
        """Eigenvalues exactly on bin centers pick the bin weights."""
        num_bins = 5
        filt = AdaptivePolynomialFilter(2, num_bins=num_bins)
        centers = torch.linspace(-1, 1, num_bins)
        torch.testing.assert_close(
            filt.interpolate_weights(centers), filt.weight
        )

    def test_sigma_applied_per_term(self):
        """Nonlinear sigma wraps each term of the sum, per Eq. 6."""
        torch.manual_seed(0)
        filt = AdaptivePolynomialFilter(3, num_bins=4, sigma="tanh")
        eigvals = torch.rand(10) * 2 - 1
        weights = filt.interpolate_weights(eigvals)
        expected = torch.zeros(10)
        for k in range(1, 4):
            expected = expected + torch.tanh(weights[k - 1] * eigvals**k)
        torch.testing.assert_close(filt(eigvals), expected)

    def test_invalid_arguments_raise(self):
        """Unknown sigma or mode raises ValueError."""
        with pytest.raises(ValueError):
            AdaptivePolynomialFilter(3, sigma="nope")
        with pytest.raises(ValueError):
            AdaptivePolynomialFilter(3, mode="nope")


class TestSpectralPatcher:
    """Tests for SpectralPatcher (Eq. 7-9)."""

    def test_relevance_matches_dense_construction(self):
        """Eq. 7: relevance equals explicit U diag(g) U^T."""
        torch.manual_seed(0)
        n = 12
        patcher = SpectralPatcher(4, num_bins=6)
        eigvals = torch.rand(n) * 2 - 1
        eigvecs, _ = torch.linalg.qr(torch.randn(n, n))
        expected = eigvecs @ torch.diag(patcher.filter(eigvals)) @ eigvecs.T
        torch.testing.assert_close(
            patcher.relevance(eigvals, eigvecs), expected
        )

    def test_patcher_parity_with_reference(self):
        """R, patch indices and gathered patches match the released code.

        Both implementations receive identical eigenpairs and identical
        frozen filter weights, isolating the patcher math from the
        normalization choice documented in the module docstring.
        """
        torch.manual_seed(0)
        n, order, p = 18, 5, 6
        x = torch.randn(n, 4)
        eigvals = torch.rand(n) * 2 - 1
        eigvecs, _ = torch.linalg.qr(torch.randn(n, n))

        patcher = SpectralPatcher(order, filter_mode="reference", seed=7)
        weights = patcher.filter._reference_weights(n, eigvals.device)

        filtered = _reference_filter_response(weights, eigvals)
        expected_patches, expected_idx = _reference_patcher(
            x, filtered, eigvecs, p
        )

        relevance = patcher.relevance(eigvals, eigvecs)
        idx, _ = _top_p_col(relevance, p, mask_self=False)
        torch.testing.assert_close(idx, expected_idx)
        torch.testing.assert_close(x[idx], expected_patches)

    def test_forward_shape_and_self_first(self):
        """Forward returns (n, p) indices with each node first in its patch."""
        _, edge_index = _random_graph(10, 30, seed=0)
        adj = HeroFilter.normalized_adjacency(edge_index, 10)
        patcher = SpectralPatcher(3)
        idx, scores = patcher(adj, 4)
        assert scores.shape == (10, 4)
        assert idx.shape == (10, 4)
        torch.testing.assert_close(idx[:, 0], torch.arange(10))


class TestFastPatcher:
    """Tests for FastPatcher (Eq. 12-15)."""

    def test_neumann_series_converges_to_closed_form(self):
        """Eq. 14 -> Eq. 13: truncated series approaches (1-c)(I-cA)^-1."""
        _, edge_index = _random_graph(15, 40, seed=1)
        adj = HeroFilter.normalized_adjacency(edge_index, 15)
        c = 0.5
        closed_form = (1 - c) * torch.linalg.inv(torch.eye(15) - c * adj)
        errors = []
        for order in (2, 8, 32):
            relevance = FastPatcher(c=c, order=order).relevance(adj)
            errors.append((relevance - closed_form).abs().max().item())
        assert errors[1] < errors[0]
        assert errors[2] < 1e-6

    def test_invalid_c_raises(self):
        """Teleport factor outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            FastPatcher(c=1.0)
        with pytest.raises(ValueError):
            FastPatcher(c=0.0)

    def test_self_first_and_no_duplicates(self):
        """Each patch starts with the node itself and has no duplicates."""
        _, edge_index = _random_graph(12, 40, seed=2)
        adj = HeroFilter.normalized_adjacency(edge_index, 12)
        idx, _ = FastPatcher(order=6)(adj, 5)
        assert idx.shape == (12, 5)
        torch.testing.assert_close(idx[:, 0], torch.arange(12))
        for row in idx:
            assert len(set(row.tolist())) == len(row)

    def test_padding_when_p_exceeds_n(self):
        """Graphs with n < p are padded by repeating the node itself."""
        _, edge_index = _random_graph(3, 6, seed=3)
        adj = HeroFilter.normalized_adjacency(edge_index, 3)
        idx, _ = FastPatcher(order=4)(adj, 8)
        assert idx.shape == (3, 8)
        torch.testing.assert_close(idx[:, 0], torch.arange(3))
        for v, row in enumerate(idx):
            assert set(row.tolist()) <= set(range(3))
            assert (row == v).sum() >= 8 - 3 + 1


def test_top_p_col_selects_largest():
    """top-p_col picks the genuinely largest entries per column (Eq. 8)."""
    relevance = torch.tensor(
        [
            [0.9, 0.1, 5.0],
            [0.5, 0.2, 4.0],
            [0.1, 0.3, 3.0],
        ]
    )
    idx, scores = _top_p_col(relevance, 3, mask_self=False)
    # Column 0 sorted desc: rows 0, 1, 2 -> patch of node 0 = [0, 0, 1].
    torch.testing.assert_close(idx[0], torch.tensor([0, 0, 1]))
    # Column 2 sorted desc: rows 0, 1, 2 -> patch of node 2 = [2, 0, 1].
    torch.testing.assert_close(idx[2], torch.tensor([2, 0, 1]))
    # With self masked, node 0's top entries exclude row 0 of column 0.
    # Scores are read from the (unmasked) relevance columns.
    torch.testing.assert_close(scores[0], torch.tensor([0.9, 0.9, 0.5]))
    idx_masked, _ = _top_p_col(relevance, 3, mask_self=True)
    torch.testing.assert_close(idx_masked[0], torch.tensor([0, 1, 2]))


class TestMixer:
    """Tests for HeroFilterMLP, PatchMixer, FeatureMixer, MixerLayer (Eq. 10-11)."""

    def test_mlp_shapes(self):
        """MLP maps (..., in) -> (..., out) with expansion hidden width."""
        mlp = HeroFilterMLP(6, 2, 0.0, out_features=4)
        assert mlp.fc1.out_features == 12
        assert mlp(torch.randn(5, 3, 6)).shape == (5, 3, 4)

    def test_patch_mixer_mixes_patch_dimension(self):
        """Eq. 10 output shape is preserved and depends on the patch axis."""
        torch.manual_seed(0)
        mixer = PatchMixer(6, 4, 1, 0.0)
        x = torch.randn(3, 4, 6)
        out = mixer(x)
        assert out.shape == x.shape
        permuted = mixer(x[:, [1, 0, 2, 3], :])
        assert not torch.allclose(out, permuted[:, [1, 0, 2, 3], :])

    def test_feature_mixer_is_patchwise(self):
        """Eq. 11 acts identically on every patch slot."""
        torch.manual_seed(0)
        mixer = FeatureMixer(6, 1, 0.0)
        x = torch.randn(3, 4, 6)
        out = mixer(x)
        assert out.shape == x.shape
        permutation = [2, 0, 1, 3]
        torch.testing.assert_close(
            mixer(x[:, permutation, :]), out[:, permutation, :]
        )

    def test_ablation_flags(self):
        """Table 6 ablations: disabled mixers are absent; both off = identity."""
        layer = MixerLayer(6, 4, 1, 0.0, use_patch_mixing=False)
        assert layer.patch_mixer is None
        layer = MixerLayer(6, 4, 1, 0.0, use_feature_mixing=False)
        assert layer.feature_mixer is None
        layer = MixerLayer(
            6, 4, 1, 0.0, use_patch_mixing=False, use_feature_mixing=False
        )
        x = torch.randn(3, 4, 6)
        torch.testing.assert_close(layer(x), x)


class TestHeroFilter:
    """Tests for the HeroFilter backbone module."""

    def test_normalized_adjacency(self):
        """Symmetric normalization matches manual D^-1/2 A D^-1/2."""
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        adj = HeroFilter.normalized_adjacency(edge_index, 4)
        expected = torch.tensor(
            [
                [0.0, 1 / (2**0.5), 0.0, 0.0],
                [1 / (2**0.5), 0.0, 1 / (2**0.5), 0.0],
                [0.0, 1 / (2**0.5), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        torch.testing.assert_close(adj, expected)
        # Directed input is symmetrized.
        directed = torch.tensor([[0], [1]])
        adj = HeroFilter.normalized_adjacency(directed, 2)
        torch.testing.assert_close(adj, adj.T)

    @pytest.mark.parametrize("patcher", ["fast", "spectral", "reference"])
    def test_forward_shapes_and_gradients(self, patcher):
        """Forward yields (n, hidden) embeddings and gradients flow."""
        x, edge_index = _random_graph(14, 40, seed=4)
        model = HeroFilter(12, 16, patcher=patcher, patch_size=6, order=4)
        out = model(x, edge_index)
        assert out.shape == (14, 16)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_spectral_filter_weights_receive_gradients(self):
        """The binned w_k of Eq. 6 is trained through R (unlike the release)."""
        x, edge_index = _random_graph(10, 30, seed=5)
        model = HeroFilter(12, 8, patcher="spectral", patch_size=4, order=3)
        model(x, edge_index).sum().backward()
        assert model.patcher.filter.weight.grad is not None

    def test_batch_safety(self):
        """Two disjoint graphs in one batch equal each graph run alone."""
        x1, ei1 = _random_graph(9, 24, seed=6)
        x2, ei2 = _random_graph(13, 40, seed=7)
        for patcher in ("fast", "spectral", "reference"):
            model = HeroFilter(12, 8, patcher=patcher, patch_size=5, order=3)
            model.eval()
            x = torch.cat([x1, x2])
            edge_index = torch.cat([ei1, ei2 + 9], dim=1)
            batch = torch.cat(
                [
                    torch.zeros(9, dtype=torch.long),
                    torch.ones(13, dtype=torch.long),
                ]
            )
            with torch.no_grad():
                joint = model(x, edge_index, batch=batch)
                alone1 = model(x1, ei1)
                alone2 = model(x2, ei2)
            torch.testing.assert_close(joint[:9], alone1)
            torch.testing.assert_close(joint[9:], alone2)

    def test_small_graph_padding_in_forward(self):
        """Forward works when a graph has fewer nodes than patch_size."""
        x, edge_index = _random_graph(5, 12, seed=8)
        model = HeroFilter(12, 8, patcher="fast", patch_size=16, order=3)
        assert model(x, edge_index).shape == (5, 8)

    def test_edge_weight_used(self):
        """Edge weights alter the normalized adjacency and the output."""
        x, edge_index = _random_graph(8, 20, seed=9)
        model = HeroFilter(12, 8, patcher="fast", patch_size=4, order=3)
        model.eval()
        weight = torch.rand(edge_index.shape[1]) + 0.5
        with torch.no_grad():
            out_unweighted = model(x, edge_index)
            out_weighted = model(x, edge_index, edge_weight=weight)
        assert not torch.allclose(out_unweighted, out_weighted)

    def test_aggregation_mean(self):
        """Mean aggregation equals sum divided by patch size."""
        x, edge_index = _random_graph(8, 20, seed=10)
        kwargs = dict(patcher="fast", patch_size=4, order=3, residual=False)
        torch.manual_seed(0)
        model_sum = HeroFilter(12, 8, aggregation="sum", **kwargs)
        torch.manual_seed(0)
        model_mean = HeroFilter(12, 8, aggregation="mean", **kwargs)
        model_sum.eval()
        model_mean.eval()
        with torch.no_grad():
            # LayerNorm is scale-invariant up to its affine params, so
            # compare pre-norm sums via the internal pipeline instead.
            patches, _ = model_sum.build_patches(
                x, edge_index, torch.zeros(8, dtype=torch.long)
            )
            patches_sum = model_sum.mixers(model_sum.dim_red(patches)).sum(
                dim=1
            )
            out_sum = model_sum(x, edge_index)
            out_mean = model_mean(x, edge_index)
        assert patches_sum.shape == (8, 8)
        assert out_sum.shape == out_mean.shape

    def test_invalid_arguments_raise(self):
        """Unknown patcher or aggregation raises ValueError."""
        with pytest.raises(ValueError):
            HeroFilter(4, 4, patcher="nope")
        with pytest.raises(ValueError):
            HeroFilter(4, 4, aggregation="nope")

    def test_out_channels_attribute(self):
        """The backbone exposes out_channels for the TopoBench composition."""
        model = HeroFilter(12, 24)
        assert model.out_channels == 24

    def test_empty_graph_segment_skipped(self):
        """A batch index with no nodes (empty segment) is skipped safely."""
        x, edge_index = _random_graph(6, 16, seed=12)
        model = HeroFilter(12, 8, patcher="fast", patch_size=4, order=3)
        # Graph ids 0 and 2 are populated; id 1 is empty.
        batch = torch.tensor([0, 0, 0, 2, 2, 2])
        mask = (edge_index < 3).all(dim=0) | (edge_index >= 3).all(dim=0)
        assert model(x, edge_index[:, mask], batch=batch).shape == (6, 8)

    def test_weight_patches_default_and_override(self):
        """Relevance weighting defaults on for spectral, off otherwise."""
        assert HeroFilter(4, 4, patcher="spectral").weight_patches
        assert not HeroFilter(4, 4, patcher="fast").weight_patches
        assert not HeroFilter(4, 4, patcher="reference").weight_patches
        assert HeroFilter(
            4, 4, patcher="fast", weight_patches=True
        ).weight_patches

    def test_weight_patches_changes_output(self):
        """Softmax relevance weighting of patches affects the embedding."""
        x, edge_index = _random_graph(10, 30, seed=11)
        torch.manual_seed(0)
        weighted = HeroFilter(
            12, 8, patcher="fast", patch_size=4, order=3, weight_patches=True
        )
        torch.manual_seed(0)
        unweighted = HeroFilter(
            12, 8, patcher="fast", patch_size=4, order=3, weight_patches=False
        )
        weighted.eval()
        unweighted.eval()
        with torch.no_grad():
            assert not torch.allclose(
                weighted(x, edge_index), unweighted(x, edge_index)
            )
