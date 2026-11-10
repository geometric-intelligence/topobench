"""Unit tests for FilterBankGNN, the Fusion/Channel infrastructure, and ACM-GNN.

These tests stress the filter-bank abstraction: Q parallel polynomial-filter
channels (reusing the basis registry) fused by a swappable Fusion. The
ACM-GNN tests pin the concrete LP/HP/ID linear filters.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from topobench.nn.backbones.graph.filter_bank.channel import (
    Channel,
    GaussianChannel,
    PPRChannel,
    apply_polynomial_filter,
    build_laplacian_apply,
)
from topobench.nn.backbones.graph.filter_bank.fusion import Fusion, SumFusion
from topobench.nn.backbones.graph.filter_bank_gnn import (
    ACMGNN,
    FAGCN,
    FBGNN,
    G2CN,
    GNNLFHF,
    FiGURe,
    FilterBankGNN,
)
from topobench.nn.backbones.graph.poly_filter.bases.chebyshev import Chebyshev
from topobench.nn.backbones.graph.poly_filter.bases.jacobi import Jacobi
from topobench.nn.backbones.graph.poly_filter.bases.legendre import Legendre
from topobench.nn.backbones.graph.poly_filter.bases.monomial import Monomial
from topobench.utils.config_resolvers import register_all_resolvers


def _ring_edge_index(n: int) -> Tensor:
    """Undirected ring graph on ``n`` nodes as a PyG edge_index."""
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


class TestSumFusion:
    """Tests for the default weighted-sum fusion."""

    def test_sum_fusion_weights_channels(self):
        """``SumFusion`` returns ``Σ_q γ_q · outs[q]``."""
        outs = [torch.ones(3, 2), 2.0 * torch.ones(3, 2), 3.0 * torch.ones(3, 2)]
        gamma = torch.tensor([1.0, 0.5, 2.0])
        y = SumFusion()(outs, gamma)
        # 1*1 + 0.5*2 + 2*3 = 1 + 1 + 6 = 8
        assert torch.allclose(y, 8.0 * torch.ones(3, 2))

    def test_sum_fusion_single_channel(self):
        """One channel reduces to ``γ_0 · out_0``."""
        out = torch.randn(4, 2)
        y = SumFusion()([out], torch.tensor([3.0]))
        assert torch.allclose(y, 3.0 * out)


class TestChannel:
    """Tests for a single filter-bank channel."""

    def test_fixed_theta_is_a_buffer_not_a_parameter(self):
        """A fixed-θ channel registers no learnable parameters of its own."""
        ch = Channel(Monomial(), K=1, theta=[1.0, -1.0])
        assert sum(p.numel() for p in ch.parameters()) == 0
        assert "theta" in dict(ch.named_buffers())

    def test_learnable_theta_is_a_parameter(self):
        """A θ=None channel makes the coefficients learnable."""
        ch = Channel(Monomial(), K=4, theta=None)
        assert ch.theta.requires_grad
        assert ch.theta.shape == (5,)

    def test_theta_length_validated(self):
        with pytest.raises(ValueError):
            Channel(Monomial(), K=1, theta=[1.0, 2.0, 3.0])  # needs K+1=2

    def test_fixed_monomial_channel_computes_linear_filter(self):
        """LP channel (θ=[1,-1], Monomial) computes ``(I - L̃) h``."""
        ch = Channel(Monomial(), K=1, theta=[1.0, -1.0])
        h = torch.randn(5, 3)
        L_apply = lambda v: 0.3 * v  # noqa: E731  -- L̃ = 0.3·I
        out = ch(L_apply, h)
        assert torch.allclose(out, h - 0.3 * h)  # (I - L̃) h = 0.7 h


class TestFilterBankGNN:
    """Stress tests for the generic backbone with hand-built channels."""

    def setup_method(self):
        """Set up shared fixtures."""
        torch.manual_seed(0)
        self.N, self.F = 8, 4
        self.edge_index = _ring_edge_index(self.N)
        self.x = torch.randn(self.N, self.F)

    def _backbone(self, channels, fusion=None):
        return FilterBankGNN(
            in_channels=self.F,
            hidden_channels=6,
            out_channels=3,
            channels=channels,
            fusion=fusion,
        )

    def test_forward_shape_and_finiteness(self):
        """End-to-end forward with two hand-built channels."""
        model = self._backbone(
            [Channel(Monomial(), K=2), Channel(Chebyshev(), K=2)]
        )
        y = model(self.x, self.edge_index)
        assert y.shape == (self.N, 3)
        assert torch.isfinite(y).all()

    def test_gamma_is_learnable_and_sized_to_channels(self):
        """One γ per channel, learnable, gradients flow back to it."""
        model = self._backbone(
            [Channel(Monomial(), K=2), Channel(Chebyshev(), K=2)]
        )
        assert model.gamma.shape == (2,)
        y = model(self.x, self.edge_index)
        y.sum().backward()
        assert model.gamma.grad is not None
        assert (model.gamma.grad.abs() > 0).any()

    def test_default_fusion_is_sum(self):
        """Omitting ``fusion`` defaults to SumFusion."""
        model = self._backbone([Channel(Monomial(), K=1)])
        assert isinstance(model.fusion, SumFusion)

    def test_custom_fusion_is_used_without_backbone_change(self):
        """A user Fusion subclass plugs in with no backbone edit."""
        calls = {}

        class _RecordingFusion(Fusion):
            def forward(self, channel_outs, gamma):
                calls["q"] = len(channel_outs)
                calls["gamma_len"] = gamma.numel()
                return channel_outs[0]

        model = self._backbone(
            [Channel(Monomial(), K=1), Channel(Monomial(), K=1)],
            fusion=_RecordingFusion(),
        )
        _ = model(self.x, self.edge_index)
        assert calls == {"q": 2, "gamma_len": 2}

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            self._backbone([])

    def test_invalid_laplacian_norm_raises(self):
        with pytest.raises(ValueError):
            FilterBankGNN(
                in_channels=self.F,
                hidden_channels=6,
                out_channels=3,
                channels=[Channel(Monomial(), K=1)],
                laplacian_norm="banana",
            )


class TestACMGNN:
    """Tests specific to the ACM-GNN variant (LP + HP + ID)."""

    def test_has_three_channels(self):
        m = ACMGNN(in_channels=4, hidden_channels=6, out_channels=3)
        assert len(m.channels) == 3
        assert m.gamma.shape == (3,)

    def test_channels_are_fixed_lp_hp_id(self):
        """The three channels compute (I-L̃)h, L̃h, and Ih respectively."""
        m = ACMGNN(in_channels=4, hidden_channels=6, out_channels=3)
        h = torch.randn(5, 2)
        L_apply = lambda v: 0.25 * v  # noqa: E731
        lp, hp, idc = (ch(L_apply, h) for ch in m.channels)
        assert torch.allclose(lp, h - 0.25 * h)   # I - L̃
        assert torch.allclose(hp, 0.25 * h)        # L̃
        assert torch.allclose(idc, h)              # I

    def test_only_gamma_is_learnable(self):
        """ACM-GNN channels are fixed; the learnable filter params are γ
        plus the pre/post MLPs (no per-channel θ)."""
        m = ACMGNN(in_channels=4, hidden_channels=6, out_channels=3)
        # No channel exposes a learnable theta.
        for ch in m.channels:
            assert sum(p.numel() for p in ch.parameters()) == 0
        assert m.gamma.requires_grad

    def test_backbone_runs_and_trains(self):
        torch.manual_seed(1)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        m = ACMGNN(in_channels=4, hidden_channels=6, out_channels=3)
        y = m(x, edge_index)
        assert y.shape == (8, 3)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert m.gamma.grad is not None
        assert (m.gamma.grad.abs() > 0).any()


class TestFBGNN:
    """Tests specific to the FBGNN variant (LP + HP, no identity)."""

    def test_has_two_channels(self):
        m = FBGNN(in_channels=4, hidden_channels=6, out_channels=3)
        assert len(m.channels) == 2
        assert m.gamma.shape == (2,)

    def test_channels_are_fixed_lp_hp(self):
        """The two channels compute (I-L̃)h and L̃h respectively."""
        m = FBGNN(in_channels=4, hidden_channels=6, out_channels=3)
        h = torch.randn(5, 2)
        L_apply = lambda v: 0.25 * v  # noqa: E731
        lp, hp = (ch(L_apply, h) for ch in m.channels)
        assert torch.allclose(lp, h - 0.25 * h)  # I - L̃
        assert torch.allclose(hp, 0.25 * h)  # L̃

    def test_only_gamma_is_learnable(self):
        """FBGNN channels are fixed; the learnable filter params are γ
        plus the pre/post MLPs (no per-channel θ)."""
        m = FBGNN(in_channels=4, hidden_channels=6, out_channels=3)
        for ch in m.channels:
            assert sum(p.numel() for p in ch.parameters()) == 0
        assert m.gamma.requires_grad

    def test_backbone_runs_and_trains(self):
        torch.manual_seed(1)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        m = FBGNN(in_channels=4, hidden_channels=6, out_channels=3)
        y = m(x, edge_index)
        assert y.shape == (8, 3)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert m.gamma.grad is not None
        assert (m.gamma.grad.abs() > 0).any()


class TestFiGURe:
    """Tests for the FiGURe variant (variable channels, learnable θ + γ)."""

    def test_default_channels_are_canonical_mono_cheb_bern(self):
        m = FiGURe(in_channels=4, hidden_channels=6, out_channels=3, K=4)
        assert len(m.channels) == 3
        assert type(m.channels[0].basis).__name__ == "Monomial"
        assert type(m.channels[1].basis).__name__ == "Chebyshev"
        assert type(m.channels[2].basis).__name__ == "Bernstein"

    def test_each_channel_has_learnable_theta(self):
        """Unlike ACM-GNN, every FiGURe channel carries a learnable θ."""
        m = FiGURe(in_channels=4, hidden_channels=6, out_channels=3, K=5)
        for ch in m.channels:
            assert ch.theta.requires_grad
            assert ch.theta.shape == (6,)  # K + 1

    def test_accepts_arbitrary_registry_bases(self):
        """Variable channels reuse the basis registry: any basis mix works."""
        m = FiGURe(
            in_channels=4,
            hidden_channels=6,
            out_channels=3,
            K=4,
            bases=[Monomial(), Chebyshev(), Jacobi(alpha=1.0, beta=1.0)],
        )
        assert len(m.channels) == 3
        assert m.gamma.shape == (3,)

    def test_gradients_flow_to_channel_thetas_and_gamma(self):
        """Both the per-channel θ and the fusion weights γ are trained."""
        torch.manual_seed(2)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        m = FiGURe(in_channels=4, hidden_channels=6, out_channels=3, K=4)
        y = m(x, edge_index)
        assert y.shape == (8, 3)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert m.gamma.grad is not None and (m.gamma.grad.abs() > 0).any()
        for ch in m.channels:
            assert ch.theta.grad is not None
            assert (ch.theta.grad.abs() > 0).any()

    def test_parameter_count(self):
        """Q channels of degree K contribute Q·(K+1) θ params plus Q γ."""
        K, Q = 4, 3
        m = FiGURe(
            in_channels=4,
            hidden_channels=6,
            out_channels=3,
            K=K,
            bases=[Monomial(), Chebyshev(), Legendre()],
        )
        theta_params = sum(ch.theta.numel() for ch in m.channels)
        assert theta_params == Q * (K + 1)
        assert m.gamma.numel() == Q


class TestFAGCN:
    """Tests specific to the FAGCN variant (LP + HP with scaling β)."""

    def test_has_two_channels(self):
        m = FAGCN(in_channels=4, hidden_channels=6, out_channels=3, beta=0.3)
        assert len(m.channels) == 2
        assert m.gamma.shape == (2,)

    def test_channels_encode_beta_scaled_lp_hp(self):
        """LP = (β+1)I - L̃, HP = (β-1)I + L̃ for the chosen β."""
        beta = 0.3
        m = FAGCN(in_channels=4, hidden_channels=6, out_channels=3, beta=beta)
        h = torch.randn(5, 2)
        L_apply = lambda v: 0.25 * v  # noqa: E731  -- L̃ = 0.25·I
        lp, hp = (ch(L_apply, h) for ch in m.channels)
        assert torch.allclose(lp, (beta + 1.0) * h - 0.25 * h)
        assert torch.allclose(hp, (beta - 1.0) * h + 0.25 * h)

    def test_beta_changes_the_fixed_coefficients(self):
        """Different β yields different (fixed) channel coefficients."""
        m1 = FAGCN(in_channels=4, hidden_channels=6, out_channels=3, beta=0.0)
        m2 = FAGCN(in_channels=4, hidden_channels=6, out_channels=3, beta=0.5)
        assert not torch.allclose(m1.channels[0].theta, m2.channels[0].theta)

    def test_only_gamma_is_learnable(self):
        m = FAGCN(in_channels=4, hidden_channels=6, out_channels=3)
        for ch in m.channels:
            assert sum(p.numel() for p in ch.parameters()) == 0
        assert m.gamma.requires_grad

    def test_backbone_runs_and_trains(self):
        torch.manual_seed(3)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        m = FAGCN(in_channels=4, hidden_channels=6, out_channels=3)
        y = m(x, edge_index)
        assert y.shape == (8, 3)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert m.gamma.grad is not None and (m.gamma.grad.abs() > 0).any()


class TestPPRChannel:
    """Tests for the GNN-LF/HF PPR channel with linear prefactor."""

    def test_matches_closed_form_on_scalar_laplacian(self):
        """For L̃ = c·I the channel reduces to a geometric series in (1-α)(1-c).

        ``(I + β L̃) Σ_k α(1-α)^k (I-L̃)^k h`` with ``L̃ = c·I`` becomes
        ``(1 + β c) · α · Σ_{k=0}^K [(1-α)(1-c)]^k · h``.
        """
        alpha, beta, c, K = 0.1, 0.5, 0.25, 3
        ch = PPRChannel(K=K, alpha=alpha, beta=beta)
        h = torch.randn(5, 2)
        L_apply = lambda v: c * v  # noqa: E731
        out = ch(L_apply, h)

        r = (1.0 - alpha) * (1.0 - c)
        geom = sum(r**k for k in range(K + 1))
        expected = (1.0 + beta * c) * alpha * geom * h
        assert torch.allclose(out, expected, atol=1e-6)

    def test_beta_zero_is_plain_ppr(self):
        """β = 0 drops the prefactor (pure PPR / low-pass)."""
        ch0 = PPRChannel(K=4, alpha=0.2, beta=0.0)
        h = torch.randn(4, 3)
        L_apply = lambda v: 0.3 * v  # noqa: E731
        # With prefactor β=0, output == the bare PPR sum.
        a = 0.2
        s = a * h
        u = h
        for k in range(1, 5):
            u = u - L_apply(u)
            s = s + a * (1 - a) ** k * u
        assert torch.allclose(ch0(L_apply, h), s, atol=1e-6)

    def test_has_no_learnable_parameters(self):
        ch = PPRChannel(K=10, alpha=0.1, beta=1.0)
        assert sum(p.numel() for p in ch.parameters()) == 0

    def test_alpha_range_validated(self):
        with pytest.raises(ValueError):
            PPRChannel(K=4, alpha=0.0, beta=0.0)
        with pytest.raises(ValueError):
            PPRChannel(K=4, alpha=1.0, beta=0.0)


class TestGaussianChannel:
    """Tests for the Gaussian band-pass channel of G²CN."""

    def test_matches_truncated_exp_on_scaled_identity(self):
        """On L̃ = c·I the channel is exp_K(α (shift - c)^2) · h."""
        h = torch.randn(4, 2)
        L_apply = lambda v: 0.5 * v  # noqa: E731  (L̃ = 0.5 I)
        ch = GaussianChannel(K=2, alpha=1.0, shift=2.0)
        out = ch(L_apply, h)
        # shift - c = 1.5; M = 1.5^2 = 2.25; Σ_{k=0}^2 2.25^k / k!
        # = 1 + 2.25 + 2.53125 = 5.78125
        assert torch.allclose(out, 5.78125 * h, atol=1e-5)

    def test_k0_is_the_identity_term(self):
        """K=0 leaves only the α^0/0! = 1 term, so g(L̃) h = h."""
        h = torch.randn(4, 2)
        ch = GaussianChannel(K=0, alpha=2.0, shift=1.3)
        assert torch.allclose(ch(lambda v: 0.7 * v, h), h)

    def test_has_no_learnable_parameters(self):
        ch = GaussianChannel(K=3, alpha=1.0, shift=1.5)
        assert sum(p.numel() for p in ch.parameters()) == 0


class TestGNNLFHF:
    """Tests specific to the GNN-LF/HF variant (two PPR channels)."""

    def test_has_two_ppr_channels(self):
        m = GNNLFHF(in_channels=4, hidden_channels=6, out_channels=3)
        assert len(m.channels) == 2
        assert all(isinstance(ch, PPRChannel) for ch in m.channels)
        assert m.gamma.shape == (2,)

    def test_default_lf_hf_betas(self):
        """Defaults give a positive-β LF channel and a negative-β HF
        channel, matching Liao's GNN-LF/HF sign convention."""
        m = GNNLFHF(in_channels=4, hidden_channels=6, out_channels=3)
        assert m.channels[0].beta > 0.0  # LF: positive prefactor
        assert m.channels[1].beta < 0.0  # HF: negative prefactor

    def test_only_gamma_is_learnable(self):
        m = GNNLFHF(in_channels=4, hidden_channels=6, out_channels=3)
        for ch in m.channels:
            assert sum(p.numel() for p in ch.parameters()) == 0
        assert m.gamma.requires_grad

    def test_backbone_runs_and_trains(self):
        torch.manual_seed(4)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        m = GNNLFHF(in_channels=4, hidden_channels=6, out_channels=3, K=5)
        y = m(x, edge_index)
        assert y.shape == (8, 3)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert m.gamma.grad is not None and (m.gamma.grad.abs() > 0).any()


class TestFilterBankHydraConfig:
    """Hydra composition smoke test for the four filter-bank variant configs.

    Composes the full ``run.yaml`` (with MUTAG, like ``test_pipeline``) for
    each ``graph/filter_bank_*`` config and asserts the backbone instantiates
    as the expected class with the expected number of channels. No training.
    """

    def setup_method(self):
        import hydra.core.global_hydra

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_all_resolvers()

    def _compose(self, model_config: str):
        import hydra

        with hydra.initialize(
            version_base=None,
            config_path="../../../../configs",
            job_name="filter_bank_hydra_smoke",
        ):
            return hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"model={model_config}",
                    "dataset=graph/MUTAG",
                    "paths=test",
                ],
                return_hydra_config=True,
            )

    @pytest.mark.parametrize(
        ("config_name", "class_name", "n_channels"),
        [
            ("graph/filter_bank_acmgnn", "ACMGNN", 3),
            ("graph/filter_bank_fagcn", "FAGCN", 2),
            ("graph/filter_bank_fbgnn", "FBGNN", 2),
            ("graph/filter_bank_figure", "FiGURe", 3),
            ("graph/filter_bank_g2cn", "G2CN", 2),
            ("graph/filter_bank_gnnlfhf", "GNNLFHF", 2),
        ],
    )
    def test_variant_config_composes_and_instantiates(
        self, config_name, class_name, n_channels
    ):
        """Each variant config resolves and builds the right backbone."""
        import hydra

        cfg = self._compose(config_name)
        assert cfg.model.backbone._target_ == (
            f"topobench.nn.backbones.{class_name}"
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        # Compare by class name, not isinstance: TopoBench's backbone
        # auto-discovery loads the module under a non-canonical name, so the
        # discovered class is a distinct object from the directly-imported one.
        assert type(backbone).__name__ == class_name
        assert len(backbone.channels) == n_channels
        assert backbone.gamma.shape == (n_channels,)


class TestG2CN:
    """Tests for the G²CN variant (two Gaussian channels)."""

    def test_has_two_gaussian_channels(self):
        m = G2CN(in_channels=4, hidden_channels=6, out_channels=3, K=4)
        assert len(m.channels) == 2
        assert all(
            type(c).__name__ == "GaussianChannel" for c in m.channels
        )
        assert m.gamma.shape == (2,)

    def test_lf_hf_shifts(self):
        """LF shift = 1 + β1 (> 1); HF shift = 1 - β2 (< 1)."""
        m = G2CN(
            in_channels=4, hidden_channels=6, out_channels=3, betas=(0.4, 0.3)
        )
        assert abs(m.channels[0].shift - 1.4) < 1e-6
        assert abs(m.channels[1].shift - 0.7) < 1e-6

    def test_backbone_runs_and_trains(self):
        torch.manual_seed(2)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        m = G2CN(in_channels=4, hidden_channels=6, out_channels=3, K=4)
        y = m(x, edge_index)
        assert y.shape == (8, 3)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert m.gamma.grad is not None
        assert (m.gamma.grad.abs() > 0).any()


class TestSharedFilterHelpers:
    """Reused accumulation / Laplacian helpers, mirroring the backbone."""

    def test_apply_polynomial_filter_matches_manual_monomial(self):
        """``apply_polynomial_filter`` with Monomial reproduces Σ θ_k L̃^k h."""
        h = torch.randn(5, 2)
        L_apply = lambda v: 0.5 * v  # noqa: E731
        theta = torch.tensor([1.0, 2.0, 3.0])  # K=2
        out = apply_polynomial_filter(Monomial(), theta, L_apply, h, K=2)
        # u0=h, u1=0.5h, u2=0.25h -> 1*h + 2*0.5h + 3*0.25h = (1+1+0.75)h
        assert torch.allclose(out, 2.75 * h)

    def test_build_laplacian_apply_matches_dense_sym_no_self_loops(self):
        """``self_loops=False`` gives L̃ = I - D^{-1/2} A D^{-1/2} (path 0-1-2)."""
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        L_apply = build_laplacian_apply(
            edge_index, None, 3, "sym", self_loops=False
        )
        inv_root2 = 1.0 / 2.0 ** 0.5
        L_dense = torch.tensor(
            [
                [1.0, -inv_root2, 0.0],
                [-inv_root2, 1.0, -inv_root2],
                [0.0, -inv_root2, 1.0],
            ]
        )
        h = torch.tensor([[1.0], [2.0], [3.0]])
        assert torch.allclose(L_apply(h), L_dense @ h, atol=1e-6)

    def test_build_laplacian_apply_matches_dense_sym_self_loops(self):
        """Default ``self_loops=True`` is L̃ = I - D̂^{-1/2}(A+I)D̂^{-1/2}.

        Liao's GCN renormalization on the path 0-1-2: adding self-loops
        raises the degrees to ``D̂ = (2, 3, 2)``.
        """
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        L_apply = build_laplacian_apply(edge_index, None, 3, "sym")
        inv_root6 = 1.0 / 6.0 ** 0.5
        L_dense = torch.tensor(
            [
                [0.5, -inv_root6, 0.0],
                [-inv_root6, 2.0 / 3.0, -inv_root6],
                [0.0, -inv_root6, 0.5],
            ]
        )
        h = torch.tensor([[1.0], [2.0], [3.0]])
        assert torch.allclose(L_apply(h), L_dense @ h, atol=1e-6)
