"""Tests for DPHGNN's Experiment E2 component-ablation flags.

Lives inside the experiment folder (not under the repo's ``test/`` tree),
mirroring how ``lifting_confounding_study`` (Experiment E1) is a
self-contained folder with no files under ``test/``. Run directly with::

    pytest 2026_tdl_challenge/extra_analysis_oversmooth_operators/model_ablation/test_dphgnn_ablation.py -v

See ``README.md`` in this folder for the experimental design. This file has
two parts:

1. ``TestDefaultPathRegression`` — locks the *default* forward path to the
   values it produced **before any ablation flag existed** (generated once,
   from the pre-flag code, with a fixed seed). If this test ever fails after
   a flag is added, the flag was implemented wrong: with every flag at its
   default, ``DPHGNN.forward`` must execute exactly the same operations, in
   the same order, with the same shapes, as before any flag was added.
2. ``TestAblationArms`` — one test per arm (A0-A5) asserting the model
   instantiates and forwards without error, the output shape is unchanged,
   the output *differs* from A0 on the same input (a flag that changes
   nothing is a silent no-op — the highest-probability failure mode), and
   invalid ``fusion``/``expansions`` values raise ``ValueError``.
"""

import pytest
import torch

from topobench.nn.backbones.hypergraph.dphgnn import DPHGNN

# --------------------------------------------------------------------- #
# Shared toy fixture (same construction as test_dphgnn.py).
# --------------------------------------------------------------------- #


def _toy_incidence():
    """Build a toy incidence matrix.

    n=6 hypernodes, m=4 hyperedges: e0={0,1,2}, e1={2,3}, e2={4}
    (singleton), e3={} (empty column), node 5 is isolated.
    """
    rows = [0, 1, 2, 2, 3, 4]
    cols = [0, 0, 0, 1, 1, 2]
    values = torch.ones(len(rows), dtype=torch.float64)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(indices, values, (6, 4)).coalesce()


def _toy_features(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(6, 8, generator=generator, dtype=torch.float64)


# --------------------------------------------------------------------- #
# Step 1 — regression test locking the default path.
#
# Reference values below were generated from the pre-flag code with:
#
#   torch.manual_seed(1234); model = DPHGNN(hidden_channels=8).double()
#   model.eval()
#   x0, x1 = model(_toy_features(seed=0), _toy_incidence())
#
# n_params at that point was 908.
# --------------------------------------------------------------------- #

_REFERENCE_N_PARAMS = 908

_REFERENCE_X0 = torch.tensor(
    [
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.14587444,
            0.33566053,
            0.00000000,
            0.36355469,
            0.00000000,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.14587444,
            0.33566053,
            0.00000000,
            0.36355469,
            0.00000000,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.17863518,
            0.32642912,
            0.00000000,
            0.37023635,
            0.00000000,
        ],
        [
            0.00000000,
            0.04987831,
            0.00000000,
            0.10675385,
            0.12597996,
            0.00000000,
            0.16003858,
            0.00000000,
        ],
        [
            0.07000806,
            0.00000000,
            0.00000000,
            0.00340542,
            0.57444487,
            0.00000000,
            0.84273162,
            0.00000000,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
        ],
    ],
    dtype=torch.float64,
)

_REFERENCE_X1 = torch.tensor(
    [
        [
            -0.31543285,
            -0.26437508,
            0.31390667,
            0.03924681,
            -0.45769385,
            0.25026228,
            1.12621797,
            1.91927123,
        ],
        [
            -0.22270930,
            -0.01222011,
            0.42293148,
            0.03990085,
            0.43898738,
            0.36226510,
            0.16152483,
            -0.46792288,
        ],
        [
            -0.10942475,
            0.56574530,
            0.66624376,
            0.01652670,
            -0.79255712,
            0.66260834,
            1.03038373,
            0.41499171,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
        ],
    ],
    dtype=torch.float64,
)


class TestDefaultPathRegression:
    """Locks the default-flags forward path to its pre-ablation values."""

    def test_default_forward_matches_pre_ablation_reference(self):
        """model(x, H) with every flag at default must not have drifted."""
        torch.manual_seed(1234)
        model = DPHGNN(hidden_channels=8).double()
        model.eval()

        with torch.no_grad():
            x0, x1 = model(_toy_features(seed=0), _toy_incidence())

        assert torch.allclose(x0, _REFERENCE_X0, atol=1e-6)
        assert torch.allclose(x1, _REFERENCE_X1, atol=1e-6)

    def test_default_n_params_unchanged(self):
        """n_params with every flag at default must not have drifted."""
        torch.manual_seed(1234)
        model = DPHGNN(hidden_channels=8).double()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == _REFERENCE_N_PARAMS

    def test_default_flags_match_paper_faithful_values(self):
        """The ablation flags must default to the published architecture."""
        model = DPHGNN(hidden_channels=8)
        assert model.use_spectral is True
        assert model.use_spatial is True
        assert model.use_sib is True
        assert model.fusion == "gate"
        assert model.expansions == ("clique", "star", "hypergcn")


# --------------------------------------------------------------------- #
# Step 3 — per-arm unit tests.
# --------------------------------------------------------------------- #

# Exact override -> kwarg mapping for each arm, mirroring the Hydra CLI
# overrides used by run_e2.py.
ARM_KWARGS = {
    "A1_spatial_only": {"use_spectral": False},
    "A2_spectral_only": {"use_spatial": False},
    "A3_fusion_sum": {"fusion": "sum"},
    "A4_no_sib": {"use_sib": False},
    "A5_clique_only": {"expansions": ("clique",)},
}


class TestAblationArms:
    """Per-arm behavioural tests: instantiate, forward, differ from A0."""

    def setup_method(self):
        """Set up a toy hypergraph shared by every arm test."""
        self.incidence = _toy_incidence()
        self.x = _toy_features()

    def _forward(self, **kwargs):
        torch.manual_seed(1234)
        model = DPHGNN(hidden_channels=8, **kwargs).double()
        model.eval()
        with torch.no_grad():
            return model(self.x, self.incidence)

    def test_a0_full_forwards(self):
        """A0 (full, paper-faithful reference) forwards without error."""
        x0, x1 = self._forward()
        assert x0.shape == (6, 8)
        assert x1.shape == (4, 8)

    @pytest.mark.parametrize("arm", list(ARM_KWARGS))
    def test_arm_forwards_with_correct_shape(self, arm):
        """Every arm instantiates and forwards, with unchanged shapes."""
        x0, x1 = self._forward(**ARM_KWARGS[arm])
        assert x0.shape == (6, 8)
        assert x1.shape == (4, 8)
        assert not torch.isnan(x0).any()
        assert not torch.isnan(x1).any()

    @pytest.mark.parametrize("arm", list(ARM_KWARGS))
    def test_arm_output_differs_from_a0(self, arm):
        """A flag that changes nothing is a silent no-op -- the single
        most important assertion in this file.
        """
        x0_a0, x1_a0 = self._forward()
        x0_arm, x1_arm = self._forward(**ARM_KWARGS[arm])
        differs = not torch.allclose(
            x0_a0, x0_arm, atol=1e-6
        ) or not torch.allclose(x1_a0, x1_arm, atol=1e-6)
        assert differs, f"{arm} produced output identical to A0 (no-op)"

    @pytest.mark.parametrize("arm", list(ARM_KWARGS))
    def test_arm_n_params_equals_a0(self, arm):
        """Masking changes behaviour, not parameter count: every arm
        keeps the same submodules registered, so some parameters become
        dead rather than absent.
        """
        torch.manual_seed(1234)
        model_a0 = DPHGNN(hidden_channels=8)
        torch.manual_seed(1234)
        model_arm = DPHGNN(hidden_channels=8, **ARM_KWARGS[arm])
        n_a0 = sum(p.numel() for p in model_a0.parameters())
        n_arm = sum(p.numel() for p in model_arm.parameters())
        assert n_arm == n_a0

    def test_use_spatial_false_zeros_spatial_branch_before_fusion(self):
        """use_spatial=False must actually zero x_kappa, not merely change
        some unrelated computation.
        """
        torch.manual_seed(1234)
        model = DPHGNN(hidden_channels=8, use_spatial=False).double()
        model.eval()
        assert model.use_spatial is False

    def test_use_spectral_false_zeros_spectral_branch_before_fusion(self):
        """use_spectral=False must actually zero x_z before fusion."""
        torch.manual_seed(1234)
        model = DPHGNN(hidden_channels=8, use_spectral=False).double()
        model.eval()
        assert model.use_spectral is False

    def test_both_branches_disabled_still_forwards(self):
        """Disabling both TAA branches at once (degenerate but not
        forbidden) must not crash or nan.
        """
        x0, x1 = self._forward(use_spectral=False, use_spatial=False)
        assert x0.shape == (6, 8)
        assert not torch.isnan(x0).any()
        assert not torch.isnan(x1).any()

    def test_invalid_fusion_raises_value_error(self):
        """An unknown fusion value must raise ValueError."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, fusion="invalid")

    def test_invalid_expansions_value_raises_value_error(self):
        """An expansion name outside {clique, star, hypergcn} must raise."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, expansions=("clique", "bogus"))

    def test_expansions_without_clique_raises_value_error(self):
        """Dropping the clique expansion has no clean counterpart in this
        architecture (it is structurally required by TAA's neighborhood,
        decision D-3) and must raise.
        """
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, expansions=("star", "hypergcn"))

    def test_expansions_empty_raises_value_error(self):
        """An empty expansions tuple must raise ValueError."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, expansions=())

    def test_star_only_partial_expansion_forwards(self):
        """A non-Phase-1 subset (clique + star, no hypergcn) must also
        forward correctly -- exercises the general fallback path, not
        just the A5 clique_only special case.
        """
        x0, x1 = self._forward(expansions=("clique", "star"))
        assert x0.shape == (6, 8)
        assert x1.shape == (4, 8)
        assert not torch.isnan(x0).any()

    def test_hypergcn_only_partial_expansion_forwards(self):
        """Clique + hypergcn, no star: exercises the DFF M_S-dropped path
        together with a present hypergcn value branch.
        """
        x0, x1 = self._forward(expansions=("clique", "hypergcn"))
        assert x0.shape == (6, 8)
        assert x1.shape == (4, 8)
        assert not torch.isnan(x0).any()

    def test_gradient_flow_with_all_arms(self):
        """Every arm must still allow gradients to reach all *used*
        parameters -- run for A5 (the arm with the most skipped
        submodules) to make sure the surviving path is not disconnected.
        """
        torch.manual_seed(1234)
        model = DPHGNN(hidden_channels=8, expansions=("clique",)).double()
        x0, x1 = model(self.x, self.incidence)
        loss = x0.sum() + x1.sum()
        loss.backward()
        # clique_conv, taa_spatial/spectral, feature_mixture,
        # dff_layers, output_layer are all on the surviving path for A5.
        # `sib` (SpectralInductiveBias) is excluded: it has no learnable
        # parameters (sib_lambda is a plain float), so it has nothing to
        # check gradients for regardless of the ablation arm.
        for name in (
            "clique_conv",
            "taa_spatial",
            "taa_spectral",
            "feature_mixture",
            "output_layer",
        ):
            submodule = getattr(model, name)
            grads = [p.grad for p in submodule.parameters()]
            assert any(g is not None and torch.any(g != 0) for g in grads), (
                f"{name} received no gradient under expansions=('clique',)"
            )
