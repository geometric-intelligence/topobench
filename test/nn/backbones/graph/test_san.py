"""Unit tests for the Sheaf Attention Network (SAN) backbone."""

import pytest
import torch

from topobench.nn.backbones.graph.nsd_utils.adjacency_builders import (
    DiagSheafAdjacencyBuilder,
    GeneralSheafAdjacencyBuilder,
    NormConnectionSheafAdjacencyBuilder,
)
from topobench.nn.backbones.graph.nsd_utils.inductive_attention_models import (
    InductiveSheafAttentionBundle,
    InductiveSheafAttentionDiag,
    InductiveSheafAttentionGeneral,
    _augment_with_self_loops,
)
from topobench.nn.backbones.graph.nsd_utils.san_attention import (
    SheafGATAttention,
)
from topobench.nn.backbones.graph.san import SANEncoder


def _make_undirected_edge_index(pairs):
    """Build a bidirectional edge_index from a list of unordered pairs.

    Parameters
    ----------
    pairs : list of tuple of int
        Unordered node-index pairs ``(i, j)`` with ``i != j``.

    Returns
    -------
    torch.Tensor
        Directed edge index of shape [2, 2 * len(pairs)].
    """
    src, tgt = [], []
    for i, j in pairs:
        src.extend([i, j])
        tgt.extend([j, i])
    return torch.tensor([src, tgt], dtype=torch.long)


@pytest.fixture
def small_graph():
    """Tiny undirected graph for forward-pass exercises.

    Returns
    -------
    tuple
        ``(x, edge_index, num_nodes)`` test artifacts.
    """
    edge_index = _make_undirected_edge_index(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    )
    x = torch.randn(4, 8)
    return x, edge_index, 4


@pytest.fixture
def random_graph():
    """Slightly larger random graph for stress tests.

    Returns
    -------
    tuple
        ``(x, edge_index, num_nodes)`` artifacts.
    """
    torch.manual_seed(7)
    n = 12
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (0, 11),
             (0, 5), (1, 6), (2, 8)]
    edge_index = _make_undirected_edge_index(pairs)
    x = torch.randn(n, 16)
    return x, edge_index, n


class TestSANEncoderInit:
    """Validate ``SANEncoder`` initialization paths."""

    def test_default_init(self):
        """Default kwargs produce a bundle SAN with d=2."""
        model = SANEncoder(input_dim=8, hidden_dim=16)
        assert model.sheaf_type == "bundle"
        assert model.d == 2
        assert model.num_layers == 2
        assert isinstance(model.san_model, InductiveSheafAttentionBundle)

    def test_diag_init(self):
        """Diag sheaf type instantiates ``InductiveSheafAttentionDiag``."""
        model = SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="diag", d=3
        )
        assert model.sheaf_type == "diag"
        assert isinstance(model.san_model, InductiveSheafAttentionDiag)

    def test_general_init(self):
        """General sheaf type instantiates ``InductiveSheafAttentionGeneral``."""
        model = SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="general", d=2
        )
        assert isinstance(model.san_model, InductiveSheafAttentionGeneral)

    def test_invalid_sheaf_type(self):
        """An unknown ``sheaf_type`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unknown sheaf type"):
            SANEncoder(input_dim=8, hidden_dim=12, sheaf_type="weird")

    def test_diag_requires_positive_d(self):
        """Diag variant rejects ``d=0``."""
        with pytest.raises(AssertionError):
            SANEncoder(input_dim=8, hidden_dim=12, sheaf_type="diag", d=0)

    def test_bundle_requires_d_gt_1(self):
        """Bundle variant rejects ``d=1``."""
        with pytest.raises(AssertionError):
            SANEncoder(input_dim=8, hidden_dim=12, sheaf_type="bundle", d=1)

    def test_general_requires_d_gt_1(self):
        """General variant rejects ``d=1``."""
        with pytest.raises(AssertionError):
            SANEncoder(
                input_dim=8, hidden_dim=12, sheaf_type="general", d=1
            )

    def test_bundle_silent_channel_truncation(self):
        """Bundle silently truncates hidden_channels = hidden_dim // d.

        Mirrors NSD's permissive contract: when ``hidden_dim`` is not
        divisible by ``d`` the inner model uses ``hidden_channels * d``
        internally; the outer projection still produces an output of
        size ``hidden_dim``.
        """
        model = SANEncoder(
            input_dim=8, hidden_dim=10, sheaf_type="bundle", d=3,
        )
        # Internal channels rounded down.
        assert model.san_model.hidden_channels == 10 // 3
        assert model.san_model.hidden_dim == (10 // 3) * 3
        # Outer projection still emits the requested dim.
        x = torch.randn(4, 8)
        edge_index = _make_undirected_edge_index([(0, 1), (1, 2), (2, 3)])
        out = model(x, edge_index)
        assert out.shape == (4, 10)

    def test_residual_flag_propagated(self):
        """The ``residual`` kwarg flows into the inductive model."""
        model = SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="bundle", d=2,
            residual=True,
        )
        assert model.san_model.residual is True

    def test_num_heads_flag_propagated(self):
        """The ``num_heads`` kwarg flows into the attention modules."""
        model = SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="diag", d=3,
            num_heads=4,
        )
        assert model.san_model.num_heads == 4
        for attn in model.san_model.sheaf_attentions:
            assert attn.num_heads == 4

    def test_get_sheaf_model(self):
        """``get_sheaf_model`` returns the inner inductive model."""
        model = SANEncoder(input_dim=8, hidden_dim=12, sheaf_type="diag")
        assert model.get_sheaf_model() is model.san_model

    @pytest.mark.parametrize("act", ["tanh", "elu", "id"])
    def test_sheaf_act_options(self, act):
        """All advertised sheaf activations construct without error.

        Parameters
        ----------
        act : str
            Sheaf activation name passed to ``SANEncoder``.
        """
        SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="diag", d=2,
            sheaf_act=act,
        )

    @pytest.mark.parametrize("orth", ["cayley", "matrix_exp"])
    def test_orth_options(self, orth):
        """Both orthogonalization methods construct without error.

        Parameters
        ----------
        orth : str
            Orthogonalization method passed to ``SANEncoder``.
        """
        SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="bundle", d=2,
            orth=orth,
        )

    def test_kwargs_are_ignored(self):
        """Extra kwargs are accepted silently for Hydra flexibility."""
        SANEncoder(
            input_dim=8, hidden_dim=12, sheaf_type="diag", d=2,
            spurious_unused_arg="ok",
        )


class TestSANEncoderForward:
    """Forward-pass behaviour of ``SANEncoder``."""

    @pytest.mark.parametrize(
        "sheaf_type,d",
        [("diag", 1), ("diag", 2), ("diag", 4),
         ("bundle", 2), ("bundle", 4),
         ("general", 2), ("general", 3)],
    )
    @pytest.mark.parametrize("residual", [False, True])
    @pytest.mark.parametrize("num_heads", [1, 2])
    def test_forward_shape(
        self, small_graph, sheaf_type, d, residual, num_heads
    ):
        """Output shape matches ``[num_nodes, hidden_dim]`` everywhere.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        sheaf_type : str
            Restriction-map family.
        d : int
            Stalk dimension.
        residual : bool
            Whether to use the Res-SheafAN update.
        num_heads : int
            Number of attention heads.
        """
        x, edge_index, n = small_graph
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=12,
            sheaf_type=sheaf_type, d=d,
            dropout=0.0, input_dropout=0.0,
            num_heads=num_heads, residual=residual,
        )
        out = model(x, edge_index)
        assert out.shape == (n, 12)
        assert torch.all(torch.isfinite(out))

    def test_forward_accepts_single_direction(self, small_graph):
        """The encoder symmetrizes single-direction edges internally.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = small_graph
        keep = edge_index[0] < edge_index[1]
        directed = edge_index[:, keep]
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=12,
            sheaf_type="bundle", d=2,
            dropout=0.0, input_dropout=0.0,
        )
        out = model(x, directed)
        assert out.shape == (n, 12)

    def test_gradient_flow(self, small_graph):
        """A summed-loss backward populates parameter grads.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, _ = small_graph
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=12,
            sheaf_type="general", d=2,
            dropout=0.0, input_dropout=0.0, num_heads=2,
        )
        out = model(x, edge_index)
        out.sum().backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.all(torch.isfinite(p.grad))

    def test_eval_mode_is_deterministic(self, small_graph):
        """With ``eval()``, dropout is disabled and outputs are stable.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, _ = small_graph
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=12,
            sheaf_type="bundle", d=2,
            dropout=0.5, input_dropout=0.5,
        )
        model.eval()
        a = model(x, edge_index)
        b = model(x, edge_index)
        torch.testing.assert_close(a, b)

    def test_train_mode_dropout_changes_output(self, small_graph):
        """Dropout in training mode produces different outputs.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, _ = small_graph
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=12,
            sheaf_type="bundle", d=2,
            dropout=0.5, input_dropout=0.5,
        )
        model.train()
        torch.manual_seed(0)
        a = model(x, edge_index)
        torch.manual_seed(1)
        b = model(x, edge_index)
        assert not torch.allclose(a, b)

    @pytest.mark.parametrize("num_layers", [1, 2, 4])
    def test_layer_depth(self, small_graph, num_layers):
        """Forward pass succeeds across layer counts.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        num_layers : int
            Number of SheafAN layers.
        """
        x, edge_index, n = small_graph
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=12,
            sheaf_type="diag", d=3,
            num_layers=num_layers,
            dropout=0.0, input_dropout=0.0,
        )
        out = model(x, edge_index)
        assert out.shape == (n, 12)

    def test_random_graph_smoke(self, random_graph):
        """Exercise a randomly-built graph with multi-head + residual.

        Parameters
        ----------
        random_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = random_graph
        model = SANEncoder(
            input_dim=x.size(1), hidden_dim=24,
            sheaf_type="bundle", d=3,
            dropout=0.1, input_dropout=0.1,
            num_heads=4, residual=True,
            orth="matrix_exp",
        )
        out = model(x, edge_index)
        assert out.shape == (n, 24)


class TestSheafGATAttention:
    """Behaviour of the multi-head GAT-style attention module."""

    @pytest.mark.parametrize("variant", ["gat", "gatv2"])
    def test_row_stochastic_over_source(self, small_graph, variant):
        """Attention coefficients sum to one per source node.

        This is the defining property of the attention matrix Lambda in
        equation (2) of Barbero et al. (2022): the softmax runs over
        ``k in N_i``, so every row is a probability mass function.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        variant : str
            Attention scoring function under test.
        """
        x, edge_index, n = small_graph
        loop = torch.arange(n).unsqueeze(0).expand(2, -1)
        aug = torch.cat([edge_index, loop], dim=1)

        attn = SheafGATAttention(
            in_channels=x.size(1), num_heads=2, attention_variant=variant
        )
        with torch.no_grad():
            alpha = attn(x, aug)
        src = aug[0]
        sums = torch.zeros(n)
        sums.scatter_add_(0, src, alpha)
        torch.testing.assert_close(sums, torch.ones(n), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("variant", ["gat", "gatv2"])
    def test_head_dim_override(self, variant):
        """Explicit ``head_dim`` decouples projection size from in_channels.

        Parameters
        ----------
        variant : str
            Attention scoring function under test.
        """
        attn = SheafGATAttention(
            in_channels=8, num_heads=2, head_dim=5, attention_variant=variant
        )
        assert attn.head_dim == 5
        assert attn.lin_src.out_features == 10

    @pytest.mark.parametrize("variant", ["gat", "gatv2"])
    def test_non_divisible_head_split_uses_floor(self, variant):
        """``head_dim`` defaults to floor(in_channels / num_heads).

        Mirrors PyG's ``GATConv`` behaviour: the input channel count need
        not be divisible by the number of heads.

        Parameters
        ----------
        variant : str
            Attention scoring function under test.
        """
        attn = SheafGATAttention(
            in_channels=7, num_heads=2, attention_variant=variant
        )
        assert attn.head_dim == 3
        assert attn.lin_src.out_features == 6

    def test_zero_heads_rejected(self):
        """``num_heads`` must be at least 1."""
        with pytest.raises(AssertionError):
            SheafGATAttention(in_channels=4, num_heads=0)

    def test_unknown_variant_rejected(self):
        """Only ``'gat'`` and ``'gatv2'`` are accepted."""
        with pytest.raises(ValueError, match="Unknown attention variant"):
            SheafGATAttention(in_channels=4, attention_variant="gatv3")

    @pytest.mark.parametrize("variant", ["gat", "gatv2"])
    def test_reset_parameters(self, variant):
        """Reset is idempotent on shape and produces finite values.

        Parameters
        ----------
        variant : str
            Attention scoring function under test.
        """
        attn = SheafGATAttention(
            in_channels=8, num_heads=2, attention_variant=variant
        )
        before = attn.lin_src.weight.detach().clone()
        attn.reset_parameters()
        after = attn.lin_src.weight.detach().clone()
        assert before.shape == after.shape
        assert torch.all(torch.isfinite(after))

    def test_variant_parameter_sets_differ(self):
        """GAT shares one W and two score halves; GATv2 does the reverse.

        Equation (2) of the paper scores ``LeakyReLU(a [W x_i || W x_j])``
        with a single shared ``W``, whereas GATv2 (dissertation section
        3.1) scores ``a^T LeakyReLU(W [x_i || x_j])`` and therefore needs
        a separate target projection.
        """
        gat = SheafGATAttention(in_channels=8, attention_variant="gat")
        assert gat.lin_tgt is None and gat.att_tgt is not None

        gatv2 = SheafGATAttention(in_channels=8, attention_variant="gatv2")
        assert gatv2.lin_tgt is not None and gatv2.att_tgt is None

    def test_variants_produce_different_scores(self, small_graph):
        """The two scoring functions are genuinely different operators.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = small_graph
        loop = torch.arange(n).unsqueeze(0).expand(2, -1)
        aug = torch.cat([edge_index, loop], dim=1)

        torch.manual_seed(0)
        gat = SheafGATAttention(in_channels=x.size(1), attention_variant="gat")
        torch.manual_seed(0)
        gatv2 = SheafGATAttention(
            in_channels=x.size(1), attention_variant="gatv2"
        )
        with torch.no_grad():
            assert not torch.allclose(gat(x, aug), gatv2(x, aug))


class TestSelfLoopAugmentation:
    """Behaviour of ``_augment_with_self_loops``."""

    def test_appends_n_loops(self, small_graph):
        """The augmented index gains exactly ``num_nodes`` columns.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        aug = _augment_with_self_loops(edge_index, n)
        assert aug.size(1) == edge_index.size(1) + n
        loop_cols = aug[:, edge_index.size(1):]
        assert torch.equal(loop_cols[0], loop_cols[1])
        assert torch.equal(loop_cols[0], torch.arange(n))


class TestAdjacencyBuilders:
    """End-to-end shape checks for the three adjacency builders."""

    @staticmethod
    def _alpha(num_edges, num_nodes):
        """Make a normalized attention vector of size ``num_edges + num_nodes``.

        Parameters
        ----------
        num_edges : int
            Number of directed edges.
        num_nodes : int
            Number of nodes; trailing entries play the role of
            self-loop attention.

        Returns
        -------
        torch.Tensor
            Softmaxed random vector of shape ``[num_edges + num_nodes]``.
        """
        rng = torch.Generator().manual_seed(123)
        return torch.softmax(
            torch.randn(num_edges + num_nodes, generator=rng), dim=0
        )

    def test_diag_builder_shapes(self, small_graph):
        """Diag builder returns a sparse adjacency over ``N*d`` rows.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        d = 3
        builder = DiagSheafAdjacencyBuilder(n, edge_index, d=d)
        maps = torch.randn(edge_index.size(1), d)
        alpha = self._alpha(edge_index.size(1), n)
        (idx, vals), saved = builder(maps, alpha)
        assert idx.size(0) == 2
        assert vals.numel() == idx.size(1)
        assert idx.max() < n * d
        assert saved.shape[0] == edge_index.size(1) // 2

    def test_bundle_builder_shapes(self, small_graph):
        """Bundle builder accepts orthogonal map parameters.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        d = 2
        builder = NormConnectionSheafAdjacencyBuilder(
            n, edge_index, d=d, orth_map="cayley",
        )
        maps = torch.randn(edge_index.size(1), d * (d + 1) // 2)
        alpha = self._alpha(edge_index.size(1), n)
        (idx, vals), saved = builder(maps, alpha)
        assert idx.size(0) == 2
        assert idx.max() < n * d
        assert saved.shape == (edge_index.size(1) // 2, d, d)

    def test_general_builder_shapes(self, small_graph):
        """General builder accepts full d x d maps.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        d = 2
        builder = GeneralSheafAdjacencyBuilder(n, edge_index, d=d)
        maps = torch.randn(edge_index.size(1), d, d)
        alpha = self._alpha(edge_index.size(1), n)
        (idx, vals), saved = builder(maps, alpha)
        assert idx.size(0) == 2
        assert idx.max() < n * d
        assert saved.shape == (edge_index.size(1) // 2, d, d)

    def test_diag_builder_rejects_wrong_shape(self, small_graph):
        """Diag builder asserts on map shape mismatch.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        builder = DiagSheafAdjacencyBuilder(n, edge_index, d=3)
        bad = torch.randn(edge_index.size(1), 2)
        with pytest.raises(AssertionError):
            builder(bad, self._alpha(edge_index.size(1), n))

    def test_bundle_builder_rejects_wrong_shape(self, small_graph):
        """Bundle builder asserts on parameter dim mismatch.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        builder = NormConnectionSheafAdjacencyBuilder(
            n, edge_index, d=3, orth_map="cayley",
        )
        bad = torch.randn(edge_index.size(1), 2)
        with pytest.raises(AssertionError):
            builder(bad, self._alpha(edge_index.size(1), n))

    def test_general_builder_rejects_wrong_shape(self, small_graph):
        """General builder asserts on full-matrix shape mismatch.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        builder = GeneralSheafAdjacencyBuilder(n, edge_index, d=3)
        bad = torch.randn(edge_index.size(1), 2, 2)
        with pytest.raises(AssertionError):
            builder(bad, self._alpha(edge_index.size(1), n))


class TestInductiveModels:
    """Direct exercises on the inductive SheafAN model classes."""

    def _config(self, **overrides):
        """Minimal config dict acceptable to ``SheafDiffusion``.

        Parameters
        ----------
        **overrides : dict
            Keys to override in the default config.

        Returns
        -------
        dict
            Config suitable for passing to an ``Inductive`` constructor.
        """
        cfg = {
            "d": 2,
            "layers": 2,
            "hidden_channels": 6,
            "input_dim": 8,
            "output_dim": 12,
            "device": "cpu",
            "input_dropout": 0.0,
            "dropout": 0.0,
            "sheaf_act": "tanh",
            "orth": "cayley",
            "num_heads": 1,
            "residual": False,
        }
        cfg.update(overrides)
        return cfg

    def test_diag_forward(self, small_graph):
        """Diag inductive model returns expected output shape.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = small_graph
        cfg = self._config(d=3, hidden_channels=4)
        model = InductiveSheafAttentionDiag(cfg)
        out = model(x, edge_index)
        assert out.shape == (n, 12)

    def test_bundle_forward_residual(self, small_graph):
        """Bundle inductive model with residual update.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = small_graph
        cfg = self._config(d=2, hidden_channels=6, residual=True)
        model = InductiveSheafAttentionBundle(cfg)
        out = model(x, edge_index)
        assert out.shape == (n, 12)

    def test_general_forward_multi_head(self, small_graph):
        """General inductive model with multi-head attention.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = small_graph
        cfg = self._config(d=2, hidden_channels=6, num_heads=4)
        model = InductiveSheafAttentionGeneral(cfg)
        out = model(x, edge_index)
        assert out.shape == (n, 12)

    def test_residual_changes_output(self, small_graph):
        """Toggling ``residual`` produces a different output.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, _ = small_graph
        torch.manual_seed(2)
        m1 = InductiveSheafAttentionBundle(self._config(residual=False))
        torch.manual_seed(2)
        m2 = InductiveSheafAttentionBundle(self._config(residual=True))
        m1.eval()
        m2.eval()
        a = m1(x, edge_index)
        b = m2(x, edge_index)
        assert not torch.allclose(a, b)

    def test_sheaf_learner_stores_L(self, small_graph):
        """Forward pass populates ``sheaf_learner.L`` for analysis.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, _ = small_graph
        cfg = self._config()
        model = InductiveSheafAttentionBundle(cfg)
        model(x, edge_index)
        for sl in model.sheaf_learners:
            assert sl.L is not None


class TestPaperProperties:
    """Assert the defining claims of Barbero et al. (2022) hold.

    These tests pin the mathematical structure the paper specifies,
    rather than the incidental shapes of this implementation, so that a
    refactor that silently breaks faithfulness fails loudly.
    """

    @staticmethod
    def _dense_adjacency(builder, maps, alpha, n, d):
        """Materialize a builder's sparse output as a dense matrix.

        Parameters
        ----------
        builder : nn.Module
            One of the sheaf adjacency builders.
        maps : torch.Tensor
            Restriction map parameters for the builder.
        alpha : torch.Tensor
            Attention coefficients over the augmented edge index.
        n : int
            Number of nodes.
        d : int
            Stalk dimension.

        Returns
        -------
        torch.Tensor
            Dense adjacency of shape [n * d, n * d].
        """
        (idx, val), _ = builder(maps, alpha)
        return torch.sparse_coo_tensor(idx, val, (n * d, n * d)).to_dense()

    @staticmethod
    def _row_stochastic_alpha(edge_index, n, seed=0):
        """Build a valid row-stochastic attention vector.

        Parameters
        ----------
        edge_index : torch.Tensor
            Directed edge index without self-loops.
        n : int
            Number of nodes.
        seed : int, optional
            Seed for the random scores. Default is 0.

        Returns
        -------
        torch.Tensor
            Attention values over ``[edges | self-loops]``.
        """
        from torch_scatter import scatter_softmax

        torch.manual_seed(seed)
        src = torch.cat([edge_index[0], torch.arange(n)])
        raw = torch.randn(edge_index.size(1) + n)
        return scatter_softmax(raw, src, dim=0)

    def test_restriction_maps_are_orthogonal(self):
        """Bundle restriction maps satisfy ``F^T F = I``.

        The paper states "for our purposes, we use orthogonal restriction
        maps, i.e. F_{v<|e} in O(d)".
        """
        from topobench.nn.backbones.graph.nsd_utils.orthogonal import (
            Orthogonal,
        )

        d = 4
        torch.manual_seed(0)
        for orth_map in ("cayley", "matrix_exp"):
            transform = Orthogonal(d=d, orthogonal_map=orth_map)
            maps = transform(torch.randn(10, d * (d + 1) // 2))
            gram = maps.transpose(-1, -2) @ maps
            torch.testing.assert_close(
                gram,
                torch.eye(d).expand_as(gram),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_adjacency_block_rows_are_row_stochastic(self, small_graph):
        """Block rows of ``Lambda_hat * A_F`` sum to one.

        Equation (3) broadcasts the row-stochastic Lambda over d x d
        blocks. With orthogonal transport blocks the spectral norm of
        block ``(i, j)`` is exactly ``alpha_ij``, so each block row must
        sum to one.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        d = 4
        alpha = self._row_stochastic_alpha(edge_index, n)
        builder = NormConnectionSheafAdjacencyBuilder(
            n, edge_index, d=d, orth_map="cayley"
        )
        torch.manual_seed(1)
        params = torch.randn(edge_index.size(1), d * (d + 1) // 2)
        adjacency = self._dense_adjacency(builder, params, alpha, n, d)

        for i in range(n):
            total = 0.0
            for j in range(n):
                block = adjacency[i * d:(i + 1) * d, j * d:(j + 1) * d]
                if block.abs().max() > 0:
                    total += torch.linalg.matrix_norm(block, ord=2).item()
            assert abs(total - 1.0) < 1e-4

    def test_diagonal_blocks_are_scaled_identity(self, small_graph):
        """Self-loop blocks equal ``alpha_ii * I_d``.

        The sheaf adjacency has added self-loops with
        ``A_F(i, i) = F_i^T F_i``, which is the identity for orthogonal
        restriction maps.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        d = 4
        num_edges = edge_index.size(1)
        alpha = self._row_stochastic_alpha(edge_index, n)
        builder = NormConnectionSheafAdjacencyBuilder(
            n, edge_index, d=d, orth_map="cayley"
        )
        torch.manual_seed(1)
        params = torch.randn(num_edges, d * (d + 1) // 2)
        adjacency = self._dense_adjacency(builder, params, alpha, n, d)

        for i in range(n):
            block = adjacency[i * d:(i + 1) * d, i * d:(i + 1) * d]
            torch.testing.assert_close(
                block,
                alpha[num_edges + i] * torch.eye(d),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_transport_blocks_are_transposes(self, small_graph):
        """``P_ji`` equals ``P_ij^T`` once attention scaling is removed.

        The sheaf adjacency is built from ``P_ij = F_i^T F_j``, so the
        reverse block must be its transpose; only the attention
        coefficient differs by direction.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        _, edge_index, n = small_graph
        d = 4
        alpha = self._row_stochastic_alpha(edge_index, n)
        builder = NormConnectionSheafAdjacencyBuilder(
            n, edge_index, d=d, orth_map="cayley"
        )
        torch.manual_seed(1)
        params = torch.randn(edge_index.size(1), d * (d + 1) // 2)
        adjacency = self._dense_adjacency(builder, params, alpha, n, d)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                forward = adjacency[i * d:(i + 1) * d, j * d:(j + 1) * d]
                if forward.abs().max() == 0:
                    continue
                reverse = adjacency[j * d:(j + 1) * d, i * d:(i + 1) * d]
                scale_f = torch.linalg.matrix_norm(forward, ord=2)
                scale_r = torch.linalg.matrix_norm(reverse, ord=2)
                torch.testing.assert_close(
                    forward / scale_f,
                    (reverse / scale_r).T,
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_residual_update_matches_equation_six(self, small_graph):
        """Res-SheafAN adds ``sigma((A - I) (I kron W1) X W2)`` to ``X``.

        Checks equation (6) against a dense reference built from the same
        adjacency, rather than merely asserting that the residual flag
        changes the output.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        import torch.nn.functional as F

        x, edge_index, n = small_graph
        d, hidden_channels = 2, 4
        config = {
            "d": d,
            "layers": 1,
            "hidden_channels": hidden_channels,
            "input_dim": x.size(1),
            "output_dim": 6,
            "device": "cpu",
            "input_dropout": 0.0,
            "dropout": 0.0,
            "sheaf_act": "tanh",
            "orth": "cayley",
            "num_heads": 1,
            "residual": True,
        }
        torch.manual_seed(3)
        model = InductiveSheafAttentionBundle(config).eval()

        # Recompute the single layer by hand from the model's own pieces.
        with torch.no_grad():
            h = model.lin1(x)
            h = F.elu(h)
            h = h.view(n * d, -1)

            builder = model._build_adjacency(n, edge_index)
            aug = _augment_with_self_loops(edge_index, n)
            maps = model.sheaf_learners[0](h.reshape(n, -1), edge_index)
            alpha = model.sheaf_attentions[0](h.reshape(n, -1), aug)
            (idx, val), _ = builder(maps, alpha)
            adjacency = torch.sparse_coo_tensor(
                idx, val, (n * d, n * d)
            ).to_dense()

            transformed = model.left_right_linear(
                h, model.lin_left_weights[0], model.lin_right_weights[0], n
            )
            expected = h + F.elu(adjacency @ transformed - transformed)
            expected = model.lin2(expected.reshape(n, -1))

            actual = model(x, edge_index)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
