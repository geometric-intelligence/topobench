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

    def test_row_stochastic_over_source(self, small_graph):
        """Attention coefficients sum to one per source node.

        Parameters
        ----------
        small_graph : tuple
            Fixture tuple ``(x, edge_index, num_nodes)``.
        """
        x, edge_index, n = small_graph
        loop = torch.arange(n).unsqueeze(0).expand(2, -1)
        aug = torch.cat([edge_index, loop], dim=1)

        attn = SheafGATAttention(in_channels=x.size(1), num_heads=2)
        with torch.no_grad():
            alpha = attn(x, aug)
        src = aug[0]
        sums = torch.zeros(n)
        sums.scatter_add_(0, src, alpha)
        torch.testing.assert_close(sums, torch.ones(n), atol=1e-5, rtol=1e-5)

    def test_head_dim_override(self):
        """Explicit ``head_dim`` decouples projection size from in_channels."""
        attn = SheafGATAttention(in_channels=8, num_heads=2, head_dim=5)
        assert attn.head_dim == 5
        assert attn.lin.out_features == 10

    def test_non_divisible_head_split_uses_floor(self):
        """``head_dim`` defaults to floor(in_channels / num_heads).

        Mirrors PyG's ``GATConv`` behaviour: the input channel count need
        not be divisible by the number of heads.
        """
        attn = SheafGATAttention(in_channels=7, num_heads=2)
        assert attn.head_dim == 3
        assert attn.lin.out_features == 6

    def test_zero_heads_rejected(self):
        """``num_heads`` must be at least 1."""
        with pytest.raises(AssertionError):
            SheafGATAttention(in_channels=4, num_heads=0)

    def test_reset_parameters(self):
        """Reset is idempotent on shape and produces finite values."""
        attn = SheafGATAttention(in_channels=8, num_heads=2)
        before = attn.lin.weight.detach().clone()
        attn.reset_parameters()
        after = attn.lin.weight.detach().clone()
        assert before.shape == after.shape
        assert torch.all(torch.isfinite(after))


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
        m1.eval(); m2.eval()
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
