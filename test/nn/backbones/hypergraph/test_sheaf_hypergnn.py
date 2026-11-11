"""Unit tests for SheafHyperGNN."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.nn.backbones.hypergraph.sheaf_hypergnn import (
    _MLP,
    SheafHyperGNN,
    _DiagonalSheafBuilder,
    _DiagonalSheafConv,
    _expand_diagonal,
    _incidence_to_hyperedge_index,
)
from topobench.nn.readouts import NoReadOut
from topobench.nn.wrappers import HypergraphWrapper

### Helpers


def _make_incidence(num_nodes=8, num_edges=5, seed=0):
    """Return reproducible sparse COO incidence with every hyperedge represented."""
    generator = torch.Generator().manual_seed(seed)
    guaranteed_edges = torch.arange(num_edges)
    guaranteed_nodes = guaranteed_edges.remainder(num_nodes)
    random_nodes = torch.randint(
        0,
        num_nodes,
        (num_nodes * 2,),
        generator=generator,
    )
    random_edges = torch.randint(
        0,
        num_edges,
        (num_nodes * 2,),
        generator=generator,
    )
    v = torch.cat((guaranteed_nodes, random_nodes))
    e = torch.cat((guaranteed_edges, random_edges))
    indices = torch.stack([v, e])
    values = torch.ones(indices.shape[1])
    inc = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_edges)
    ).coalesce()
    return inc


class _CaptureProjection(nn.Module):
    """Record projection inputs while returning fixed-width zeros."""

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.last_input = None

    def forward(self, x):
        """Record and replace the input."""
        self.last_input = x.detach().clone()
        return x.new_zeros((x.shape[0], self.out_channels))


def _dense_sheaf_conv_reference(conv, x, h_idx, h_val, num_nodes, num_edges):
    """Build the diagonal sheaf operator densely from its matrix definition."""
    x = conv.lin(x)
    node_idx, edge_idx = h_idx
    stalk_dim = conv.d
    node_size = num_nodes * stalk_dim
    edge_size = num_edges * stalk_dim

    h_dense = x.new_zeros((node_size, edge_size))
    h_dense[node_idx, edge_idx] = h_val

    # B counts structural incidences, not restriction-map magnitudes.
    incidence_mask = x.new_zeros((node_size, edge_size))
    incidence_mask[node_idx, edge_idx] = 1
    edge_degree = incidence_mask.sum(dim=0)
    if conv.norm_type in {"degree_norm", "sym_degree_norm"}:
        node_degree = incidence_mask.sum(dim=1)
    else:
        # For a diagonal sheaf, the block degree is diag(H H^T).
        node_degree = h_dense.square().sum(dim=1)

    node_power = (
        -0.5
        if conv.norm_type in {"sym_degree_norm", "sym_block_norm"}
        else -1.0
    )
    node_norm = x.new_zeros(node_size)
    edge_norm = x.new_zeros(edge_size)
    node_mask = node_degree > 0
    edge_mask = edge_degree > 0
    node_norm[node_mask] = node_degree[node_mask].pow(node_power)
    edge_norm[edge_mask] = edge_degree[edge_mask].reciprocal()

    if conv.norm_type in {"sym_degree_norm", "sym_block_norm"}:
        x_for_diffusion = x * node_norm.unsqueeze(-1)
        identity_term = x_for_diffusion
    else:
        x_for_diffusion = x
        identity_term = x

    hbht = h_dense @ torch.diag(edge_norm) @ h_dense.t()
    node_cells = torch.arange(node_size, device=x.device) // stalk_dim
    block_diagonal_mask = node_cells[:, None] == node_cells[None, :]
    adjusted_operator = hbht - 2.0 * hbht * block_diagonal_mask

    return identity_term + node_norm.unsqueeze(-1) * (
        adjusted_operator @ x_for_diffusion
    )


### Tests for incidence conversion

# TopoBench stores hypergraph structure as a dense or sparse node-by-hyperedge
# incidence matrix. SheafHyperGNN instead uses a two-row coordinate tensor
# containing one (node, hyperedge) pair per nonzero incidence. The conversion
# also returns the original number of hyperedges so that empty incidence
# columns, representing isolated hyperedges, are not lost.


class TestIncidenceToHyperedgeIndex:
    """Tests for converting TopoBench incidence matrices."""

    def test_dense_incidence(self):
        """Dense input is converted to node-hyperedge coordinate pairs."""
        incidence = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, -1.0, 0.0], # -1 to verify that every nonzero entry counts as an incidence
            ]
        )

        hyperedge_index, num_edges = _incidence_to_hyperedge_index(incidence)

        expected = torch.tensor([[0, 0, 1], [0, 2, 1]])
        assert torch.equal(hyperedge_index, expected)
        assert num_edges == 3

    def test_sparse_incidence_matches_dense(self):
        """Sparse COO and dense inputs produce the same coordinates."""
        incidence = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, -1.0, 0.0],
            ]
        )

        dense_index, dense_num_edges = _incidence_to_hyperedge_index(incidence)
        sparse_index, sparse_num_edges = _incidence_to_hyperedge_index(
            incidence.to_sparse()
        )

        assert torch.equal(sparse_index, dense_index)
        assert sparse_num_edges == dense_num_edges

    def test_sparse_explicit_zero_is_not_an_incidence(self):
        """Stored COO zeros are ignored, matching dense-matrix behavior."""
        indices = torch.tensor([[0, 0, 1], [0, 1, 1]])
        values = torch.tensor([1.0, 0.0, 1.0])
        sparse_incidence = torch.sparse_coo_tensor(
            indices,
            values,
            size=(2, 3),
        )

        sparse_index, num_edges = _incidence_to_hyperedge_index(
            sparse_incidence
        )
        dense_index, _ = _incidence_to_hyperedge_index(
            sparse_incidence.to_dense()
        )

        assert torch.equal(sparse_index, dense_index)
        assert num_edges == 3

    def test_isolated_hyperedge_is_counted(self):
        """An empty incidence column remains included in the hyperedge count."""
        incidence = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        hyperedge_index, num_edges = _incidence_to_hyperedge_index(incidence)

        assert torch.equal(hyperedge_index, torch.tensor([[0, 1], [0, 1]]))
        assert num_edges == 3


### Tests for SheafHyperGNN model


class TestSheafHyperGNN:
    """Tests for SheafHyperGNN forward pass and parameter reset."""

    # The submitted configuration uses only a subset of the options retained
    # from the reference implementation. Compatibility tests below also cover
    # the other options that remain part of the backbone's public interface.

    def test_diagonal_forward_shape(self):
        """Output preserves every stalk coordinate for the TopoBench readout."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch, hidden_channels=hidden_ch, stalk_dim=2
        )
        out, hyp = model(x, inc)

        assert out.shape == (num_nodes, 2 * hidden_ch)
        assert hyp is None

    def test_stalk_dim_one(self):
        """stalk_dim=1 (special case) produces a valid scalar-sheaf representation."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch, hidden_channels=hidden_ch, stalk_dim=1
        )
        out, _ = model(x, inc)

        assert out.shape == (num_nodes, hidden_ch)

    def test_dynamic_sheaf(self, mocker):
        """dynamic_sheaf=True recomputes restriction maps per layer."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch,
            hidden_channels=hidden_ch,
            stalk_dim=2,
            num_layers=2,
            dynamic_sheaf=True,
        )
        builder_spies = [
            mocker.spy(builder, "forward") for builder in model.sheaf_builders
        ]

        out, _ = model(x, inc)

        assert out.shape == (num_nodes, 2 * hidden_ch)
        assert len(model.sheaf_builders) == model.num_layers
        assert all(spy.call_count == 1 for spy in builder_spies)

    def test_static_sheaf_reuses_one_builder(self, mocker):
        """A static sheaf predicts one map that is shared by every layer."""
        num_nodes, in_channels = 8, 12
        model = SheafHyperGNN(
            in_channels=in_channels,
            hidden_channels=16,
            stalk_dim=2,
            num_layers=3,
            dynamic_sheaf=False,
        )
        builder_spy = mocker.spy(model.sheaf_builders[0], "forward")

        model(
            torch.randn(num_nodes, in_channels),
            _make_incidence(num_nodes, 5),
        )

        assert len(model.sheaf_builders) == 1
        assert builder_spy.call_count == 1

    def test_average_hyperedge_initialization_includes_isolated_edge(self):
        """Average initialization returns node means and zero for an empty edge."""
        model = SheafHyperGNN(in_channels=2, hidden_channels=4, num_layers=1)
        x = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [7.0, 8.0],
            ]
        )
        hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])

        edge_features = model._init_hyperedge_attr(
            x,
            hyperedge_index,
            num_edges=3,
        )

        expected = torch.tensor(
            [
                [2.0, 3.0],
                [7.0, 8.0],
                [0.0, 0.0],
            ]
        )
        assert torch.equal(edge_features, expected)

    def test_forward_recomputes_hyperedge_features(self, mocker):
        """Every forward pass initializes features for its current batch."""
        model = SheafHyperGNN(
            in_channels=3,
            hidden_channels=4,
            num_layers=1,
        )
        init_spy = mocker.spy(model, "_init_hyperedge_attr")
        incidence = _make_incidence(num_nodes=4, num_edges=3)

        model(torch.randn(4, 3), incidence)
        model(torch.randn(4, 3), incidence)

        assert init_spy.call_count == 2

    def test_dropout_is_applied_only_between_diffusion_layers(self, mocker):
        """A three-layer model applies feature dropout exactly twice."""
        model = SheafHyperGNN(
            in_channels=3,
            hidden_channels=4,
            num_layers=3,
            dropout=0.5,
            sheaf_dropout=False,
        )
        dropout_spy = mocker.spy(F, "dropout")

        model(torch.randn(4, 3), _make_incidence(4, 3))

        assert dropout_spy.call_count == model.num_layers - 1

    @pytest.mark.parametrize(
        "norm_type",
        ["degree_norm", "sym_degree_norm", "block_norm", "sym_block_norm"],
    )
    def test_empty_incidence_forward(self, norm_type):
        """A complete model handles a batch with no incidences."""
        num_nodes, num_edges, in_channels = 3, 2, 4
        model = SheafHyperGNN(
            in_channels=in_channels,
            hidden_channels=5,
            num_layers=1,
            sheaf_normtype=norm_type,
        )
        incidence = torch.zeros(num_nodes, num_edges)

        out, _ = model(torch.randn(num_nodes, in_channels), incidence)

        assert out.shape == (num_nodes, 2 * 5)
        assert torch.isfinite(out).all()

    def test_rand_hedge_init(self):
        """init_hedge='rand' path runs without error."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch, hidden_channels=hidden_ch, init_hedge="rand"
        )
        out, _ = model(x, inc)
        assert out.shape == (num_nodes, 2 * hidden_ch)

    def test_sparse_coo_input(self):
        """Works when incidence is already in COO format (as TopoBench provides)."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(in_channels=in_ch, hidden_channels=hidden_ch)
        out, _ = model(x, inc)
        assert out.shape == (num_nodes, 2 * hidden_ch)

    def test_reset_parameters(self):
        """The model resets every component without error."""
        model = SheafHyperGNN(in_channels=12, hidden_channels=16)
        model.reset_parameters()

    def test_output_is_not_nan(self):
        """Forward pass produces finite outputs."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(in_channels=in_ch, hidden_channels=hidden_ch)
        model.eval()
        with torch.no_grad():
            out, _ = model(x, inc)
        assert torch.isfinite(out).all()

    def test_gradients_flow_to_all_parameters(self):
        """The loss reaches the input projection, sheaf builder, and convs."""
        model = SheafHyperGNN(
            in_channels=12,
            hidden_channels=16,
            stalk_dim=2,
            num_layers=2,
        )
        x = torch.randn(8, 12)

        out, _ = model(x, _make_incidence(8, 5))
        out.square().mean().backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"non-finite gradient for {name}"
            )

    @pytest.mark.parametrize(
        "model_options",
        [
            {"left_proj": True},
            {"residual": True},
            {"sheaf_special_head": True},
        ],
    )
    def test_optional_diagonal_paths_are_finite(self, model_options):
        """Reference options retained by the port execute for stalk_dim > 1."""
        model = SheafHyperGNN(
            in_channels=12,
            hidden_channels=16,
            stalk_dim=3,
            **model_options,
        )

        out, _ = model(torch.randn(8, 12), _make_incidence(8, 5))

        assert out.shape == (8, 3 * 16)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        ("model_options", "message"),
        [
            ({"stalk_dim": 0}, "stalk_dim"),
            ({"num_layers": 0}, "num_layers"),
            ({"dropout": -0.1}, "dropout"),
            ({"dropout": 1.1}, "dropout"),
            ({"init_hedge": "invalid"}, "init_hedge"),
            ({"sheaf_act": "invalid"}, "sheaf_act"),
            ({"sheaf_pred_block": "invalid"}, "prediction_type"),
            ({"sheaf_normtype": "invalid"}, "norm_type"),
        ],
    )
    def test_invalid_configuration_is_rejected(self, model_options, message):
        """Invalid model options fail during construction with a useful message."""
        with pytest.raises(ValueError, match=message):
            SheafHyperGNN(
                in_channels=4,
                hidden_channels=4,
                **model_options,
            )


### Tests for _MLP


class TestMLP:
    """Tests for internal _MLP helper."""

    def test_shape(self):
        """Output has correct shape."""
        mlp = _MLP(in_channels=8, out_channels=4)
        x = torch.randn(10, 8)
        out = mlp(x)
        assert out.shape == (10, 4)

    def test_input_layer_norm_precedes_linear_projection(self):
        """input_norm applies LayerNorm before the single linear layer."""
        mlp = _MLP(in_channels=3, out_channels=2, input_norm=True)
        x = torch.tensor([[1.0, 2.0, 5.0], [-2.0, 0.0, 4.0]])
        with torch.no_grad():
            mlp.lins[0].weight.copy_(
                torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0]])
            )
            mlp.lins[0].bias.copy_(torch.tensor([0.25, -0.5]))

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(
            variance + mlp.normalizations[0].eps
        )
        expected = normalized @ mlp.lins[0].weight.t() + mlp.lins[0].bias

        assert torch.allclose(mlp(x), expected)

    def test_reset_parameters(self):
        """Reset restores LayerNorm parameters and reinitializes the projection."""
        mlp = _MLP(8, 4, input_norm=True)
        with torch.no_grad():
            mlp.normalizations[0].weight.zero_()
            mlp.normalizations[0].bias.fill_(3.0)
            mlp.lins[0].weight.zero_()

        mlp.reset_parameters()

        assert torch.equal(
            mlp.normalizations[0].weight,
            torch.ones_like(mlp.normalizations[0].weight),
        )
        assert torch.equal(
            mlp.normalizations[0].bias,
            torch.zeros_like(mlp.normalizations[0].bias),
        )
        assert not torch.equal(
            mlp.lins[0].weight,
            torch.zeros_like(mlp.lins[0].weight),
        )


### Tests for the diagonal sheaf builder


class TestDiagonalSheafBuilder:
    """Tests for restriction-map predictor."""

    # The submission uses cp_decomp with tanh and no sheaf dropout or special
    # head. The tests also cover the other supported reference variants.

    @pytest.fixture
    def basic_inputs(self):
        num_nodes, num_edges, d, H = 8, 5, 2, 16
        inc = _make_incidence(num_nodes, num_edges)
        hyperedge_index = inc.coalesce().indices()
        x = torch.randn(num_nodes * d, H)
        e = torch.randn(num_edges * d, H)
        return x, e, hyperedge_index, num_nodes, num_edges, d, H

    @pytest.mark.parametrize(
        "prediction_type",
        ["MLP_var1", "MLP_var2", "MLP_var3", "cp_decomp"],
    )
    def test_prediction_variants_construct_reference_edge_features(
        self,
        prediction_type,
    ):
        """Each predictor builds the hyperedge features used by the reference."""
        hidden_channels, stalk_dim, num_edges = 2, 2, 2
        x = torch.tensor(
            [
                [1.0, -2.0],
                [3.0, 4.0],
                [-1.0, 2.0],
            ]
        )
        e = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        hyperedge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
        row, col = hyperedge_index
        builder = _DiagonalSheafBuilder(
            hidden_channels,
            stalk_dim,
            False,
            sheaf_act="none",
            prediction_type=prediction_type,
            input_norm=False,
        )
        capture = _CaptureProjection(stalk_dim)
        builder.sheaf_lin = capture

        if prediction_type == "MLP_var3":
            builder.sheaf_lin2 = nn.Identity()
        elif prediction_type == "cp_decomp":
            cp_w = nn.Linear(
                hidden_channels + 1,
                hidden_channels,
                bias=False,
            )
            with torch.no_grad():
                cp_w.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ]
                    )
                )
            builder.cp_W = cp_w
            builder.cp_V = nn.Identity()

        out = builder._predict_blocks(x, e, hyperedge_index, num_edges)

        x_row = x[row]
        if prediction_type == "MLP_var1":
            expected_edge_features = e[col]
        else:
            pooled_features = []
            for edge in range(num_edges):
                members = x[row[col == edge]]
                if prediction_type == "MLP_var2":
                    pooled = members.mean(dim=0)
                elif prediction_type == "MLP_var3":
                    pooled = members.sum(dim=0)
                else:
                    pooled = torch.relu(torch.tanh(members).prod(dim=0))
                    pooled = pooled + torch.relu(members.sum(dim=0))
                pooled_features.append(pooled)
            expected_edge_features = torch.stack(pooled_features)[col]

        expected_input = torch.cat((x_row, expected_edge_features), dim=-1)
        assert torch.equal(capture.last_input, expected_input)
        assert torch.equal(
            out,
            torch.zeros(hyperedge_index.shape[1], stalk_dim),
        )

    def test_diagonal_output_shapes(self, basic_inputs):
        """Diagonal builder returns [2, K*d] index and [K*d] values."""
        x, e, hyperedge_index, num_nodes, num_edges, d, H = basic_inputs
        K = hyperedge_index.size(1)

        builder = _DiagonalSheafBuilder(H, d, False)
        h_idx, h_val = builder(
            x,
            e,
            hyperedge_index,
            num_nodes,
            num_edges,
        )

        assert h_idx.shape == (2, K * d)
        assert h_val.shape == (K * d,)

    def test_default_tanh_values_are_bounded(self, basic_inputs):
        """Default restriction-map values are bounded by tanh."""
        x, e, hyperedge_index, num_nodes, num_edges, d, H = basic_inputs

        builder = _DiagonalSheafBuilder(H, d, False)
        _, h_val = builder(x, e, hyperedge_index, num_nodes, num_edges)

        assert (h_val >= -1).all() and (h_val <= 1).all()

    def test_none_activation_returns_raw_projection(self):
        """sheaf_act='none' returns the predictor output without activation."""
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        e = torch.tensor([[5.0, 6.0]])
        hyperedge_index = torch.tensor([[0, 1], [0, 0]])
        builder = _DiagonalSheafBuilder(
            hidden_channels=2,
            stalk_dim=2,
            apply_dropout=False,
            sheaf_act="none",
            prediction_type="MLP_var1",
        )
        predictor_input = torch.cat((x, e.expand(2, -1)), dim=-1)

        expected = builder.sheaf_lin(predictor_input)
        actual = builder._predict_blocks(x, e, hyperedge_index, num_edges=1)

        assert torch.equal(actual, expected)

    def test_sigmoid_values_in_unit_interval(self, basic_inputs):
        """Sigmoid activation remains available as a configured variant."""
        x, e, hyperedge_index, num_nodes, num_edges, d, H = basic_inputs

        builder = _DiagonalSheafBuilder(
            H,
            d,
            False,
            sheaf_act="sigmoid",
        )
        _, h_val = builder(x, e, hyperedge_index, num_nodes, num_edges)

        assert (h_val >= 0).all() and (h_val <= 1).all()

    def test_special_head_is_fixed_to_one(self, basic_inputs):
        """The special scalar head reproduces an ordinary incidence entry."""
        x, e, hyperedge_index, num_nodes, num_edges, d, H = basic_inputs
        builder = _DiagonalSheafBuilder(H, d, False, special_head=True)

        _, h_val = builder(x, e, hyperedge_index, num_nodes, num_edges)
        blocks = h_val.view(-1, d)

        assert torch.equal(blocks[:, -1], torch.ones_like(blocks[:, -1]))

    def test_sheaf_dropout_zeros_maps_at_probability_one(self, basic_inputs):
        """Restriction-map dropout uses the configured probability."""
        x, e, hyperedge_index, num_nodes, num_edges, d, H = basic_inputs
        builder = _DiagonalSheafBuilder(
            H,
            d,
            True,
            dropout=1.0,
        )
        builder.train()

        _, h_val = builder(x, e, hyperedge_index, num_nodes, num_edges)

        assert torch.equal(h_val, torch.zeros_like(h_val))

    def test_empty_incidence_records_empty_test_map(self):
        """The empty builder path returns valid tensors and records its test map."""
        num_nodes, num_edges, d, H = 3, 2, 2, 4
        builder = _DiagonalSheafBuilder(H, d, False)
        hyperedge_index = torch.empty((2, 0), dtype=torch.long)

        h_idx, h_val = builder(
            torch.randn(num_nodes * d, H),
            torch.randn(num_edges * d, H),
            hyperedge_index,
            num_nodes,
            num_edges,
        )

        assert h_idx.shape == (2, 0)
        assert h_val.shape == (0,)
        assert builder.last_restriction_maps.shape == (0, d)

    @pytest.mark.parametrize(
        "prediction_type",
        ["MLP_var1", "MLP_var2", "MLP_var3", "cp_decomp"],
    )
    def test_reset_parameters(self, basic_inputs, prediction_type):
        """Every predictor variant resets its configured layers without error."""
        _, _, _, _, _, d, H = basic_inputs
        builder = _DiagonalSheafBuilder(
            H,
            d,
            False,
            prediction_type=prediction_type,
        )
        builder.reset_parameters()

    @pytest.mark.parametrize("stalk_dim", [1, 2, 3, 6])
    def test_expand_diagonal_matches_expected_coordinates(self, stalk_dim):
        """Expansion creates the expected diagonal coordinates and value order."""
        hyperedge_index = torch.tensor([[1, 2, 4], [0, 3, 1]])
        restriction_diagonals = torch.arange(
            hyperedge_index.shape[1] * stalk_dim,
            dtype=torch.float32,
        ).view(-1, stalk_dim)

        expanded_index, expanded_values = _expand_diagonal(
            hyperedge_index,
            restriction_diagonals,
            stalk_dim,
        )

        expected_coordinates = [
            [
                node * stalk_dim + coordinate,
                hyperedge * stalk_dim + coordinate,
            ]
            for node, hyperedge in hyperedge_index.t().tolist()
            for coordinate in range(stalk_dim)
        ]
        expected_index = torch.tensor(expected_coordinates).t().contiguous()
        expected_values = restriction_diagonals.reshape(-1)

        assert torch.equal(expanded_index, expected_index)
        assert torch.equal(expanded_values, expected_values)


### Tests for diagonal sheaf diffusion


class TestDiagonalSheafConv:
    """Tests for sheaf diffusion convolution."""

    # The submission uses symmetric degree normalization without left
    # projection or a convolution-level residual. The tests also cover the
    # other supported reference variants.

    def test_output_shape(self):
        """Output shape equals input shape [N*d, H]."""
        num_nodes, num_edges, d, H = 8, 5, 2, 16
        inc = _make_incidence(num_nodes, num_edges)
        hyperedge_index = inc.coalesce().indices()
        x = torch.randn(num_nodes * d, H)
        builder = _DiagonalSheafBuilder(H, d, False)
        h_idx, h_val = builder(
            x,
            torch.randn(num_edges * d, H),
            hyperedge_index,
            num_nodes,
            num_edges,
        )

        conv = _DiagonalSheafConv(H, d)
        out = conv(x, h_idx, h_val, num_nodes, num_edges)

        assert out.shape == (num_nodes * d, H)

    def test_output_finite(self):
        """Diffusion output is finite (no NaN/Inf from degree normalisation)."""
        num_nodes, num_edges, d, H = 8, 5, 2, 16
        inc = _make_incidence(num_nodes, num_edges)
        hyperedge_index = inc.coalesce().indices()

        x = torch.randn(num_nodes * d, H)
        builder = _DiagonalSheafBuilder(H, d, False)
        h_idx, h_val = builder(
            x,
            torch.randn(num_edges * d, H),
            hyperedge_index,
            num_nodes,
            num_edges,
        )

        conv = _DiagonalSheafConv(H, d)
        out = conv(x, h_idx, h_val, num_nodes, num_edges)

        assert torch.isfinite(out).all()

    def test_residual_and_bias_are_added_after_diffusion(self):
        """Residual input and output bias are added to the diffusion result."""
        num_nodes, num_edges, d, H = 3, 2, 2, 3
        hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        restriction_maps = torch.tensor(
            [[0.2, -0.7], [1.1, 0.3], [-0.5, 0.9], [0.4, -1.2]]
        )
        h_idx, h_val = _expand_diagonal(
            hyperedge_index,
            restriction_maps,
            d,
        )
        x = torch.randn(num_nodes * d, H)
        base_conv = _DiagonalSheafConv(H, d, bias=False, residual=False)
        residual_conv = _DiagonalSheafConv(H, d, bias=True, residual=True)
        residual_conv.lin.load_state_dict(base_conv.lin.state_dict())
        with torch.no_grad():
            residual_conv.bias.fill_(0.25)

        base_out = base_conv(x, h_idx, h_val, num_nodes, num_edges)
        residual_out = residual_conv(
            x,
            h_idx,
            h_val,
            num_nodes,
            num_edges,
        )
        transformed = residual_conv.lin(x)

        assert torch.allclose(
            residual_out,
            base_out + transformed + residual_conv.bias,
        )

    def test_left_projection_matches_manually_preprojected_input(self):
        """The optional stalk projection is applied before channel projection."""
        num_nodes, num_edges, d, H = 3, 2, 2, 3
        hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        restriction_maps = torch.tensor(
            [[0.2, -0.7], [1.1, 0.3], [-0.5, 0.9], [0.4, -1.2]]
        )
        h_idx, h_val = _expand_diagonal(
            hyperedge_index,
            restriction_maps,
            d,
        )
        x = torch.randn(num_nodes * d, H)
        projected_conv = _DiagonalSheafConv(
            H,
            d,
            left_proj=True,
            bias=False,
        )
        plain_conv = _DiagonalSheafConv(H, d, bias=False)
        plain_conv.lin.load_state_dict(projected_conv.lin.state_dict())

        manually_projected = x.t().reshape(-1, d)
        manually_projected = projected_conv.lin_left_proj(manually_projected)
        manually_projected = manually_projected.reshape(
            -1,
            num_nodes * d,
        ).t()

        projected_out = projected_conv(
            x,
            h_idx,
            h_val,
            num_nodes,
            num_edges,
        )
        expected_out = plain_conv(
            manually_projected,
            h_idx,
            h_val,
            num_nodes,
            num_edges,
        )

        assert torch.allclose(projected_out, expected_out)

    def test_reset_parameters_resets_bias_and_left_projection(self):
        """Reset covers the optional stalk projection and output bias."""
        conv = _DiagonalSheafConv(
            hidden_channels=3,
            stalk_dim=2,
            input_norm=True,
            left_proj=True,
            bias=True,
        )
        with torch.no_grad():
            conv.bias.fill_(4.0)
            conv.lin_left_proj.normalizations[0].weight.zero_()

        conv.reset_parameters()

        assert torch.equal(conv.bias, torch.zeros_like(conv.bias))
        assert torch.equal(
            conv.lin_left_proj.normalizations[0].weight,
            torch.ones_like(conv.lin_left_proj.normalizations[0].weight),
        )

        conv_without_bias = _DiagonalSheafConv(3, 2, bias=False)
        conv_without_bias.reset_parameters()
        assert conv_without_bias.bias is None

    @pytest.mark.parametrize(
        ("norm_type", "expected_scale"),
        [
            ("degree_norm", 1.0),
            ("sym_degree_norm", 0.0),
            ("block_norm", 1.0),
            ("sym_block_norm", 0.0),
        ],
    )
    def test_isolated_node_uses_zero_degree_convention(
        self, norm_type, expected_scale
    ):
        """An isolated node is finite and follows the reference D^-1 mask."""
        num_nodes, num_edges, d, H = 3, 1, 2, 3
        hyperedge_index = torch.tensor([[0, 1], [0, 0]])
        maps = torch.tensor([[0.5, 0.8], [-0.3, 1.2]])
        h_idx, h_val = _expand_diagonal(hyperedge_index, maps, d)
        x = torch.randn(num_nodes * d, H)
        conv = _DiagonalSheafConv(
            H,
            d,
            norm_type=norm_type,
            input_norm=False,
            bias=False,
        )
        with torch.no_grad():
            conv.lin.lins[0].weight.copy_(torch.eye(H))
            conv.lin.lins[0].bias.zero_()

        out = conv(x, h_idx, h_val, num_nodes, num_edges)
        isolated = slice(2 * d, 3 * d)

        assert torch.isfinite(out).all()
        assert torch.allclose(out[isolated], expected_scale * x[isolated])

    @pytest.mark.parametrize(
        ("norm_type", "expected_scale"),
        [
            ("degree_norm", 2.0),
            ("sym_degree_norm", 1.0),
            ("block_norm", 2.0),
            ("sym_block_norm", 1.0),
        ],
    )
    def test_empty_incidence_and_residual(self, norm_type, expected_scale):
        """Empty incidence follows reference normalization and residual rules."""
        num_nodes, num_edges, d, H = 3, 2, 2, 4
        x = torch.randn(num_nodes * d, H)
        h_idx = torch.empty((2, 0), dtype=torch.long)
        h_val = torch.empty(0)
        conv = _DiagonalSheafConv(
            H,
            d,
            norm_type=norm_type,
            residual=True,
            bias=False,
        )

        transformed = conv.lin(x)
        out = conv(x, h_idx, h_val, num_nodes, num_edges)

        assert torch.allclose(out, expected_scale * transformed)

    @pytest.mark.parametrize(
        "norm_type",
        ["degree_norm", "sym_degree_norm", "block_norm", "sym_block_norm"],
    )
    def test_diffusion_matches_dense_reference(self, norm_type):
        """Diffusion output matches the explicit dense reference operator."""
        torch.manual_seed(0)
        num_nodes, num_edges, d, H = 3, 2, 2, 4
        hyperedge_index = torch.tensor(
            [
                [0, 1, 1, 2],
                [0, 0, 1, 1],
            ],
            dtype=torch.long,
        )
        restriction_maps = torch.tensor(
            [
                [0.2, -0.7],
                [1.1, 0.3],
                [-0.5, 0.9],
                [0.4, -1.2],
            ],
            dtype=torch.float32,
        )
        h_idx, h_val = _expand_diagonal(
            hyperedge_index,
            restriction_maps,
            d,
        )
        x = torch.randn(num_nodes * d, H)

        conv = _DiagonalSheafConv(H, d, norm_type=norm_type, bias=False)
        conv.eval()

        out_scatter = conv(x, h_idx, h_val, num_nodes, num_edges)
        out_dense = _dense_sheaf_conv_reference(
            conv, x, h_idx, h_val, num_nodes, num_edges
        )

        assert torch.allclose(out_scatter, out_dense, atol=1e-6)

    def test_invalid_normalisation_is_rejected(self):
        """The public normalization enum fails early on an invalid value."""
        with pytest.raises(ValueError, match="norm_type"):
            _DiagonalSheafConv(4, 2, norm_type="not_a_norm")


def test_config_disables_wrapper_residual_and_uses_node_readout():
    """Config preserves backbone embeddings until the node-level classifier."""
    cfg = OmegaConf.load("configs/model/hypergraph/sheaf_hypergnn.yaml")

    assert cfg.backbone.stalk_dim == 6
    assert cfg.backbone.sheaf_pred_block == "cp_decomp"
    assert cfg.backbone_wrapper.num_cell_dimensions == 1
    assert cfg.readout.num_cell_dimensions == 1
    assert cfg.backbone_wrapper.residual_connections is False
    assert cfg.readout.readout_name == "NoReadOut"


def test_wrapper_and_readout_smoke_with_config_dimensions():
    """HypergraphWrapper output can pass through the configured readout."""
    cfg = OmegaConf.load("configs/model/hypergraph/sheaf_hypergnn.yaml")
    num_nodes, in_ch, hidden_ch = 6, 8, 8
    inc = _make_incidence(num_nodes=num_nodes, num_edges=4)

    batch = Data(
        x_0=torch.randn(num_nodes, in_ch),
        incidence_hyperedges=inc,
        y=torch.zeros(num_nodes, dtype=torch.long),
        batch_0=torch.zeros(num_nodes, dtype=torch.long),
    )

    stalk_dim = cfg.backbone.stalk_dim
    backbone = SheafHyperGNN(
        in_channels=in_ch,
        hidden_channels=hidden_ch,
        stalk_dim=stalk_dim,
    )
    embedding_dim = stalk_dim * hidden_ch
    wrapper = HypergraphWrapper(
        backbone,
        out_channels=embedding_dim,
        num_cell_dimensions=cfg.backbone_wrapper.num_cell_dimensions,
        residual_connections=cfg.backbone_wrapper.residual_connections,
    )
    model_out = wrapper(batch)
    assert model_out["x_0"].shape == (num_nodes, embedding_dim)

    readout = NoReadOut(
        readout_name="NoReadOut",
        num_cell_dimensions=cfg.readout.num_cell_dimensions,
        hidden_dim=embedding_dim,
        out_channels=3,
        task_level="node",
        pooling_type="sum",
    )
    model_out = readout(model_out, batch)

    assert model_out["logits"].shape == (num_nodes, 3)
