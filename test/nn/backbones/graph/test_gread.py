"""Unit tests for the GREAD backbone."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.graph import GREAD
from topobench.nn.wrappers.graph import GNNWrapper


class TestGREAD:
    """Unit tests for the GREAD backbone."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(0)
        self.num_nodes = 8
        self.in_channels = 12
        self.hidden_channels = 16
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_index = torch.randint(
            0, self.num_nodes, (2, self.num_nodes * 2)
        )

    @pytest.mark.parametrize(
        "reaction_term",
        [
            "bspm",
            "fisher",
            "allen-cahn",
            "zeldovich",
            "st",
            "fb",
            "fb3",
            "none",
        ],
    )
    def test_forward_reaction_terms(self, reaction_term):
        """Test the forward pass for every reaction term.

        Parameters
        ----------
        reaction_term : str
            Reaction term to test.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term=reaction_term,
        )
        out = model(self.x, self.edge_index)
        assert out.shape == (self.num_nodes, self.hidden_channels)
        assert torch.isfinite(out).all()

    def test_invalid_reaction_term(self):
        """Test that an invalid reaction term raises a ValueError."""
        with pytest.raises(ValueError):
            GREAD(self.in_channels, self.hidden_channels, reaction_term="foo")

    def test_invalid_data_norm(self):
        """Test that an invalid data normalization raises a ValueError."""
        with pytest.raises(ValueError):
            GREAD(self.in_channels, self.hidden_channels, data_norm="foo")

    def test_gcn_norm(self):
        """Test the forward pass with symmetric (GCN) normalization."""
        model = GREAD(self.in_channels, self.hidden_channels, data_norm="gcn")
        out = model(self.x, self.edge_index)
        assert out.shape == (self.num_nodes, self.hidden_channels)

    def test_beta_diag_and_source(self):
        """Test the diagonal beta parameterization and the source term."""
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            beta_diag=True,
            add_source=True,
        )
        assert model.b_w.shape == (self.hidden_channels,)
        out = model(self.x, self.edge_index)
        out.sum().backward()
        assert model.b_w.grad is not None
        assert model.source_train.grad is not None

    def test_gradient_flow(self):
        """Test that gradients flow through alpha, beta and the encoder."""
        model = GREAD(self.in_channels, self.hidden_channels)
        out = model(self.x, self.edge_index)
        out.sum().backward()
        assert model.alpha_train.grad is not None
        assert model.beta_train.grad is not None
        assert model.encoder.weight.grad is not None

    def test_euler_step_pure_diffusion(self):
        """Test one Euler step of pure diffusion against a manual computation.

        With ``reaction_term="none"``, one Euler step computes
        ``H(t + tau) = H(t) + tau * sigmoid(alpha) * (A - I) H(t)``.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="none",
            time=1.0,
            step_size=1.0,
            xn_activation=True,
        )
        model.eval()
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h0 = model.encoder(self.x)
            ah0 = model.spmm(edge_index, edge_weight, self.num_nodes, h0)
            alpha = torch.sigmoid(model.alpha_train)
            expected = torch.relu(h0 + 1.0 * alpha * (ah0 - h0))
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_rw_normalization_column_stochastic(self):
        """Test that the rw-normalized adjacency is column-normalized.

        Matches ``get_rw_adj(..., norm_dim=1)`` of the reference
        implementation, where the degree is computed over the target
        (column) index.
        """
        model = GREAD(self.in_channels, self.hidden_channels)
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        col_sums = torch.zeros(self.num_nodes)
        col_sums.scatter_add_(0, edge_index[1], edge_weight)
        assert torch.allclose(col_sums, torch.ones(self.num_nodes), atol=1e-6)

    def test_beta_diag_weights_reaction_per_channel(self):
        """Test the ``beta_diag`` update against a manual Euler step.

        With ``beta_diag=True`` the reaction is scaled by the trainable
        diagonal :math:`\\beta_W` — used directly, not through a sigmoid —
        instead of the scalar gate :math:`\\beta = \\sigma(\\beta_{train})`,
        matching the "(VC)" variants of the paper.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="fisher",
            time=1.0,
            step_size=1.0,
            beta_diag=True,
            xn_activation=False,
        )
        model.eval()
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h0 = model.encoder(self.x)
            ax = model.spmm(edge_index, edge_weight, self.num_nodes, h0)
            diffusion = ax - h0
            reaction = -(h0 - 1.0) * h0
            alpha = torch.sigmoid(model.alpha_train)
            expected = h0 + (alpha * diffusion + reaction * model.b_w)
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

        # The per-channel gate must not collapse to the scalar beta path.
        scalar_beta = torch.sigmoid(model.beta_train)
        scalar_expected = h0 + (alpha * diffusion + scalar_beta * reaction)
        assert not torch.allclose(out, scalar_expected, atol=1e-4)

    def test_self_loop_weight(self):
        """Test that ``self_loop_weight`` enters the normalized adjacency.

        On the single-edge graph ``0 -> 1`` with self-loop fill value
        :math:`w`, the column-stochastic normalization :math:`A D^{-1}`
        (degrees over the target index) gives :math:`A_{0,0} = 1`,
        :math:`A_{0,1} = 1 / (1 + w)` and :math:`A_{1,1} = w / (1 + w)`.
        """
        edge_index = torch.tensor([[0], [1]])
        for weight in (0.5, 2.0):
            model = GREAD(
                self.in_channels,
                self.hidden_channels,
                self_loop_weight=weight,
            )
            out_index, out_weight = model.normalize_adjacency(
                edge_index, None, 2
            )
            dense = torch.zeros(2, 2)
            dense[out_index[0], out_index[1]] = out_weight
            expected = torch.tensor(
                [
                    [1.0, 1.0 / (1.0 + weight)],
                    [0.0, weight / (1.0 + weight)],
                ]
            )
            assert torch.allclose(dense, expected, atol=1e-6)

        # A non-positive weight must leave the edge set untouched.
        model = GREAD(
            self.in_channels, self.hidden_channels, self_loop_weight=0.0
        )
        out_index, out_weight = model.normalize_adjacency(edge_index, None, 2)
        assert out_index.shape == (2, 1)
        assert torch.allclose(out_weight, torch.tensor([1.0]), atol=1e-6)

    def test_dropout_is_training_only(self):
        """Test that dropout perturbs training passes but not eval passes."""
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            input_dropout=0.9,
            dropout=0.9,
        )
        model.train()
        torch.manual_seed(1)
        first = model(self.x, self.edge_index)
        torch.manual_seed(2)
        second = model(self.x, self.edge_index)
        assert not torch.allclose(first, second)

        model.eval()
        with torch.no_grad():
            repeatable = model(self.x, self.edge_index)
            assert torch.allclose(
                repeatable, model(self.x, self.edge_index), atol=1e-6
            )
            # In eval mode the result must not depend on the dropout rates.
            model.input_dropout = 0.0
            model.dropout = 0.0
            assert torch.allclose(
                repeatable, model(self.x, self.edge_index), atol=1e-6
            )

    def test_euler_step_schedule(self):
        """Test the Euler step schedule for divisible and partial horizons."""
        model = GREAD(
            self.in_channels, self.hidden_channels, time=3.0, step_size=1.0
        )
        assert model.step_sizes == [1.0, 1.0, 1.0]
        model = GREAD(
            self.in_channels, self.hidden_channels, time=0.5, step_size=1.0
        )
        assert model.step_sizes == [0.5]
        model = GREAD(
            self.in_channels, self.hidden_channels, time=2.5, step_size=1.0
        )
        assert model.step_sizes == [1.0, 1.0, 0.5]

    def test_tiny_horizon_step_schedule(self):
        """Test that a tiny positive horizon still yields one step."""
        model = GREAD(
            self.in_channels, self.hidden_channels, time=1e-12, step_size=1.0
        )
        assert model.step_sizes == [1e-12]
        assert sum(model.step_sizes) == 1e-12

    def test_near_multiple_horizon_step_schedule(self):
        """Test that the step sizes sum exactly to a near-multiple horizon."""
        time = 1.0000000005
        model = GREAD(
            self.in_channels, self.hidden_channels, time=time, step_size=1.0
        )
        assert len(model.step_sizes) == 2
        assert sum(model.step_sizes) == time

    def test_default_xn_activation_disabled(self):
        """Test that the terminal ReLU is disabled by default.

        Matches the ``XN_activation`` CLI default of the reference
        implementation.
        """
        model = GREAD(self.in_channels, self.hidden_channels)
        assert model.xn_activation is False
        model.eval()
        with torch.no_grad():
            out = model(self.x, self.edge_index)
        assert (out < 0).any()

    @pytest.mark.parametrize(
        "time, step_size",
        [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -0.5)],
    )
    def test_nonpositive_time_or_step_size_rejected(self, time, step_size):
        """Test that non-positive time or step size raises a ValueError.

        Parameters
        ----------
        time : float
            Terminal integration time.
        step_size : float
            Euler step size.
        """
        with pytest.raises(ValueError):
            GREAD(
                self.in_channels,
                self.hidden_channels,
                time=time,
                step_size=step_size,
            )

    def test_partial_final_step_numerical(self):
        """Test integration with a partial final step against manual Euler.

        With ``reaction_term="none"`` and ``time=2.5``, ``step_size=1.0``,
        the model must take steps of 1.0, 1.0 and 0.5:
        ``H_{k+1} = H_k + tau_k * sigmoid(alpha) * (A - I) H_k``.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="none",
            time=2.5,
            step_size=1.0,
            xn_activation=True,
        )
        model.eval()
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h = model.encoder(self.x)
            alpha = torch.sigmoid(model.alpha_train)
            for tau in (1.0, 1.0, 0.5):
                ah = model.spmm(edge_index, edge_weight, self.num_nodes, h)
                h = h + tau * alpha * (ah - h)
            expected = torch.relu(h)
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_short_horizon_numerical(self):
        """Test that ``time < step_size`` takes a single step of size time.

        With ``time=0.5``, ``step_size=1.0`` a single Euler step of size
        0.5 must be taken:
        ``H(0.5) = H(0) + 0.5 * sigmoid(alpha) * (A - I) H(0)``.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="none",
            time=0.5,
            step_size=1.0,
            xn_activation=True,
        )
        model.eval()
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h0 = model.encoder(self.x)
            ah0 = model.spmm(edge_index, edge_weight, self.num_nodes, h0)
            alpha = torch.sigmoid(model.alpha_train)
            expected = torch.relu(h0 + 0.5 * alpha * (ah0 - h0))
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_zero_initialized_source_contributes_nothing(self):
        """Test that a zero-initialized source term contributes zero.

        With ``add_source=True`` and the source coefficient at its zero
        initialization, the output must equal that of an identical model
        without the source term.
        """
        torch.manual_seed(1)
        with_source = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="none",
            add_source=True,
        )
        torch.manual_seed(1)
        without_source = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="none",
            add_source=False,
        )
        with_source.eval()
        without_source.eval()
        assert with_source.source_train.item() == 0.0
        with torch.no_grad():
            out_with = with_source(self.x, self.edge_index)
            out_without = without_source(self.x, self.edge_index)
        assert torch.allclose(out_with, out_without, atol=1e-7)

    def test_source_term_no_sigmoid(self):
        """Test the source-term contribution numerically without sigmoid.

        With a source coefficient ``s``, each Euler step must add
        ``tau * s * H(0)`` (not ``tau * sigmoid(s) * H(0)``).
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term="none",
            time=1.0,
            step_size=1.0,
            add_source=True,
            xn_activation=False,
        )
        model.eval()
        with torch.no_grad():
            model.source_train.fill_(0.3)
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h0 = model.encoder(self.x)
            ah0 = model.spmm(edge_index, edge_weight, self.num_nodes, h0)
            alpha = torch.sigmoid(model.alpha_train)
            expected = h0 + 1.0 * (alpha * (ah0 - h0) + 0.3 * h0)
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_xn_activation_toggle(self):
        """Test the terminal ReLU toggle numerically.

        With ``xn_activation=False`` the raw terminal state is returned;
        with ``xn_activation=True`` its ReLU is returned.
        """
        torch.manual_seed(2)
        relu_model = GREAD(
            self.in_channels,
            self.hidden_channels,
            xn_activation=True,
        )
        torch.manual_seed(2)
        raw_model = GREAD(
            self.in_channels,
            self.hidden_channels,
            xn_activation=False,
        )
        relu_model.eval()
        raw_model.eval()
        with torch.no_grad():
            out_relu = relu_model(self.x, self.edge_index)
            out_raw = raw_model(self.x, self.edge_index)
        assert (out_raw < 0).any()
        assert torch.allclose(out_relu, torch.relu(out_raw), atol=1e-7)

    @pytest.mark.parametrize(
        "reaction_term, reaction_fn",
        [
            ("fisher", lambda h, h0: -(h - 1) * h),
            ("allen-cahn", lambda h, h0: -(h**2 - 1) * h),
            ("zeldovich", lambda h, h0: -(h**2 - h) * h),
            ("st", lambda h, h0: h0),
        ],
    )
    def test_pointwise_reactions_numerical(self, reaction_term, reaction_fn):
        """Test pointwise reaction terms against manual Euler computations.

        One Euler step computes ``H1 = H0 + tau * (sigmoid(alpha) *
        (A - I) H0 + sigmoid(beta) * r(H0))``.

        Parameters
        ----------
        reaction_term : str
            Reaction term to test.
        reaction_fn : callable
            Manual implementation of the reaction term.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term=reaction_term,
            time=1.0,
            step_size=1.0,
            xn_activation=False,
        )
        model.eval()
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h0 = model.encoder(self.x)
            ah0 = model.spmm(edge_index, edge_weight, self.num_nodes, h0)
            alpha = torch.sigmoid(model.alpha_train)
            beta = torch.sigmoid(model.beta_train)
            expected = h0 + 1.0 * (
                alpha * (ah0 - h0) + beta * reaction_fn(h0, h0)
            )
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.parametrize("reaction_term", ["bspm", "fb", "fb3"])
    def test_graph_reactions_numerical(self, reaction_term):
        """Test graph-coupled reaction terms against manual computations.

        Covers ``bspm`` (``-A(A - I)H``), ``fb`` (``H + AH``) and
        ``fb3`` (``fb`` plus an extra ``H`` inside the beta gate).

        Parameters
        ----------
        reaction_term : str
            Reaction term to test.
        """
        model = GREAD(
            self.in_channels,
            self.hidden_channels,
            reaction_term=reaction_term,
            time=1.0,
            step_size=1.0,
            xn_activation=False,
        )
        model.eval()
        edge_index, edge_weight = model.normalize_adjacency(
            self.edge_index, None, self.num_nodes
        )
        with torch.no_grad():
            h0 = model.encoder(self.x)
            ah0 = model.spmm(edge_index, edge_weight, self.num_nodes, h0)
            diffusion = ah0 - h0
            alpha = torch.sigmoid(model.alpha_train)
            beta = torch.sigmoid(model.beta_train)
            if reaction_term == "bspm":
                reaction = -model.spmm(
                    edge_index, edge_weight, self.num_nodes, diffusion
                )
                f = alpha * diffusion + beta * reaction
            else:
                reaction = h0 + model.spmm(
                    edge_index, edge_weight, self.num_nodes, h0
                )
                if reaction_term == "fb3":
                    f = alpha * diffusion + beta * (reaction + h0)
                else:
                    f = alpha * diffusion + beta * reaction
            expected = h0 + 1.0 * f
            out = model(self.x, self.edge_index)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_edge_weight_passthrough(self):
        """Test the forward pass with explicit edge weights."""
        model = GREAD(self.in_channels, self.hidden_channels)
        edge_weight = torch.rand(self.edge_index.size(1))
        out = model(self.x, self.edge_index, edge_weight=edge_weight)
        assert out.shape == (self.num_nodes, self.hidden_channels)

    def test_with_gnn_wrapper(self, random_graph_input):
        """Test GREAD within the GNNWrapper.

        Parameters
        ----------
        random_graph_input : tuple
            Fixture with random graph inputs.
        """
        x, _, _, edges_1, _ = random_graph_input
        batch = torch_geometric.data.Data(
            x_0=x,
            y=x,
            x=x,
            edge_index=edges_1,
            batch_0=torch.zeros(x.shape[0], dtype=torch.long),
        )
        model = GREAD(x.shape[1], x.shape[1])
        wrapper = GNNWrapper(
            model,
            out_channels=x.shape[1],
            num_cell_dimensions=1,
        )
        model_out = wrapper(batch)
        assert model_out["x_0"].shape == x.shape
