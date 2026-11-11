"""Unit tests for the gauge-equivariant graph backbone."""

import math

import pytest
import torch
import torch_geometric
from torch import nn

from topobench.nn.backbones.graph.gauge import (
    FFBlock,
    GatedFlatteningLayer,
    GaugeLayer,
    GaugeModel,
    LocalCoordinatesLayer,
    MultiHeadFF,
    MultiHeadLinear,
    NodeUpdateLayer,
    activation_dict,
)
from topobench.nn.wrappers.graph import GaugeWrapper


def _is_orthonormal(Q, atol=1e-5):
    """Check that the frames in ``Q`` have orthonormal rows.

    Parameters
    ----------
    Q : torch.Tensor
        Per-node frames of shape ``[N, r, d]``.
    atol : float, optional
        Absolute tolerance for the identity comparison (default: 1e-5).

    Returns
    -------
    bool
        True if ``Q @ Q^T`` equals the ``r x r`` identity for every node.
    """
    N, r, _ = Q.shape
    gram = Q @ Q.transpose(-2, -1)
    eye = torch.eye(r).expand(N, r, r)
    return torch.allclose(gram, eye, atol=atol)


class TestMultiHeadLinear:
    """Tests for the per-head linear layer."""

    def test_forward_shape(self):
        """Output has shape ``[N, r, out_channels]``."""
        layer = MultiHeadLinear(in_channels=4, out_channels=6, r_dim=3)
        Z = torch.randn(10, 3, 4)
        out = layer(Z)
        assert out.shape == (10, 3, 6)

    def test_parameter_shapes(self):
        """Stacked weight and bias have the expected per-head shapes."""
        layer = MultiHeadLinear(in_channels=4, out_channels=6, r_dim=3)
        assert layer.superW.shape == (3, 6, 4)
        assert layer.superB.shape == (3, 6)

    def test_no_bias(self):
        """With ``bias=False`` no bias parameter is registered."""
        layer = MultiHeadLinear(
            in_channels=4, out_channels=6, r_dim=3, bias=False
        )
        assert layer.superB is None
        Z = torch.randn(5, 3, 4)
        assert layer(Z).shape == (5, 3, 6)

    def test_heads_are_independent(self):
        """Each head applies its own weight matrix and bias.

        The batched ``einsum`` must agree with applying each head's linear map
        one at a time.
        """
        layer = MultiHeadLinear(in_channels=4, out_channels=6, r_dim=3)
        Z = torch.randn(10, 3, 4)
        out = layer(Z)
        for h in range(3):
            expected = Z[:, h, :] @ layer.superW[h].T + layer.superB[h]
            assert torch.allclose(out[:, h, :], expected, atol=1e-5)

    def test_permuting_one_head_leaves_others_untouched(self):
        """Changing a head's input only changes that head's output."""
        layer = MultiHeadLinear(
            in_channels=4, out_channels=6, r_dim=3, bias=False
        )
        Z = torch.randn(10, 3, 4)
        out = layer(Z)
        Z2 = Z.clone()
        Z2[:, 1, :] = torch.randn(10, 4)
        out2 = layer(Z2)
        assert torch.allclose(out[:, 0, :], out2[:, 0, :])
        assert torch.allclose(out[:, 2, :], out2[:, 2, :])
        assert not torch.allclose(out[:, 1, :], out2[:, 1, :])

    def test_reset_parameters_bound(self):
        """Weights are initialized within ``1 / sqrt(in_channels)``."""
        in_channels = 9
        layer = MultiHeadLinear(
            in_channels=in_channels, out_channels=6, r_dim=3
        )
        bound = 1 / math.sqrt(in_channels)
        assert layer.superW.abs().max().item() <= bound + 1e-6
        assert layer.superB.abs().max().item() <= bound + 1e-6

    def test_reset_parameters_changes_weights(self):
        """Calling ``reset_parameters`` re-samples the weights."""
        layer = MultiHeadLinear(in_channels=4, out_channels=6, r_dim=3)
        before = layer.superW.clone()
        layer.reset_parameters()
        assert not torch.allclose(before, layer.superW)


class TestMultiHeadFF:
    """Tests for the per-head feed-forward network."""

    def test_single_layer_shape(self):
        """With ``hidden_dims=None`` the network is a single per-head map."""
        net = MultiHeadFF(in_channels=4, out_channels=2, r=3)
        # exactly one MultiHeadLinear, no activation / dropout
        assert len(net.model) == 1
        assert isinstance(net.model[0], MultiHeadLinear)
        Z = torch.randn(7, 3, 4)
        assert net(Z).shape == (7, 3, 2)

    def test_multi_layer_shape(self):
        """Hidden dims add intermediate per-head layers."""
        net = MultiHeadFF(
            in_channels=4, out_channels=2, r=3, hidden_dims=[8, 8]
        )
        Z = torch.randn(7, 3, 4)
        assert net(Z).shape == (7, 3, 2)

    def test_no_activation_after_output(self):
        """Activation and dropout appear only between layers."""
        net = MultiHeadFF(in_channels=4, out_channels=2, r=3, hidden_dims=[8])
        # Sequence: Linear, Act, Dropout, Linear -> last module is Linear
        assert isinstance(net.model[-1], MultiHeadLinear)
        linears = [m for m in net.model if isinstance(m, MultiHeadLinear)]
        assert len(linears) == 2

    @pytest.mark.parametrize("act", list(activation_dict.keys()))
    def test_activation_choice(self, act):
        """Every activation in ``activation_dict`` is wired up correctly.

        Parameters
        ----------
        act : str
            Name of the activation function to test.
        """
        net = MultiHeadFF(
            in_channels=4, out_channels=2, r=3, hidden_dims=[8], act=act
        )
        acts = [m for m in net.model if isinstance(m, activation_dict[act])]
        assert len(acts) == 1
        Z = torch.randn(7, 3, 4)
        assert net(Z).shape == (7, 3, 2)

    def test_final_activation(self):
        """``final_activation`` appends the activation after the output layer."""
        net = MultiHeadFF(
            in_channels=4,
            out_channels=2,
            r=3,
            hidden_dims=[8],
            act="leaky_relu",
            final_activation=True,
        )
        # Output layer is followed by the activation, not left bare.
        assert isinstance(net.model[-1], nn.LeakyReLU)
        # Two activations total: one between layers, one after the output.
        acts = [m for m in net.model if isinstance(m, nn.LeakyReLU)]
        assert len(acts) == 2
        Z = torch.randn(7, 3, 4)
        assert net(Z).shape == (7, 3, 2)


class TestFFBlock:
    """Tests for the feed-forward block."""

    def test_forward_shape(self):
        """Output has the requested number of channels."""
        block = FFBlock(in_channels=4, out_channels=8, hidden_dim=16)
        x = torch.randn(5, 4)
        assert block(x).shape == (5, 8)

    def test_norm_on_input_channels(self):
        """The pre-norm LayerNorm normalizes over ``in_channels``."""
        block = FFBlock(in_channels=4, out_channels=8, hidden_dim=16)
        assert block.norm.normalized_shape == (4,)

    def test_extra_leading_dims(self):
        """The block broadcasts over arbitrary leading dimensions."""
        block = FFBlock(in_channels=4, out_channels=8, hidden_dim=16)
        x = torch.randn(5, 3, 4)
        assert block(x).shape == (5, 3, 8)

    def test_rejects_negative_hidden_layers(self):
        """A negative number of hidden layers is rejected."""
        with pytest.raises(AssertionError):
            FFBlock(
                in_channels=4, out_channels=8, hidden_dim=16, n_hidden_layers=-1
            )

    def test_zero_hidden_layers_is_single_linear(self):
        """With no hidden layers the block is a single linear map."""
        block = FFBlock(
            in_channels=4, out_channels=8, hidden_dim=16, n_hidden_layers=0
        )
        linears = [m for m in block.model if isinstance(m, nn.Linear)]
        assert len(linears) == 1
        assert linears[0].in_features == 4
        assert linears[0].out_features == 8
        assert block(torch.randn(5, 4)).shape == (5, 8)

    @pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
    def test_num_hidden_layers(self, n_hidden_layers):
        """Multiple hidden layers still produce the right output shape.

        Parameters
        ----------
        n_hidden_layers : int
            Number of hidden layers to configure.
        """
        block = FFBlock(
            in_channels=4,
            out_channels=8,
            hidden_dim=16,
            n_hidden_layers=n_hidden_layers,
        )
        x = torch.randn(5, 4)
        assert block(x).shape == (5, 8)

    def test_default_activation_is_gelu(self):
        """The default activation stays GELU for backward compatibility."""
        block = FFBlock(in_channels=4, out_channels=8, hidden_dim=16)
        acts = [m for m in block.model if isinstance(m, nn.GELU)]
        assert len(acts) == 1

    @pytest.mark.parametrize("act", list(activation_dict.keys()))
    def test_activation_choice(self, act):
        """Every activation in ``activation_dict`` is wired up correctly.

        Parameters
        ----------
        act : str
            Name of the activation function to test.
        """
        block = FFBlock(
            in_channels=4,
            out_channels=8,
            hidden_dim=16,
            n_hidden_layers=2,
            act=act,
        )
        acts = [m for m in block.model if isinstance(m, activation_dict[act])]
        # one activation per hidden layer
        assert len(acts) == 2
        x = torch.randn(5, 4)
        assert block(x).shape == (5, 8)


class TestLocalCoordinatesLayer:
    """Tests for the local coordinate (frame) layer."""

    def test_output_shape(self, simple_graph_0):
        """The layer returns one ``[r, d]`` frame per node.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        Z = torch.randn(simple_graph_0.num_nodes, d)
        Q = layer(Z, simple_graph_0.edge_index)
        assert Q.shape == (simple_graph_0.num_nodes, r, d)

    def test_frames_orthonormal(self, simple_graph_0):
        """The QR step yields orthonormal per-node frames.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        Z = torch.randn(simple_graph_0.num_nodes, d)
        Q = layer(Z, simple_graph_0.edge_index)
        assert _is_orthonormal(Q)

    def test_no_nan(self, simple_graph_0):
        """Output frames are finite.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 4
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        Z = torch.randn(simple_graph_0.num_nodes, d)
        Q = layer(Z, simple_graph_0.edge_index)
        assert not torch.isnan(Q).any()
        assert not torch.isinf(Q).any()

    def test_all_nodes_isolated_no_nan(self):
        """
        A graph with zero edges still produces finite, orthonormal frames.

        """
        d, r = 8, 3
        edge_index = torch.tensor([[], []], dtype=torch.long)
        Z = torch.randn(3, d)
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        Q = layer(Z, edge_index)
        assert Q.shape == (3, r, d)
        assert _is_orthonormal(Q)
        assert not torch.isnan(Q).any()

    def test_isolated_nodes_independent(self):
        """
        With no edges, changing one node's features must not affect others.

        """
        d, r = 8, 3
        edge_index = torch.tensor([[],[]], dtype =torch.long)
        Z = torch.randn(3, d)
        Z1 = Z.clone()
        Z1[1] = torch.randn(d)
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        layer.eval()
        Q = layer(Z, edge_index)
        Q1 = layer(Z1, edge_index)
        assert torch.allclose(Q[0], Q1[0])
        assert torch.allclose(Q[2], Q1[2])
        assert not torch.allclose(Q[1], Q1[1])

    def test_r_greater_than_d_raises(self):
        """``r > d_embedd`` is rejected instead of silently truncating to d.

        A d-dimensional space admits at most d orthonormal frame vectors, so
        the layer must refuse to be constructed rather than dropping subspaces.
        """
        with pytest.raises(ValueError):
            LocalCoordinatesLayer(r_subspaces=8, d_embedd=3)

    def test_respects_edge_direction(self):
        """A directed edge only influences its destination, never its source."""
        d, r = 8, 3
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        Z = torch.randn(2, d)
        Z1 = Z.clone()
        Z1[1] = torch.randn(d)
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        layer.eval()
        Q = layer(Z, edge_index)
        Q1 = layer(Z1, edge_index)
        assert torch.allclose(Q[0], Q1[0])
        assert not torch.allclose(Q[1], Q1[1])

    def test_star_topology_no_nan(self):
        """A hub node with far more neighbors than the rest of the graph must not produce NaNs."""
        d, r = 8, 3
        n_leaves = 20
        N = n_leaves + 1
        leaves = torch.arange(1, N)
        hub = torch.zeros(n_leaves, dtype=torch.long)
        src = torch.cat([hub, leaves])
        dst = torch.cat([leaves, hub])
        edge_index = torch.stack([src, dst])
        layer = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        Z = torch.randn(N, d)
        Q = layer(Z, edge_index)
        assert Q.shape == (N, r, d)
        assert _is_orthonormal(Q)
        assert not torch.isnan(Q).any() and not torch.isinf(Q).any()


class TestGatedFlatteningLayer:
    """Tests for the gated flattening (frame smoothing) layer."""

    def test_output_shape_and_orthonormality(self, simple_graph_0):
        """Smoothing preserves the shape and orthonormality of the frames.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        N = simple_graph_0.num_nodes
        # Build orthonormal input frames via the local-coords layer.
        coords = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)
        Q = coords(torch.randn(N, d), simple_graph_0.edge_index)

        gate = GatedFlatteningLayer(r=r)
        Qnew = gate(Q, simple_graph_0.edge_index)
        assert Qnew.shape == (N, r, d)
        assert _is_orthonormal(Qnew)

    def test_no_learnable_parameters(self):
        """The gated flattening layer is parameter-free."""
        gate = GatedFlatteningLayer(r=3)
        assert list(gate.parameters()) == []

    @pytest.mark.xfail(
        reason="QR's backward is undefined for rank-deficient input; "
        "gamma=1.0 zeroes an isolated node's blend, causing NaN/Inf grads."
    )
    def test_isolated_node_gamma_one_backward(self):
        """Gradients must stay finite even when gamma=1.0 zeroes an isolated node's blend."""
        d, r = 8, 3
        N = 3
        edge_index = torch.tensor([[], []], dtype=torch.long)
        Z = torch.randn(N, d, requires_grad=True)
        Q = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)(Z, edge_index)
        gate = GatedFlatteningLayer(r=r, gamma=1.0)
        Qnew = gate(Q, edge_index)
        Qnew.sum().backward()
        assert Z.grad is not None
        assert not torch.isnan(Z.grad).any()
        assert not torch.isinf(Z.grad).any()


class TestNodeUpdateLayer:
    """Tests for the node feature update layer."""

    def test_output_shape(self, simple_graph_0):
        """The update returns ``out_channels`` features per node.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        N = simple_graph_0.num_nodes
        layer = NodeUpdateLayer(in_channels=d, out_channels=d)
        Z = torch.randn(N, d)
        Q = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)(
            Z, simple_graph_0.edge_index
        )
        out = layer(Z, Q, simple_graph_0.edge_index)
        assert out.shape == (N, d)

    def test_residual_disabled(self, simple_graph_0):
        """``phi_hidden_layers=None`` disables the residual MLP.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        N = simple_graph_0.num_nodes
        layer = NodeUpdateLayer(
            in_channels=d, out_channels=d, phi_hidden_layers=None
        )
        assert layer.phi is None
        Z = torch.randn(N, d)
        Q = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)(
            Z, simple_graph_0.edge_index
        )
        assert layer(Z, Q, simple_graph_0.edge_index).shape == (N, d)

    def test_residual_enabled(self, simple_graph_0):
        """The residual MLP is built when enabled.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        layer = NodeUpdateLayer(
            in_channels=8, out_channels=8, phi_hidden_layers=1
        )
        assert isinstance(layer.phi, FFBlock)

    def test_supports_different_in_out_channels(self, simple_graph_0):
        """The update maps to ``out_channels`` even when it differs from ``in_channels``.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        out_channels = 12
        N = simple_graph_0.num_nodes
        Z = torch.randn(N, d)
        Q = LocalCoordinatesLayer(r_subspaces=r, d_embedd=d)(
            Z, simple_graph_0.edge_index
        )
        layer = NodeUpdateLayer(
            in_channels=d, out_channels=out_channels, phi_hidden_layers=1
        )
        assert layer(Z, Q, simple_graph_0.edge_index).shape == (N, out_channels)


class TestGaugeLayer:
    """Tests for a single gauge message-passing layer."""

    def test_output_shapes(self, simple_graph_0):
        """The layer returns updated features and orthonormal frames.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        d, r = 8, 3
        N = simple_graph_0.num_nodes
        layer = GaugeLayer(d_embedd=d, r=r, n_gated=2)
        x = torch.randn(N, d)
        Znew, Q = layer(x, simple_graph_0.edge_index)
        assert Znew.shape == (N, d)
        assert Q.shape == (N, r, d)
        assert _is_orthonormal(Q)

    @pytest.mark.parametrize("n_gated", [0, 1, 3])
    def test_num_gated_layers(self, simple_graph_0, n_gated):
        """The number of gated flattening sublayers is configurable.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        n_gated : int
            Number of gated flattening layers to stack.
        """
        d, r = 8, 3
        layer = GaugeLayer(d_embedd=d, r=r, n_gated=n_gated)
        assert len(layer.gated_flattening_layers) == n_gated
        x = torch.randn(simple_graph_0.num_nodes, d)
        Znew, Q = layer(x, simple_graph_0.edge_index)
        assert _is_orthonormal(Q)

    def test_all_nodes_isolated_no_nan(self):

        """A zero-edge graph stays finite through the full gauge layer."""

        d, r = 8, 3
        N = 3
        edge_index = torch.tensor([[], []], dtype=torch.long)
        layer = GaugeLayer(d_embedd=d, r=r, n_gated=2)
        x = torch.randn(N, d)
        Znew, Q = layer(x, edge_index)
        assert Znew.shape == (N, d)
        assert Q.shape == (N, r, d)
        assert _is_orthonormal(Q)
        assert not torch.isnan(Znew).any() and not torch.isinf(Znew).any()
        assert not torch.isnan(Q).any() and not torch.isinf(Q).any()



class TestGaugeModel:
    """Tests for the full gauge model."""

    def test_forward_shapes(self, simple_graph_0):
        """The model maps input features to embeddings and final frames.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        in_channels, d, r = 5, 8, 3
        N = simple_graph_0.num_nodes
        model = GaugeModel(
            n_layers=2, in_channels=in_channels, r=r, d_embedd=d
        )
        x = torch.randn(N, in_channels)
        z, Q = model(x, simple_graph_0.edge_index)
        assert z.shape == (N, d)
        assert Q.shape == (N, r, d)
        assert _is_orthonormal(Q)

    def test_self_loops_do_not_affect_output(self, simple_graph_0):
        """Adding self-loops leaves the model output unchanged.

        The model strips self-loops from ``edge_index`` before aggregating, so
        feeding a graph augmented with a self-loop on every node must yield the
        same embeddings and frames as the original graph.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        in_channels, d, r = 5, 8, 3
        N = simple_graph_0.num_nodes
        model = GaugeModel(
            n_layers=2, in_channels=in_channels, r=r, d_embedd=d
        )
        model.eval()  # disable dropout so the comparison is deterministic
        x = torch.randn(N, in_channels)

        edge_index = simple_graph_0.edge_index
        looped, _ = torch_geometric.utils.add_self_loops(
            edge_index, num_nodes=N
        )

        z, Q = model(x, edge_index)
        z_looped, Q_looped = model(x, looped)

        assert torch.allclose(z, z_looped, atol=1e-6)
        assert torch.allclose(Q, Q_looped, atol=1e-6)

    def test_return_initial_true_returns_initial_projection(
        self, simple_graph_0
    ):
        """With ``return_initial=True`` the initial projection is returned.

        The default forward pass yields the triple ``(z, Q, z0)`` where ``z0``
        is exactly the input projection of ``x`` (before any gauge layer).

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        in_channels, d, r = 5, 8, 3
        N = simple_graph_0.num_nodes
        model = GaugeModel(
            n_layers=2, in_channels=in_channels, r=r, d_embedd=d
        )
        model.eval()
        x = torch.randn(N, in_channels)

        out = model(x, simple_graph_0.edge_index, return_initial=True)
        assert len(out) == 3
        z, Q, z0 = out
        assert z.shape == (N, d)
        assert Q.shape == (N, r, d)
        assert z0.shape == (N, d)
        # z0 is precisely the input projection, unaffected by the gauge layers.
        assert torch.allclose(z0, model.input_projector(x))

    def test_return_initial_default_returns_pair(self, simple_graph_0):
        """Omitting ``return_initial`` returns the ``(z, Q)`` pair by default.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8)
        x = torch.randn(simple_graph_0.num_nodes, 5)
        assert len(model(x, simple_graph_0.edge_index)) == 2

    def test_return_initial_false_returns_pair(self, simple_graph_0):
        """With ``return_initial=False`` only ``(z, Q)`` is returned.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8)
        x = torch.randn(simple_graph_0.num_nodes, 5)
        out = model(x, simple_graph_0.edge_index)
        assert len(out) == 2

    @pytest.mark.parametrize("n_layers", [1, 2, 4])
    def test_num_layers(self, simple_graph_0, n_layers):
        """The model stacks the requested number of gauge layers.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        n_layers : int
            Number of gauge layers to stack.
        """
        in_channels, d, r = 5, 8, 3
        model = GaugeModel(
            n_layers=n_layers, in_channels=in_channels, r=r, d_embedd=d
        )
        assert len(model.layers) == n_layers
        x = torch.randn(simple_graph_0.num_nodes, in_channels)
        z, _ = model(x, simple_graph_0.edge_index)
        assert z.shape == (simple_graph_0.num_nodes, d)

    def test_no_nan(self, simple_graph_0):
        """The final embeddings are finite.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8)
        x = torch.randn(simple_graph_0.num_nodes, 5)
        z, Q = model(x, simple_graph_0.edge_index)
        assert not torch.isnan(z).any() and not torch.isinf(z).any()
        assert not torch.isnan(Q).any() and not torch.isinf(Q).any()

    def test_backward_pass(self, simple_graph_0):
        """Gradients flow back to the input and the parameters.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8)
        x = torch.randn(simple_graph_0.num_nodes, 5, requires_grad=True)
        z, _ = model(x, simple_graph_0.edge_index)
        z.sum().backward()
        assert x.grad is not None
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_deterministic_in_eval(self, simple_graph_0):
        """Two eval-mode passes on the same input agree.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8)
        model.eval()
        x = torch.randn(simple_graph_0.num_nodes, 5)
        z1, _ = model(x, simple_graph_0.edge_index)
        z2, _ = model(x, simple_graph_0.edge_index)
        assert torch.allclose(z1, z2)

    def test_batched_graphs(self, simple_graph_0, simple_graph_1):
        """The model handles a batch of disconnected graphs.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            First test graph fixture.
        simple_graph_1 : torch_geometric.data.Data
            Second test graph fixture.
        """
        batch = torch_geometric.data.Batch.from_data_list(
            [simple_graph_0, simple_graph_1]
        )
        n_total = simple_graph_0.num_nodes + simple_graph_1.num_nodes
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8)
        x = torch.randn(n_total, 5)
        z, _ = model(x, batch.edge_index)
        assert z.shape == (n_total, 8)

    @pytest.mark.parametrize("act", list(activation_dict.keys()))
    def test_activation_propagates_to_ffblocks(self, act):
        """The ``act`` argument reaches every feed-forward block.

        The activation must flow from the model down to the ``fflayer`` of each
        local-coordinates layer and the ``phi`` residual of each node update.

        Parameters
        ----------
        act : str
            Name of the activation function to test.
        """
        model = GaugeModel(n_layers=2, in_channels=5, r=3, d_embedd=8, act=act)
        expected = activation_dict[act]
        for layer in model.layers:
            fflayer = layer.local_coords_layer.fflayer
            assert any(isinstance(m, expected) for m in fflayer.model)
            phi = layer.node_update_layer.phi
            assert any(isinstance(m, expected) for m in phi.model)

    def test_f_sim_act_defaults_to_leaky_relu(self):
        """The similarity scorer defaults to LeakyReLU, independent of ``act``."""
        model = GaugeModel(
            n_layers=2, in_channels=5, r=3, d_embedd=8, act="gelu"
        )
        for layer in model.layers:
            f_sim = layer.local_coords_layer.f_sim
            assert any(isinstance(m, nn.LeakyReLU) for m in f_sim.model)

    @pytest.mark.parametrize("f_sim_act", list(activation_dict.keys()))
    def test_f_sim_act_applied_to_final_score(self, f_sim_act):
        """``f_sim`` ends on its activation, matching the reference score_lin.

        The reference ``score_lin`` (Linear -> activation) activates the score
        before the softmax; here ``f_sim`` is built with ``final_activation``,
        so the last module of its sequential is the activation.

        Parameters
        ----------
        f_sim_act : str
            Name of the similarity-scorer activation to test.
        """
        model = GaugeModel(
            n_layers=2,
            in_channels=5,
            r=3,
            d_embedd=8,
            f_sim_act=f_sim_act,
        )
        expected = activation_dict[f_sim_act]
        for layer in model.layers:
            f_sim = layer.local_coords_layer.f_sim
            assert f_sim.final_activation is True
            assert isinstance(f_sim.model[-1], expected)

    @pytest.mark.parametrize("f_sim_act", list(activation_dict.keys()))
    def test_f_sim_act_propagates(self, f_sim_act):
        """The ``f_sim_act`` argument reaches every similarity network.

        The knob is independent of ``act``: it must only affect ``f_sim``, not
        the ``fflayer`` feed-forward blocks.

        Parameters
        ----------
        f_sim_act : str
            Name of the similarity-scorer activation to test.
        """
        model = GaugeModel(
            n_layers=2,
            in_channels=5,
            r=3,
            d_embedd=8,
            act="gelu",
            f_sim_act=f_sim_act,
        )
        expected = activation_dict[f_sim_act]
        for layer in model.layers:
            f_sim = layer.local_coords_layer.f_sim
            assert any(isinstance(m, expected) for m in f_sim.model)
            # ``act`` still governs the feed-forward block independently.
            fflayer = layer.local_coords_layer.fflayer
            assert any(isinstance(m, nn.GELU) for m in fflayer.model)

    def test_dropout_propagates_to_ffblocks(self):
        """The ``dropout`` argument reaches every feed-forward block.

        The probability must flow from the model down to the ``fflayer`` of
        each local-coordinates layer and the ``phi`` residual of each node
        update.
        """
        model = GaugeModel(
            n_layers=2, in_channels=5, r=3, d_embedd=8, dropout=0.42
        )
        for layer in model.layers:
            fflayer = layer.local_coords_layer.fflayer
            phi = layer.node_update_layer.phi
            for block in (fflayer, phi):
                dropouts = [
                    m for m in block.model if isinstance(m, nn.Dropout)
                ]
                assert dropouts
                assert all(m.p == 0.42 for m in dropouts)

    def test_f_sim_dropout_propagates(self):
        """The ``f_sim_dropout`` knob reaches ``f_sim`` and is independent.

        It must only configure the similarity network, leaving the
        feed-forward block dropout governed by ``dropout``. Because ``f_sim``
        has no hidden layers, ``MultiHeadFF`` (which applies dropout only
        between layers) instantiates no active dropout module, so the check is
        on the propagated probability rather than on the module list.
        """
        model = GaugeModel(
            n_layers=2,
            in_channels=5,
            r=3,
            d_embedd=8,
            dropout=0.1,
            f_sim_dropout=0.5,
        )
        for layer in model.layers:
            local = layer.local_coords_layer
            assert local.f_sim_dropout == 0.5
            assert local.f_sim.dropout == 0.5
            # The feed-forward block keeps its own dropout probability.
            ff_drops = [
                m for m in local.fflayer.model if isinstance(m, nn.Dropout)
            ]
            assert all(m.p == 0.1 for m in ff_drops)

    def test_all_nodes_isolated_no_nan(self):
        """A zero-edge graph stays finite through the full stacked model."""
        in_channels, d, r, N = 5, 8, 3, 3
        edge_index = torch.tensor([[], []], dtype=torch.long)
        model = GaugeModel(
            n_layers=2, in_channels=in_channels, r=r, d_embedd=d
        )
        x = torch.randn(N, in_channels)
        z, Q = model(x, edge_index)
        assert z.shape == (N, d)
        assert Q.shape == (N, r, d)
        assert _is_orthonormal(Q)
        assert not torch.isnan(z).any() and not torch.isinf(z).any()
        assert not torch.isnan(Q).any() and not torch.isinf(Q).any()


class TestGaugeWrapper:
    """Tests for the topobench wrapper around the gauge model."""

    def test_forward(self, simple_graph_0):
        """The wrapper forwards node embeddings and the initial state.

        In addition to ``x_0`` it exposes the initial projection ``z_0`` and
        the final frames ``Q`` (consumed by the custom loss).

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        # The wrapper adds a residual (batch.x_0 + model output), so the input
        # width must match the embedding width ``d``.
        in_channels = d = 8
        r = 3
        N = simple_graph_0.num_nodes
        model = GaugeModel(
            n_layers=2, in_channels=in_channels, r=r, d_embedd=d
        )
        wrapper = GaugeWrapper(model, out_channels=d, num_cell_dimensions=1)
        batch = torch_geometric.data.Data(
            x_0=torch.randn(N, in_channels),
            edge_index=simple_graph_0.edge_index,
            y=simple_graph_0.y,
            batch_0=torch.zeros(N, dtype=torch.long),
        )
        model_out = wrapper(batch)
        assert model_out["x_0"].shape == (N, d)
        assert model_out["z_0"].shape == (N, d)
        assert model_out["Q"].shape == (N, r, d)
        assert "labels" in model_out
        assert "batch_0" in model_out

    def test_forward_residual_off(self, simple_graph_0):
        """With the residual disabled the backbone width may differ from the input.

        This mirrors the shipped config (``residual_connections: false``,
        ``out_channels = d_embedd`` != encoder width), so the embedding width is
        free to differ from ``in_channels``.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        in_channels, d, r = 5, 8, 3
        N = simple_graph_0.num_nodes
        model = GaugeModel(
            n_layers=2, in_channels=in_channels, r=r, d_embedd=d
        )
        wrapper = GaugeWrapper(
            model,
            out_channels=d,
            num_cell_dimensions=1,
            residual_connections=False,
        )
        assert wrapper.residual_connections is False
        batch = torch_geometric.data.Data(
            x_0=torch.randn(N, in_channels),
            edge_index=simple_graph_0.edge_index,
            y=simple_graph_0.y,
            batch_0=torch.zeros(N, dtype=torch.long),
        )
        model_out = wrapper(batch)
        assert model_out["x_0"].shape == (N, d)
