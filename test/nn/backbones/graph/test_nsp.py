r"""Unit tests for NSP (Neural Sheaf Propagation).

The tests are organised around the claims of Suk et al., "Surfing on the Neural
Sheaf" (https://openreview.net/pdf?id=xOXFkyRzTlu), referred to as [1]:

* the layer equation
  :math:`\mathbf{X}_{t+1} = 2\mathbf{X}_t - \mathbf{X}_{t-1} -
  h\,\sigma(\Delta_{\mathcal{F}(t)}(\mathbf{I} \otimes W_1^t)\mathbf{X}_t W_2^t)`
  of [1], Section 3, checked against a dense replay of the same algebra;
* the sheaf Laplacian properties of [1], Section 2 (symmetric positive
  semi-definite, and bounded spectrum once normalised);
* Proposition 1 of [1], energy conservation, which is what separates the wave
  equation from sheaf diffusion.
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import remove_self_loops, to_undirected

from topobench.nn.backbones.graph.nsd_utils.orthogonal import Orthogonal
from topobench.nn.backbones.graph.nsp import SHEAF_TYPES, NSPEncoder


SHEAF_TYPE_NAMES = sorted(SHEAF_TYPES)


def make_encoder(**kwargs):
    """Build an NSPEncoder with test-friendly, deterministic defaults.

    Parameters
    ----------
    **kwargs : dict
        Overrides forwarded to :class:`NSPEncoder`.

    Returns
    -------
    NSPEncoder
        Encoder in eval mode with dropout disabled.
    """
    config = dict(
        input_dim=12,
        hidden_dim=10,
        d=2,
        sheaf_type="bundle",
        num_layers=3,
        dropout=0.0,
        input_dropout=0.0,
    )
    config.update(kwargs)
    return NSPEncoder(**config).eval()


def ring_graph(num_nodes=9, feat_dim=12, seed=0):
    """Build a connected ring graph with random features.

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes. Default is 9.
    feat_dim : int, optional
        Node feature dimension. Default is 12.
    seed : int, optional
        Seed for the feature draw. Default is 0.

    Returns
    -------
    x : torch.Tensor
        Node features of shape ``[num_nodes, feat_dim]``.
    edge_index : torch.Tensor
        Edge indices of shape ``[2, num_nodes]``.
    """
    torch.manual_seed(seed)
    source = torch.arange(num_nodes)
    target = (source + 1) % num_nodes
    return torch.randn(num_nodes, feat_dim), torch.stack([source, target])


def dense_laplacian(model, cochain, edge_index, layer=0):
    r"""Densify :math:`\Delta_{\mathcal{F}(t)}` as the model builds it.

    Parameters
    ----------
    model : NSPEncoder
        Encoder whose sheaf learner and Laplacian builder to use.
    cochain : torch.Tensor
        0-cochain of shape ``[num_nodes * d, channels]`` driving the sheaf.
    edge_index : torch.Tensor
        Undirected, self-loop-free edge indices.
    layer : int, optional
        Index of the sheaf learner to query. Default is 0.

    Returns
    -------
    torch.Tensor
        Dense sheaf Laplacian of shape ``[num_nodes * d, num_nodes * d]``.
    """
    num_nodes = cochain.size(0) // model.d
    builder = model.builder_cls(
        num_nodes, edge_index, d=model.d, **model.builder_kwargs
    )
    maps = model.sheaf_learners[layer](
        cochain.reshape(num_nodes, -1), edge_index
    )
    (indices, values), _ = builder(maps)
    size = num_nodes * model.d
    return torch.zeros(size, size).index_put_(
        (indices[0], indices[1]), values, accumulate=True
    )


def encode(model, x):
    r"""Run the input MLP and stack the result into a 0-cochain.

    Parameters
    ----------
    model : NSPEncoder
        Encoder providing ``lin1``.
    x : torch.Tensor
        Node features of shape ``[num_nodes, input_dim]``.

    Returns
    -------
    torch.Tensor
        0-cochain :math:`\mathbf{X}_0` of shape ``[num_nodes * d, channels]``.
    """
    return F.elu(model.lin1(x)).view(x.size(0) * model.d, -1)


class TestPaperEquations:
    """Check the implementation against the equations of [1]."""

    @pytest.mark.parametrize("sheaf_type", SHEAF_TYPE_NAMES)
    def test_layer_equation_matches_dense_reference(self, sheaf_type):
        """Each layer reproduces the leapfrog equation of [1], Section 3.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_type=sheaf_type, num_layers=3)
        undirected = to_undirected(
            remove_self_loops(edge_index)[0], num_nodes=x.size(0)
        )
        num_nodes = x.size(0)

        # Replay X_{t+1} = 2 X_t - X_{t-1} - h sigma(Delta (I kron W_1) X_t W_2)
        # with dense algebra, reusing the model's own weights and sheaf.
        identity = torch.eye(num_nodes)
        x_prev = x_curr = encode(model, x)
        for layer in range(model.num_layers):
            laplacian = dense_laplacian(model, x_curr, undirected, layer)
            mixed = (
                torch.kron(identity, model.lin_left_weights[layer].weight)
                @ x_curr
                @ model.lin_right_weights[layer].weight.t()
            )
            x_prev, x_curr = (
                x_curr,
                2 * x_curr
                - x_prev
                - model.step_size * F.elu(laplacian @ mixed),
            )
        expected = model.lin2(x_curr.reshape(num_nodes, -1))

        assert torch.allclose(
            model(x, edge_index), expected, atol=1e-5, rtol=1e-4
        )

    @pytest.mark.parametrize("sheaf_type", SHEAF_TYPE_NAMES)
    def test_step_size_zero_freezes_the_cochain(self, sheaf_type):
        """``h = 0`` leaves the leapfrog at its initial condition.

        With :math:`\\mathbf{X}_{-1} = \\mathbf{X}_0` (zero initial velocity)
        the recurrence collapses to :math:`\\mathbf{X}_{t+1} = 2\\mathbf{X}_t -
        \\mathbf{X}_{t-1} = \\mathbf{X}_0`, so no number of layers may move it.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        x, edge_index = ring_graph()
        model = make_encoder(
            sheaf_type=sheaf_type, num_layers=7, step_size=0.0
        )
        frozen = model.lin2(encode(model, x).reshape(x.size(0), -1))

        assert torch.allclose(model(x, edge_index), frozen, atol=1e-6)

    @pytest.mark.parametrize("sheaf_type", SHEAF_TYPE_NAMES)
    def test_sheaf_laplacian_is_symmetric_psd(self, sheaf_type):
        """[1], Section 2: :math:`L_{\\mathcal{F}}` is symmetric PSD.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_type=sheaf_type)
        undirected = to_undirected(
            remove_self_loops(edge_index)[0], num_nodes=x.size(0)
        )
        laplacian = dense_laplacian(model, encode(model, x), undirected)

        assert torch.allclose(laplacian, laplacian.t(), atol=1e-6)
        assert torch.linalg.eigvalsh(laplacian).min() > -1e-5

    @pytest.mark.parametrize("sheaf_type", ["diag", "bundle"])
    def test_normalised_variants_have_bounded_spectrum(self, sheaf_type):
        """The normalised :math:`\\Delta_{\\mathcal{F}}` has eigenvalues in [0, 2].

        [1], Section 2 defines :math:`\\Delta_{\\mathcal{F}} = D^{-1/2}
        L_{\\mathcal{F}} D^{-1/2}`, whose spectrum is what keeps the
        non-dissipative leapfrog stable at ``step_size = 1``.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_type=sheaf_type)
        undirected = to_undirected(
            remove_self_loops(edge_index)[0], num_nodes=x.size(0)
        )
        laplacian = dense_laplacian(model, encode(model, x), undirected)

        assert torch.linalg.eigvalsh(laplacian).max() <= 2.0 + 1e-5

    def test_energy_conservation(self):
        r"""[1], Proposition 1: the wave equation preserves the sheaf energy.

        Runs the linear leapfrog on the Laplacian the model actually builds and
        tracks :math:`\mathcal{E}_{\mathcal{F}} = \tfrac{1}{2}(\lVert
        \dot{\mathbf{X}} \rVert^2 + \mathbf{X}^\top \Delta_{\mathcal{F}}
        \mathbf{X})`. A first-order (diffusive) step on the same operator is
        included as a control: it must visibly dissipate the same energy.
        """
        x, edge_index = ring_graph()
        model = make_encoder(num_layers=1)
        undirected = to_undirected(
            remove_self_loops(edge_index)[0], num_nodes=x.size(0)
        )
        x_0 = encode(model, x)
        laplacian = dense_laplacian(model, x_0, undirected)

        step = 0.05

        def energy(current, previous):
            velocity = (current - previous) / step
            return 0.5 * (
                (velocity * velocity).sum()
                + (current * (laplacian @ current)).sum()
            )

        # Half step to start the leapfrog from zero initial velocity.
        x_prev = x_0
        x_curr = x_0 - 0.5 * step**2 * (laplacian @ x_0)
        wave = [energy(x_curr, x_prev)]
        for _ in range(400):
            x_prev, x_curr = (
                x_curr,
                2 * x_curr - x_prev - step**2 * (laplacian @ x_curr),
            )
            wave.append(energy(x_curr, x_prev))
        wave = torch.stack(wave) / wave[0]

        assert wave.max() < 1.05
        assert wave.min() > 0.95

        x_prev = x_curr = x_0
        for _ in range(400):
            x_prev, x_curr = (
                x_curr,
                x_curr - step**2 * (laplacian @ x_curr),
            )
        assert energy(x_curr, x_prev) / (wave[0] * energy(x_0, x_0)) < 0.5

    @pytest.mark.parametrize("orth", ["cayley", "matrix_exp"])
    def test_bundle_restriction_maps_are_orthogonal(self, orth):
        """O(d)-NSP transports isometrically, so its maps satisfy Q Q^T = I.

        Parameters
        ----------
        orth : str
            Orthogonal parameterisation under test.
        """
        d = 4
        maps = Orthogonal(d=d, orthogonal_map=orth)(
            torch.randn(7, d * (d + 1) // 2)
        )

        assert torch.allclose(
            maps @ maps.transpose(-1, -2),
            torch.eye(d).expand_as(maps),
            atol=1e-5,
        )

    def test_variant_table_matches_the_paper(self):
        """The three variants of [1], Table 1, are wired as documented."""
        assert SHEAF_TYPE_NAMES == ["bundle", "diag", "general"]

        # Diag-NSP normalises explicitly; O(d)-NSP's builder normalises by
        # construction and instead needs the orthogonal parameterisation;
        # Gen-NSP is the documented unnormalised deviation.
        assert make_encoder(sheaf_type="diag").builder_kwargs == {
            "normalised": True
        }
        assert make_encoder(sheaf_type="bundle").builder_kwargs == {
            "orth_map": "cayley"
        }
        assert make_encoder(sheaf_type="general").builder_kwargs == {}

    @pytest.mark.parametrize(
        ("sheaf_type", "expected"),
        [("diag", (3,)), ("bundle", (6,)), ("general", (3, 3))],
    )
    def test_restriction_map_parameter_shapes(self, sheaf_type, expected):
        """Each family asks its learner for the right per-edge parameters.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        expected : tuple of int
            Expected per-edge parameter shape for ``d = 3``.
        """
        model = make_encoder(sheaf_type=sheaf_type, d=3, hidden_dim=12)

        assert model.sheaf_learners[0].out_shape == expected


class TestNSPEncoder:
    """Check construction, propagation and gradients of the encoder."""

    def test_default_construction(self):
        """Defaults follow the O(d)-NSP variant of [1]."""
        model = NSPEncoder(input_dim=12, hidden_dim=10)

        assert model.sheaf_type == "bundle"
        assert model.d == 2
        assert model.step_size == 1.0
        assert model.hidden_channels == 5
        assert len(model.sheaf_learners) == model.num_layers == 2
        assert len(model.lin_left_weights) == 2
        assert len(model.lin_right_weights) == 2

    def test_unknown_sheaf_type_raises(self):
        """An unsupported restriction-map family is rejected."""
        with pytest.raises(ValueError, match="Unknown sheaf_type"):
            NSPEncoder(input_dim=12, hidden_dim=10, sheaf_type="wave")

    @pytest.mark.parametrize("sheaf_type", ["bundle", "general"])
    def test_scalar_stalk_rejected_for_matrix_variants(self, sheaf_type):
        """``d = 1`` degenerates O(d)-NSP and Gen-NSP to a scalar sheaf.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        with pytest.raises(ValueError, match="needs d > 1"):
            NSPEncoder(
                input_dim=12, hidden_dim=10, d=1, sheaf_type=sheaf_type
            )

    def test_diag_variant_accepts_scalar_stalk(self):
        """Diag-NSP with ``d = 1`` is the scalar sheaf of Hansen and Gebhart."""
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_type="diag", d=1, hidden_dim=8)

        assert model(x, edge_index).shape == (x.size(0), 8)

    def test_non_positive_stalk_rejected(self):
        """A stalk dimension must be at least 1."""
        with pytest.raises(ValueError, match="needs d > 1"):
            NSPEncoder(input_dim=12, hidden_dim=10, d=0, sheaf_type="diag")

    def test_indivisible_hidden_dim_rejected(self):
        """A stalk that does not divide the width would silently truncate it."""
        with pytest.raises(ValueError, match="must be a multiple of"):
            NSPEncoder(input_dim=12, hidden_dim=10, d=4)

    def test_unknown_sheaf_activation_rejected(self):
        """The sheaf learner rejects unsupported activations."""
        with pytest.raises(ValueError, match="Unsupported act"):
            NSPEncoder(input_dim=12, hidden_dim=10, sheaf_act="softmax")

    @pytest.mark.parametrize("sheaf_type", SHEAF_TYPE_NAMES)
    @pytest.mark.parametrize("num_layers", [1, 2, 5])
    def test_forward_shape(self, sheaf_type, num_layers):
        """Propagation preserves the node count and returns ``hidden_dim``.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        num_layers : int
            Number of leapfrog steps.
        """
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_type=sheaf_type, num_layers=num_layers)
        out = model(x, edge_index)

        assert out.shape == (x.size(0), model.hidden_dim)
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("sheaf_type", SHEAF_TYPE_NAMES)
    def test_batched_graphs_stay_independent(self, sheaf_type):
        """The sheaf Laplacian is block diagonal, so batching is exact.

        This is what makes the inductive, graph-level setting of the challenge
        well defined: a batch must give the same answer as separate forwards.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        torch.manual_seed(1)
        first = Data(
            x=torch.randn(5, 12),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        )
        second = Data(
            x=torch.randn(4, 12),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        )
        batch = Batch.from_data_list([first, second])
        model = make_encoder(sheaf_type=sheaf_type)

        joint = model(batch.x, batch.edge_index, batch=batch.batch)
        separate = torch.cat(
            [
                model(first.x, first.edge_index),
                model(second.x, second.edge_index),
            ]
        )

        assert torch.allclose(joint, separate, atol=1e-5)

    @pytest.mark.parametrize("sheaf_type", SHEAF_TYPE_NAMES)
    def test_every_parameter_receives_gradient(self, sheaf_type):
        """No dead parameters: the leapfrog uses every weight it registers.

        Parameters
        ----------
        sheaf_type : str
            Restriction-map family under test.
        """
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_type=sheaf_type)
        model.train()
        model(x, edge_index).sum().backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} got no gradient"
            assert parameter.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_self_loops_are_removed(self):
        """Self-loops are not edges of a cellular sheaf and are stripped.

        The Laplacian builders pair every edge with its reverse, which a
        self-loop breaks, so they must never reach the builder.
        """
        x, edge_index = ring_graph()
        loops = torch.arange(x.size(0)).repeat(2, 1)
        model = make_encoder()

        with_loops = model(x, torch.cat([edge_index, loops], dim=1))

        assert torch.allclose(with_loops, model(x, edge_index), atol=1e-6)

    def test_directed_input_is_symmetrised(self):
        """A cellular sheaf lives on an undirected graph ([1], Section 2)."""
        x, edge_index = ring_graph()
        model = make_encoder()
        reversed_index = edge_index.flip(0)

        assert torch.allclose(
            model(x, edge_index), model(x, reversed_index), atol=1e-6
        )

    def test_graph_without_edges(self):
        """An edgeless graph has a zero Laplacian and must still propagate."""
        x, _ = ring_graph()
        model = make_encoder()
        out = model(x, torch.empty(2, 0, dtype=torch.long))

        assert out.shape == (x.size(0), model.hidden_dim)
        assert torch.all(torch.isfinite(out))

    def test_isolated_nodes(self):
        """Degree-zero nodes must not divide by zero when normalising."""
        x, _ = ring_graph()
        model = make_encoder(sheaf_type="diag")
        out = model(x, torch.tensor([[0, 1], [1, 0]]))

        assert torch.all(torch.isfinite(out))

    def test_divergence_is_reported(self):
        """A non-dissipative scheme can blow up, and must say so."""
        x, edge_index = ring_graph()
        model = make_encoder(num_layers=6, step_size=1e30)

        with pytest.raises(RuntimeError, match="diverged"):
            model(x, edge_index)

    def test_eval_is_deterministic_and_train_is_not(self):
        """Dropout is active in training mode only."""
        x, edge_index = ring_graph()
        model = NSPEncoder(
            input_dim=12, hidden_dim=10, dropout=0.5, input_dropout=0.5
        )

        model.eval()
        assert torch.allclose(model(x, edge_index), model(x, edge_index))

        model.train()
        torch.manual_seed(0)
        first = model(x, edge_index)
        torch.manual_seed(1)
        assert not torch.allclose(first, model(x, edge_index))

    @pytest.mark.parametrize("sheaf_act", ["id", "tanh", "elu"])
    def test_sheaf_activations(self, sheaf_act):
        """All supported sheaf-learner activations propagate.

        Parameters
        ----------
        sheaf_act : str
            Activation under test.
        """
        x, edge_index = ring_graph()
        model = make_encoder(sheaf_act=sheaf_act)

        assert torch.all(torch.isfinite(model(x, edge_index)))

    def test_step_size_scales_the_propagation_term(self):
        """``step_size`` is the leapfrog step ``h`` of [1], Table 2.

        One layer starts from zero velocity, so it reduces to
        :math:`\\mathbf{X}_1 = \\mathbf{X}_0 - h\\,\\sigma(\\cdot)` and the
        deviation from the frozen state must be exactly linear in ``h``.
        """
        x, edge_index = ring_graph()
        deltas = []
        for step_size in (0.25, 0.5):
            torch.manual_seed(3)
            model = make_encoder(num_layers=1, step_size=step_size)
            frozen = model.lin2(encode(model, x).reshape(x.size(0), -1))
            deltas.append(model(x, edge_index) - frozen)

        assert torch.allclose(2 * deltas[0], deltas[1], atol=1e-5)

    def test_unused_arguments_are_ignored(self):
        """Wrapper-supplied edge and batch tensors do not change the output."""
        x, edge_index = ring_graph()
        model = make_encoder()

        assert torch.allclose(
            model(x, edge_index),
            model(
                x,
                edge_index,
                edge_attr=torch.randn(edge_index.size(1), 4),
                edge_weight=torch.rand(edge_index.size(1)),
                batch=torch.zeros(x.size(0), dtype=torch.long),
                unexpected="ignored",
            ),
        )

    def test_learned_laplacian_is_stored_for_inspection(self):
        """Each layer keeps its transport maps, as the NSD utilities expect."""
        x, edge_index = ring_graph()
        model = make_encoder()
        model(x, edge_index)

        for learner in model.sheaf_learners:
            assert learner.L is not None
            assert not learner.L.requires_grad
