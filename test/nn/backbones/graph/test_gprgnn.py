"""Unit tests for the GPR-GNN backbone."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.graph.gprgnn import GPRGNN, GPRProp
from topobench.nn.wrappers.graph import GNNWrapper


def _make_batch(x, edge_index):
    """Build a minimal single-graph batch for the GNN wrapper.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix.
    edge_index : torch.Tensor
        Graph connectivity in COO format.

    Returns
    -------
    torch_geometric.data.Data
        A batch consumable by :class:`GNNWrapper`.
    """
    return torch_geometric.data.Data(
        x_0=x,
        x=x,
        y=torch.randint(0, 2, (x.shape[0],)),
        edge_index=edge_index,
        batch_0=torch.zeros(x.shape[0], dtype=torch.long),
    )


@pytest.mark.parametrize("init", ["SGC", "PPR", "NPPR", "Random"])
def test_gprgnn_forward(random_graph_input, init):
    """GPR-GNN returns node embeddings with the hidden dimension.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    init : str
        GPR coefficient initialization scheme under test.
    """
    x, _, _, edges_1, _ = random_graph_input
    # The wrapper adds a residual (backbone output + input x_0), so the
    # hidden dimension must match the input feature dimension here.
    hidden = x.shape[1]
    model = GPRGNN(x.shape[1], hidden, K=4, alpha=0.1, init=init)
    wrapper = GNNWrapper(model, out_channels=hidden, num_cell_dimensions=1)

    _ = wrapper.__repr__()
    _ = model.prop1.__repr__()

    out = wrapper(_make_batch(x, edges_1))
    assert out["x_0"].shape == (x.shape[0], hidden)


def test_gprgnn_warm_start_and_reset(random_graph_input):
    """Warm-start init accepts provided coefficients and survives reset.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, _, _ = random_graph_input
    K = 4
    gamma = torch.ones(K + 1) / (K + 1)
    model = GPRGNN(x.shape[1], 8, K=K, alpha=0.1, init="WS", gamma=gamma)
    torch.testing.assert_close(model.prop1.temp.detach(), gamma)

    model.reset_parameters()
    torch.testing.assert_close(model.prop1.temp.detach(), gamma)


def test_gprgnn_dprate_zero(random_graph_input):
    """The dprate=0 branch runs without dropping the hidden features.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GPRGNN(x.shape[1], 8, K=3, dprate=0.0)
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 8)


def test_gprprop_k0_is_identity(random_graph_input):
    """With K=0 and PPR init, propagation reduces to the identity.

    PPR sets gamma_0 = 1 when K=0, and there are no hops, so the layer
    must return its input unchanged (Z = gamma_0 * H = H).

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    prop = GPRProp(K=0, alpha=0.1, init="PPR")
    torch.testing.assert_close(prop(x, edges_1), x)


def test_gprprop_sgc_init_is_single_hop():
    """SGC init places all mass on a single hop (a pure k-hop filter).

    This is the limiting case where GPR reduces to SGC: gamma is a
    one-hot vector at the hop given by ``alpha``.
    """
    prop = GPRProp(K=5, alpha=2, init="SGC")
    expected = torch.zeros(6)
    expected[2] = 1.0
    torch.testing.assert_close(prop.temp.detach(), expected)


def test_gprprop_ppr_reduces_to_appnp(random_graph_input):
    """With PPR init, GPRProp matches PyG's official APPNP bit-for-bit.

    APPNP's K-step personalized-PageRank propagation expands to
    :math:`\\sum_k \\alpha(1-\\alpha)^k \\tilde{A}^k H` with the final hop
    weighted :math:`(1-\\alpha)^K` -- exactly GPRProp's PPR-initialized
    coefficients before any training. Agreement with PyG's APPNP
    validates the gcn-normalization, hop recursion, and aggregation
    against a trusted external reference (the GPR analog of GATE's
    GATv2 reduction test).

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    from torch_geometric.nn import APPNP

    x, _, _, edges_1, _ = random_graph_input
    K, alpha = 10, 0.1
    prop = GPRProp(K=K, alpha=alpha, init="PPR")
    ref = APPNP(K=K, alpha=alpha, dropout=0.0)
    torch.testing.assert_close(
        prop(x, edges_1), ref(x, edges_1), rtol=1e-5, atol=1e-5
    )


def test_gprgnn_permutation_equivariance(random_graph_input):
    """Relabeling the nodes permutes the outputs identically.

    Permutation equivariance is the defining symmetry of a message-
    passing GNN; the GPR propagation (symmetric normalized adjacency)
    must preserve it.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GPRGNN(x.shape[1], x.shape[1], K=4, dropout=0.0, dprate=0.0)
    model.eval()

    perm = torch.randperm(x.shape[0])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(x.shape[0])

    out = model(x, edges_1)
    out_perm = model(x[perm], inv[edges_1])
    torch.testing.assert_close(out_perm, out[perm], rtol=1e-4, atol=1e-4)


def test_gprprop_coefficients_are_learnable(random_graph_input):
    """The GPR coefficients receive gradients (they are learned).

    This is the whole point of *Generalized* PageRank: the per-hop
    weights are trainable, so a backward pass must populate temp.grad.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GPRGNN(x.shape[1], x.shape[1], K=4, dropout=0.0, dprate=0.0)
    model(x, edges_1).sum().backward()
    assert model.prop1.temp.grad is not None
    assert model.prop1.temp.grad.abs().sum() > 0


def test_gprprop_invalid_init():
    """An unknown init scheme raises a ValueError."""
    with pytest.raises(ValueError):
        GPRProp(K=4, alpha=0.1, init="nope")


def test_gprprop_warm_start_requires_gamma():
    """WS init without coefficients raises a ValueError."""
    with pytest.raises(ValueError):
        GPRProp(K=4, alpha=0.1, init="WS", gamma=None)
