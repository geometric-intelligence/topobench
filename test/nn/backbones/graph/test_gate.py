"""Unit tests for the GATE backbone."""

import pytest
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import add_self_loops, remove_self_loops

from topobench.nn.backbones.graph.gate import GATE, GATEConv
from topobench.nn.wrappers.graph import GNNWrapper


def _dense_gate_reference(conv, x, edge_index):
    """Recompute GATEConv's output with explicit dense per-node loops.

    Independent implementation of the GATE update (Eq. 1/2/4, no
    ``MessagePassing``); agreement with the layer is the parity check on
    attention, the self/neighbour split, softmax scope, and aggregation.

    Parameters
    ----------
    conv : GATEConv
        The layer under test (used for its weights and flags).
    x : torch.Tensor
        Node features.
    edge_index : torch.Tensor
        Graph connectivity (self-loops added internally, as in forward).

    Returns
    -------
    torch.Tensor
        The reference output, same shape as ``conv(x, edge_index)``.
    """
    conv.eval()
    h, c = conv.heads, conv.out_channels
    x_l = conv.lin_l(x).view(-1, h, c)
    x_r = conv.lin_r(x).view(-1, h, c)
    ei, _ = remove_self_loops(edge_index)
    ei, _ = add_self_loops(ei, num_nodes=x.size(0))
    src, dst = ei[0], ei[1]

    pre = F.leaky_relu(x_l[dst] + x_r[src], conv.negative_slope)
    att, att2 = conv.att.squeeze(0), conv.att2.squeeze(0)
    is_self = (src == dst).view(-1, 1)
    logit = torch.where(is_self, (pre * att2).sum(-1), (pre * att).sum(-1))

    out = torch.zeros(x.size(0), h, c)
    for i in range(x.size(0)):
        m = dst == i
        a = torch.softmax(logit[m], dim=0)  # (deg, H)
        out[i] = (x_r[src[m]] * a.unsqueeze(-1)).sum(0)  # value = U h_u
    return out.reshape(x.size(0), h * c) if conv.concat else out.mean(1)


@pytest.mark.parametrize(
    "share_att,concat", [(False, True), (True, True), (False, False)]
)
def test_gate_parity_dense(random_graph_input, share_att, concat):
    """GATEConv matches an independent dense recomputation (fidelity).

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    share_att : bool
        Whether self/neighbour edges share one attention vector.
    concat : bool
        Whether heads are concatenated.
    """
    x, _, _, edges_1, _ = random_graph_input
    conv = GATEConv(
        x.shape[1], 5, heads=2, share_att=share_att,
        concat=concat, dropout=0.0,
    )
    # Random (non-zero) attention so the test exercises real coefficients.
    with torch.no_grad():
        conv.att.normal_()
        if not share_att:
            conv.att2.normal_()
    conv.eval()
    torch.testing.assert_close(
        conv(x, edges_1), _dense_gate_reference(conv, x, edges_1),
        rtol=1e-5, atol=1e-5,
    )


def test_gate_reduces_to_pyg_gatv2(random_graph_input):
    """In its GATv2 special case, GATEConv matches PyG's official GATv2Conv.

    With one shared attention vector, GATE collapses to standard GATv2
    (the value is already the source transform for every edge). Copying
    PyG's weights and asserting bit-for-bit agreement validates our
    attention routing (i/j assignment, softmax scope, self-loop handling,
    aggregation) against a trusted reference implementation.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    from torch_geometric.nn import GATv2Conv

    x, _, _, edges_1, _ = random_graph_input
    ei, _ = remove_self_loops(edges_1)  # both add their own self-loops
    heads, c = 2, 5
    ref = GATv2Conv(
        x.shape[1], c, heads=heads, concat=True,
        add_self_loops=True, bias=False, share_weights=False,
    ).eval()
    ours = GATEConv(
        x.shape[1], c, heads=heads, concat=True,
        share_att=True, dropout=0.0,
    ).eval()

    # Copy by ROLE, not name: PyG's lin_l is the source transform (and
    # value), lin_r the target transform; ours uses the paper's opposite
    # naming (lin_r = source = value, lin_l = target).
    with torch.no_grad():
        ours.lin_r.weight.copy_(ref.lin_l.weight)  # source / value (U)
        ours.lin_l.weight.copy_(ref.lin_r.weight)  # target (V)
        ours.lin_l.bias.zero_()
        ours.lin_r.bias.zero_()
        ours.att.copy_(ref.att)  # att2 is att (share_att=True)

    torch.testing.assert_close(
        ours(x, ei), ref(x, ei), rtol=1e-5, atol=1e-5
    )


def test_gate_uniform_attention_at_init(random_graph_input):
    """At init (zero attention) GATE is uniform mean aggregation (Thm 4.3).

    Zero-initialized attention vectors make every logit zero, so softmax
    is uniform over the closed neighbourhood and the layer reduces to a
    mean of the source transform ``U h_u`` -- the paper's "no initial
    inductive bias" property. (Also covers isolated nodes: a node whose
    only edge is its self-loop returns its own transform.)

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    conv = GATEConv(x.shape[1], 6, heads=1, dropout=0.0)  # att zero at init
    conv.eval()
    out = conv(x, edges_1)

    x_r = conv.lin_r(x)
    ei, _ = remove_self_loops(edges_1)
    ei, _ = add_self_loops(ei, num_nodes=x.shape[0])
    src, dst = ei[0], ei[1]
    expected = torch.stack(
        [x_r[src[dst == i]].mean(0) for i in range(x.shape[0])]
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_gate_share_att_ties_vectors():
    """share_att=True ties the two attention vectors; False keeps them apart."""
    tied = GATEConv(8, 4, share_att=True)
    assert tied.att2 is tied.att
    untied = GATEConv(8, 4, share_att=False)
    assert untied.att2 is not untied.att


def test_gate_wrapper_forward(random_graph_input):
    """GATE returns node embeddings of the hidden dimension via the wrapper.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    hidden = x.shape[1]
    model = GATE(x.shape[1], hidden, num_layers=2, heads=2)
    wrapper = GNNWrapper(model, out_channels=hidden, num_cell_dimensions=1)
    _ = wrapper.__repr__()
    _ = model.convs[0].__repr__()
    batch = torch_geometric.data.Data(
        x_0=x, x=x, y=torch.randint(0, 2, (x.shape[0],)),
        edge_index=edges_1,
        batch_0=torch.zeros(x.shape[0], dtype=torch.long),
    )
    out = wrapper(batch)
    assert out["x_0"].shape == (x.shape[0], hidden)


def test_gate_permutation_equivariance(random_graph_input):
    """Relabeling nodes permutes the GATE outputs identically.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GATE(x.shape[1], x.shape[1], num_layers=2, heads=2, dropout=0.0)
    # Non-zero attention so equivariance is tested on real attention.
    for conv in model.convs:
        with torch.no_grad():
            conv.att.normal_()
            conv.att2.normal_()
    model.eval()
    perm = torch.randperm(x.shape[0])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(x.shape[0])
    out = model(x, edges_1)
    out_perm = model(x[perm], inv[edges_1])
    torch.testing.assert_close(out_perm, out[perm], rtol=1e-4, atol=1e-4)


def test_gate_parameters_learnable(random_graph_input):
    """A backward pass populates gradients for weights and attention.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GATE(x.shape[1], 8, num_layers=2, heads=2)
    model(x, edges_1).sum().backward()
    assert model.convs[0].att.grad is not None
    assert model.convs[0].lin_l.weight.grad is not None
    model.reset_parameters()
