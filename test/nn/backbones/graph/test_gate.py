"""Unit tests for the GATE backbone."""

import pytest
import torch
import torch_geometric
from torch_geometric.utils import add_self_loops, remove_self_loops

from topobench.nn.backbones.graph.gate import GATE, GATEConv
from topobench.nn.wrappers.graph import GNNWrapper


def _dense_gate_reference(conv, x, edge_index):
    """Recompute GATEConv's output with explicit dense per-node loops.

    This is an independent implementation of the documented GATE update
    (no ``MessagePassing``/``propagate``); agreement with the layer is the
    parity check on attention, masking, softmax scope, and aggregation.

    Parameters
    ----------
    conv : GATEConv
        The layer under test (used for its weights and flags).
    x : torch.Tensor
        Node features.
    edge_index : torch.Tensor
        Graph connectivity (self-loops are added internally, matching
        ``GATEConv.forward``).

    Returns
    -------
    torch.Tensor
        The reference output, same shape as ``conv(x, edge_index)``.
    """
    conv.eval()
    h, c = conv.heads, conv.out_channels
    x_l = conv.lin_l(x).view(-1, h, c)
    x_r = conv.lin_r(x).view(-1, h, c)
    x_s = conv.lin_s(x).view(-1, h, c)

    ei, _ = remove_self_loops(edge_index)
    ei, _ = add_self_loops(ei, num_nodes=x.size(0))
    src, dst = ei[0], ei[1]

    e = torch.nn.functional.leaky_relu(
        x_l[dst] + x_r[src], conv.negative_slope
    )  # (E, H, C)
    att = conv.att.squeeze(0)
    att2 = conv.att2.squeeze(0)
    is_self = (src == dst).view(-1, 1)
    logit = torch.where(
        is_self, (e * att2).sum(-1), (e * att).sum(-1)
    )  # (E, H)

    out = torch.zeros(x.size(0), h, c)
    for i in range(x.size(0)):
        mask = dst == i
        a = torch.softmax(logit[mask], dim=0)  # (deg, H)
        s = src[mask]
        self_e = (s == i).view(-1, 1, 1).to(x.dtype)
        val = torch.where(self_e.bool(), x_s[s], x_r[s])  # (deg, H, C)
        if conv.has_omega:
            omega = conv.omega.view(h, c)
            contrib = val * (self_e - omega * (self_e - a.unsqueeze(-1)))
        else:
            contrib = val * a.unsqueeze(-1)
        out[i] = contrib.sum(0)
    return out.reshape(x.size(0), h * c) if conv.concat else out.mean(1)


@pytest.mark.parametrize(
    "share_att,has_omega,concat",
    [(False, True, True), (False, False, True), (True, True, True),
     (False, True, False)],
)
def test_gate_parity_dense(random_graph_input, share_att, has_omega, concat):
    """GATEConv matches an independent dense recomputation (fidelity).

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    share_att : bool
        Whether self/neighbour edges share one attention vector.
    has_omega : bool
        Whether the learned self/neighbour gate is active.
    concat : bool
        Whether heads are concatenated.
    """
    x, _, _, edges_1, _ = random_graph_input
    conv = GATEConv(
        x.shape[1], 5, heads=2, share_att=share_att,
        has_omega=has_omega, concat=concat, dropout=0.0,
    )
    conv.eval()
    ours = conv(x, edges_1)
    ref = _dense_gate_reference(conv, x, edges_1)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_gate_reduces_to_pyg_gatv2(random_graph_input):
    """In its GATv2 special case, GATEConv matches PyG's official GATv2Conv.

    This is an external fidelity check: with one shared attention vector,
    no omega, and the self-value tied to ``lin_r``, GATE collapses to
    standard GATv2. Copying PyG's weights and asserting bit-for-bit
    agreement validates our attention routing (the i/j assignment,
    softmax scope, self-loop handling, and aggregation) against a trusted
    reference implementation.

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
        share_att=True, has_omega=False, dropout=0.0,
    ).eval()

    # Copy by ROLE, not by name: PyG's lin_l is the source transform (and
    # value), lin_r the target transform; ours uses the paper's opposite
    # naming (lin_r = source = value, lin_l = target).
    with torch.no_grad():
        ours.lin_r.weight.copy_(ref.lin_l.weight)  # source / neighbour value
        ours.lin_s.weight.copy_(ref.lin_l.weight)  # self value (= source W)
        ours.lin_l.weight.copy_(ref.lin_r.weight)  # target
        ours.lin_l.bias.zero_()
        ours.lin_r.bias.zero_()
        ours.lin_s.bias.zero_()
        ours.att.copy_(ref.att)  # att2 is att (share_att=True)

    torch.testing.assert_close(
        ours(x, ei), ref(x, ei), rtol=1e-5, atol=1e-5
    )


def test_gate_omega_zero_switches_off_neighbours(random_graph_input):
    """omega=0 fully gates off neighbours (GATE's defining claim).

    With ``omega = 0`` the gate zeroes every neighbour message and passes
    the self-value through unchanged, so each node's output collapses
    exactly to its own value transform ``lin_s(x)``. This is a direct,
    closed-form check of the paper's "keep out intrusive neighbours"
    mechanism, independent of the formula-level parity tests.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    conv = GATEConv(x.shape[1], 6, heads=1, has_omega=True, dropout=0.0)
    conv.eval()
    with torch.no_grad():
        conv.omega.zero_()
    torch.testing.assert_close(
        conv(x, edges_1), conv.lin_s(x), rtol=1e-5, atol=1e-5
    )


def test_gate_share_att_ties_vectors():
    """With share_att=True the two attention vectors are the same object."""
    conv = GATEConv(8, 4, share_att=True)
    assert conv.att2 is conv.att
    conv2 = GATEConv(8, 4, share_att=False)
    assert conv2.att2 is not conv2.att


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
    model.eval()
    perm = torch.randperm(x.shape[0])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(x.shape[0])
    out = model(x, edges_1)
    out_perm = model(x[perm], inv[edges_1])
    torch.testing.assert_close(out_perm, out[perm], rtol=1e-4, atol=1e-4)


def test_gate_parameters_learnable(random_graph_input):
    """A backward pass populates gradients, including omega.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GATE(x.shape[1], 8, num_layers=2, heads=2, has_omega=True)
    model(x, edges_1).sum().backward()
    assert model.convs[0].omega.grad is not None
    assert model.convs[0].att.grad is not None
    model.reset_parameters()
