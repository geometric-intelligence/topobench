"""Unit tests for the CIN (Cell Isomorphism Network) backbone."""

import pytest
import torch
import torch_geometric.data

from topobench.nn.backbones.cell.cin import CIN, CINLayer

# ──────────────────────────────────────────────────────────────────────────── #
# A minimal synthetic cell complex
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture()
def cell_complex_batch():
    """Minimal synthetic cell complex batch for testing.

    Complex: 5 nodes, 6 edges, 2 rings.  All features in R^8.
    The connectivity mirrors a simple graph with two 3-cycles sharing an edge.
    """
    torch.manual_seed(0)
    N0, N1, N2, D = 5, 6, 2, 8

    x_0 = torch.randn(N0, D)
    x_1 = torch.randn(N1, D)
    x_2 = torch.randn(N2, D)

    # adjacency_0: [N0, N0] sparse — upper-adjacency for nodes (A_up,0)
    # A_up,0[i,j] = 1 if nodes i and j share a common edge
    adj0_rows = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 0])
    adj0_cols = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3, 0, 4])
    adjacency_0 = torch.sparse_coo_tensor(
        torch.stack([adj0_rows, adj0_cols]),
        torch.ones(adj0_rows.size(0)),
        size=(N0, N0),
    ).float()

    # adjacency_1: [N1, N1] sparse — upper-adjacency for edges (A_up,1)
    # A_up,1[i,j] = 1 if edges i and j share a common ring
    adj1_rows = torch.tensor([0, 1, 2, 3, 4, 5])
    adj1_cols = torch.tensor([1, 0, 3, 2, 5, 4])
    adjacency_1 = torch.sparse_coo_tensor(
        torch.stack([adj1_rows, adj1_cols]),
        torch.ones(adj1_rows.size(0)),
        size=(N1, N1),
    ).float()

    # incidence_1: B1 [N0, N1] sparse; B1[v,e]=1 iff v ∈ boundary(edge e)
    b1_idx = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5],
        ]
    )
    incidence_1 = torch.sparse_coo_tensor(
        b1_idx, torch.ones(b1_idx.size(1)), size=(N0, N1)
    )

    # incidence_2: B2 [N1, N2] sparse; B2[e,r]=1 iff e ∈ boundary(ring r)
    b2_idx = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    incidence_2 = torch.sparse_coo_tensor(
        b2_idx, torch.ones(b2_idx.size(1)), size=(N1, N2)
    )

    batch = torch_geometric.data.Data(
        x_0=x_0,
        x_1=x_1,
        x_2=x_2,
        adjacency_0=adjacency_0,
        adjacency_1=adjacency_1,
        incidence_1=incidence_1,
        incidence_2=incidence_2,
        y=torch.zeros(1, dtype=torch.long),
        batch_0=torch.zeros(N0, dtype=torch.long),
    )
    batch._N0, batch._N1, batch._N2, batch._D = N0, N1, N2, D
    return batch


# ──────────────────────────────────────────────────────────────────────────── #
# CINLayer tests
# ──────────────────────────────────────────────────────────────────────────── #


class TestCINLayer:
    """Tests for CINLayer — single message passing step."""

    def _make_layer(self, d=8, out=16, has_2_cells=True):
        return CINLayer(
            in_channels_0=d,
            in_channels_1=d,
            in_channels_2=d if has_2_cells else 0,
            out_channels=out,
        )

    def test_output_shapes(self, cell_complex_batch):
        """All output tensors must have shape (N_k, out_channels)."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        out0, out1, out2 = layer(
            b.x_0,
            b.x_1,
            b.x_2,
            b.adjacency_0,
            b.adjacency_1,
            b.incidence_1,
            b.incidence_2,
        )
        assert out0.shape == (b._N0, 16)
        assert out1.shape == (b._N1, 16)
        assert out2.shape == (b._N2, 16)

    def test_no_2cells_returns_none(self, cell_complex_batch):
        """Without 2-cells, x_2_new must be None."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16, has_2_cells=False)
        out0, out1, out2 = layer(
            b.x_0,
            b.x_1,
            None,
            b.adjacency_0,
            None,
            b.incidence_1,
            None,
        )
        assert out0.shape == (b._N0, 16)
        assert out1.shape == (b._N1, 16)
        assert out2 is None

    def test_gradients_all_streams(self, cell_complex_batch):
        """Gradients must flow to all three cell-feature inputs."""
        b = cell_complex_batch
        x0 = b.x_0.requires_grad_(True)
        x1 = b.x_1.requires_grad_(True)
        x2 = b.x_2.requires_grad_(True)
        layer = self._make_layer(d=b._D, out=16)
        out0, out1, out2 = layer(
            x0,
            x1,
            x2,
            b.adjacency_0,
            b.adjacency_1,
            b.incidence_1,
            b.incidence_2,
        )
        (out0.sum() + out1.sum() + out2.sum()).backward()
        assert x0.grad is not None, "No gradient to x_0"
        assert x1.grad is not None, "No gradient to x_1"
        assert x2.grad is not None, "No gradient to x_2"

    def test_2cell_boundary_depends_on_edges(self, cell_complex_batch):
        """2-cell (ring) features must depend on 1-cell (edge) features.

        Paper Section 4 and Figure 6 show rings receive boundary messages from
        their constituent edges: m_B(ring) = AGG_{e ∈ B(ring)} MLP_B(h_e).
        Theorem 7 only drops coboundary and lower-adjacency — boundary messages
        are always kept.
        """
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        layer.eval()

        with torch.no_grad():
            _, _, out2_orig = layer(
                b.x_0,
                b.x_1,
                b.x_2,
                b.adjacency_0,
                b.adjacency_1,
                b.incidence_1,
                b.incidence_2,
            )
            x1_perturbed = b.x_1 + 10.0
            _, _, out2_perturbed = layer(
                b.x_0,
                x1_perturbed,
                b.x_2,
                b.adjacency_0,
                b.adjacency_1,
                b.incidence_1,
                b.incidence_2,
            )

        assert not torch.allclose(out2_orig, out2_perturbed, atol=1e-5), (
            "2-cell output did not change when edge features changed — "
            "boundary message m_B(ring)←edges is missing"
        )

    def test_1cell_boundary_depends_on_nodes(self, cell_complex_batch):
        """1-cell (edge) features must depend on 0-cell (node) features.

        Paper Section 4 and Figure 6 (orange arrows) show bonds receive
        boundary messages from their endpoint atoms:
        m_B(edge) = AGG_{v ∈ B(edge)} MLP_B(h_v) = B1^T @ msg_boundary_1(x_0).
        This specifically tests the fix for the missing node→edge boundary message.
        """
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        layer.eval()

        with torch.no_grad():
            _, out1_orig, _ = layer(
                b.x_0,
                b.x_1,
                b.x_2,
                b.adjacency_0,
                b.adjacency_1,
                b.incidence_1,
                b.incidence_2,
            )
            x0_perturbed = b.x_0 + 10.0
            _, out1_perturbed, _ = layer(
                x0_perturbed,
                b.x_1,
                b.x_2,
                b.adjacency_0,
                b.adjacency_1,
                b.incidence_1,
                b.incidence_2,
            )

        assert not torch.allclose(out1_orig, out1_perturbed, atol=1e-5), (
            "1-cell output did not change when node features changed — "
            "boundary message m_B(edge)←nodes (B1^T @ msg(x_0)) is missing"
        )

    def test_boundary_aggregate_non_zero(self, cell_complex_batch):
        """Boundary aggregation (B1^T @ msg(x_0)) produces non-zero output."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        # B1^T: transpose=False, x_sender=x_0, incidence=B1[N0,N1]
        # Result shape [N1, out] — boundary message from nodes to edges
        agg = layer._incidence_aggregate(
            x_sender=b.x_0,
            incidence=b.incidence_1,
            msg_fn=layer.msg_boundary_1,
            transpose=False,
            n_receivers=b._N1,
        )
        assert agg.shape == (b._N1, 16)
        assert agg.abs().sum() > 0

    def test_incidence_aggregate_dense(self, cell_complex_batch):
        """_incidence_aggregate must also handle dense (non-sparse) matrices.

        Covers the dense branch of the helper and checks it matches the
        sparse result for the same connectivity.
        """
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        dense_incidence = b.incidence_1.to_dense()
        # Test the bridge-edge path (transpose=True, B1 @ msg(x_1))
        agg_dense = layer._incidence_aggregate(
            x_sender=b.x_1,
            incidence=dense_incidence,
            msg_fn=layer.msg_bridge_e,
            transpose=True,
            n_receivers=b._N0,
        )
        agg_sparse = layer._incidence_aggregate(
            x_sender=b.x_1,
            incidence=b.incidence_1,
            msg_fn=layer.msg_bridge_e,
            transpose=True,
            n_receivers=b._N0,
        )
        assert agg_dense.shape == (b._N0, 16)
        assert torch.allclose(agg_dense, agg_sparse, atol=1e-5)

    def test_incidence_aggregate_dense_transpose_false(
        self, cell_complex_batch
    ):
        """Dense path with transpose=False (edges→rings via B2^T)."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        dense_b2 = b.incidence_2.to_dense()
        agg = layer._incidence_aggregate(
            x_sender=b.x_1,
            incidence=dense_b2,
            msg_fn=layer.msg_B_2,
            transpose=False,
            n_receivers=b._N2,
        )
        assert agg.shape == (b._N2, 16)

    def test_upper_aggregate_dense(self, cell_complex_batch):
        """_upper_aggregate must also handle dense adjacency matrices."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        dense_adj = b.adjacency_0.to_dense()
        agg_dense = layer._upper_aggregate(
            x_neighbor=b.x_0,
            adjacency=dense_adj,
            msg_fn=layer.msg_up_0,
            n_receivers=b._N0,
        )
        agg_sparse = layer._upper_aggregate(
            x_neighbor=b.x_0,
            adjacency=b.adjacency_0,
            msg_fn=layer.msg_up_0,
            n_receivers=b._N0,
        )
        assert agg_dense.shape == (b._N0, 16)
        assert torch.allclose(agg_dense, agg_sparse, atol=1e-5)

    def test_upper_aggregate_zero_on_empty_adj(self, cell_complex_batch):
        """Upper aggregation must return zeros when adjacency has no edges."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        empty_adj = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            size=(b._N0, b._N0),
        ).float()
        agg = layer._upper_aggregate(
            x_neighbor=b.x_0,
            adjacency=empty_adj,
            msg_fn=layer.msg_up_0,
            n_receivers=b._N0,
        )
        assert agg.shape == (b._N0, 16)
        assert agg.abs().sum() == 0.0

    def test_upper_aggregate_zero_on_none_adj(self, cell_complex_batch):
        """Upper aggregation must return zeros when adjacency is None."""
        b = cell_complex_batch
        layer = self._make_layer(d=b._D, out=16)
        agg = layer._upper_aggregate(
            x_neighbor=b.x_0,
            adjacency=None,
            msg_fn=layer.msg_up_0,
            n_receivers=b._N0,
        )
        assert agg.shape == (b._N0, 16)
        assert agg.abs().sum() == 0.0

    def test_make_update_mlp(self):
        """Per-stream update MLP must map (N, C) -> (N, C).

        Mirrors ``update_up_nn`` / ``update_boundaries_nn`` in the reference
        implementation (cwn-main, ``mp/layers.py::SparseCINConv``): two
        Linear layers each followed by BatchNorm and ReLU.
        """
        mlp = CINLayer._make_update_mlp(16)
        out = mlp(torch.randn(4, 16))
        assert out.shape == (4, 16)
        linears = [m for m in mlp if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 2

    def test_make_combine_mlp(self):
        """Combine MLP must map the concatenated streams (N, 2C) -> (N, C).

        Mirrors ``combine_nn`` in the reference implementation (cwn-main,
        ``mp/layers.py::SparseCINConv``).
        """
        mlp = CINLayer._make_combine_mlp(16)
        out = mlp(torch.randn(4, 32))
        assert out.shape == (4, 16)

    def test_two_step_update_separate_eps_streams(self, cell_complex_batch):
        """Each rank must carry two GIN eps parameters (up and boundary).

        The two-step update of Appendix E.3 (and
        ``SparseCINCochainConv.forward`` in cwn-main, lines 184-199) applies
        separate (1+eps)-weighted updates to the upper and boundary
        multi-sets before combining them.
        """
        layer = self._make_layer(d=8, out=16)
        for name in [
            "eps_up_0",
            "eps_B_0",
            "eps_up_1",
            "eps_B_1",
            "eps_up_2",
            "eps_B_2",
        ]:
            assert hasattr(layer, name), f"Missing eps parameter: {name}"
        # Streams must be combined by a dedicated MLP per rank
        for name in ["combine_0", "combine_1", "combine_2"]:
            assert hasattr(layer, name), f"Missing combine MLP: {name}"


# ──────────────────────────────────────────────────────────────────────────── #
# CIN tests
# ──────────────────────────────────────────────────────────────────────────── #


class TestCIN:
    """Tests for the full CIN backbone (stacked CINLayers)."""

    def _model(self, b, n_layers=3, dropout=0.0):
        return CIN(
            in_channels_0=b._D,
            in_channels_1=b._D,
            in_channels_2=b._D,
            hid_channels=16,
            n_layers=n_layers,
            dropout=dropout,
        )

    def test_output_shapes(self, cell_complex_batch):
        """CIN must return tensors of shape (N_k, hid_channels)."""
        b = cell_complex_batch
        model = self._model(b)
        out0, out1, out2 = model(
            b.x_0,
            b.x_1,
            b.x_2,
            b.adjacency_0,
            b.adjacency_1,
            b.incidence_1,
            b.incidence_2,
        )
        assert out0.shape == (b._N0, 16)
        assert out1.shape == (b._N1, 16)
        assert out2.shape == (b._N2, 16)

    def test_single_layer(self, cell_complex_batch):
        """A single CIN layer must still run and produce correct shapes."""
        b = cell_complex_batch
        model = self._model(b, n_layers=1)
        out0, out1, out2 = model(
            b.x_0,
            b.x_1,
            b.x_2,
            b.adjacency_0,
            b.adjacency_1,
            b.incidence_1,
            b.incidence_2,
        )
        assert out0.shape == (b._N0, 16)

    def test_all_dims_updated_across_layers(self, cell_complex_batch):
        """After forward pass, all three cell-dim outputs must differ from input
        projections — verifying that 0-cells and 2-cells are genuinely updated
        (fixing the TopoModelX CWN bug where only x_1 was updated in the loop).
        """
        b = cell_complex_batch
        model = self._model(b, n_layers=2)
        model.eval()

        out0, out1, out2 = model(
            b.x_0,
            b.x_1,
            b.x_2,
            b.adjacency_0,
            b.adjacency_1,
            b.incidence_1,
            b.incidence_2,
        )

        proj0_only = torch.nn.functional.elu(model.proj_0(b.x_0))
        proj2_only = torch.nn.functional.elu(model.proj_2(b.x_2))

        assert not torch.allclose(out0, proj0_only, atol=1e-5), (
            "x_0 was NOT updated by message passing (TopoModelX bug still present)"
        )
        assert not torch.allclose(out2, proj2_only, atol=1e-5), (
            "x_2 was NOT updated by message passing (TopoModelX bug still present)"
        )

    def test_no_2cells(self, cell_complex_batch):
        """CIN must handle absence of 2-cells (x_2=None) gracefully."""
        b = cell_complex_batch
        model = CIN(
            in_channels_0=b._D,
            in_channels_1=b._D,
            in_channels_2=0,
            hid_channels=16,
            n_layers=2,
        )
        out0, out1, out2 = model(
            b.x_0,
            b.x_1,
            None,
            b.adjacency_0,
            None,
            b.incidence_1,
            None,
        )
        assert out0.shape == (b._N0, 16)
        assert out1.shape == (b._N1, 16)
        assert out2 is None

    def test_end_to_end_backprop(self, cell_complex_batch):
        """Loss on all outputs must allow gradients to reach all parameters."""
        b = cell_complex_batch
        model = self._model(b)
        out0, out1, out2 = model(
            b.x_0,
            b.x_1,
            b.x_2,
            b.adjacency_0,
            b.adjacency_1,
            b.incidence_1,
            b.incidence_2,
        )
        loss = out0.mean() + out1.mean() + out2.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for parameter: {name}"

    def test_eval_deterministic(self, cell_complex_batch):
        """In eval mode, two identical forward passes must produce identical outputs."""
        b = cell_complex_batch
        model = self._model(b, dropout=0.5)
        model.eval()
        with torch.no_grad():
            out_a = model(
                b.x_0,
                b.x_1,
                b.x_2,
                b.adjacency_0,
                b.adjacency_1,
                b.incidence_1,
                b.incidence_2,
            )
            out_b = model(
                b.x_0,
                b.x_1,
                b.x_2,
                b.adjacency_0,
                b.adjacency_1,
                b.incidence_1,
                b.incidence_2,
            )
        assert torch.allclose(out_a[0], out_b[0])
        assert torch.allclose(out_a[1], out_b[1])
        assert torch.allclose(out_a[2], out_b[2])
