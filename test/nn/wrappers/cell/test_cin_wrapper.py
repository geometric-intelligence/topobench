"""Unit tests for CINWrapper."""

import pytest
import torch
import torch_geometric.data

from topobench.nn.backbones.cell.cin import CIN
from topobench.nn.wrappers.cell.cin_wrapper import CINWrapper

# ──────────────────────────────────────────────────────────────────────────── #
# A minimal synthetic cell complex
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture()
def cin_batch():
    """Minimal synthetic cell complex for wrapper tests.

    5 nodes, 6 edges, 2 rings. All features in R^8.
    """
    torch.manual_seed(42)
    N0, N1, N2, D = 5, 6, 2, 8

    x_0 = torch.randn(N0, D)
    x_1 = torch.randn(N1, D)
    x_2 = torch.randn(N2, D)

    adj0_idx = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]]
    )
    adjacency_0 = torch.sparse_coo_tensor(
        adj0_idx, torch.ones(adj0_idx.size(1)), size=(N0, N0)
    ).float()

    adj1_idx = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    adjacency_1 = torch.sparse_coo_tensor(
        adj1_idx, torch.ones(adj1_idx.size(1)), size=(N1, N1)
    ).float()

    b1_idx = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5],
        ]
    )
    incidence_1 = torch.sparse_coo_tensor(
        b1_idx, torch.ones(b1_idx.size(1)), size=(N0, N1)
    )

    b2_idx = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]])
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
# CINWrapper tests
# ──────────────────────────────────────────────────────────────────────────── #


class TestCINWrapper:
    """Tests for CINWrapper — forward method."""

    def _make_wrapper(self, b, n_layers=2):
        model = CIN(b._D, b._D, b._D, hid_channels=b._D, n_layers=n_layers)
        return CINWrapper(model, out_channels=b._D, num_cell_dimensions=3)

    def test_output_keys(self, cin_batch):
        """Wrapper must return all expected keys: labels, batch_0, x_0, x_1, x_2."""
        wrapper = self._make_wrapper(cin_batch)
        out = wrapper(cin_batch)
        for key in ("labels", "batch_0", "x_0", "x_1", "x_2"):
            assert key in out, f"Missing key: {key}"

    def test_output_shapes(self, cin_batch):
        """Output tensors must have correct spatial shapes."""
        b = cin_batch
        wrapper = self._make_wrapper(b)
        out = wrapper(b)
        assert out["x_0"].shape == (b._N0, b._D)
        assert out["x_1"].shape == (b._N1, b._D)
        assert out["x_2"].shape == (b._N2, b._D)

    def test_labels_and_batch_passthrough(self, cin_batch):
        """labels and batch_0 must be passed through unchanged."""
        b = cin_batch
        wrapper = self._make_wrapper(b)
        out = wrapper(b)
        assert torch.equal(out["labels"], b.y)
        assert torch.equal(out["batch_0"], b.batch_0)

    def test_adjacency_0_used_for_nodes_not_adjacency_1(self, cin_batch):
        """Node upper messages must use adjacency_0, not adjacency_1.

        This tests the fix to the original CWNWrapper bug where
        batch.adjacency_1 was incorrectly passed as adjacency_0.

        With n_layers=1, x_0 depends only on adjacency_0 (unchanged between
        the two runs) and batch.x_1 (unchanged), so zeroing adjacency_1 must
        leave x_0 identical while changing x_1 (which uses adjacency_1 for
        edge upper messages).
        """
        b = cin_batch
        b_zeroed = b.clone()
        b_zeroed.adjacency_1 = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            size=(b._N1, b._N1),
        ).float()

        model = CIN(b._D, b._D, b._D, hid_channels=b._D, n_layers=1)
        model.eval()
        wrapper = CINWrapper(model, out_channels=b._D, num_cell_dimensions=3)

        with torch.no_grad():
            out_orig = wrapper(b)
            out_zeroed = wrapper(b_zeroed)

        assert torch.allclose(out_orig["x_0"], out_zeroed["x_0"], atol=1e-6), (
            "Node outputs differ when adjacency_1 was zeroed — wrapper is "
            "incorrectly using adjacency_1 for node messages"
        )
        assert not torch.allclose(
            out_orig["x_1"], out_zeroed["x_1"], atol=1e-6
        ), (
            "Edge outputs identical after zeroing adjacency_1 — adjacency_1 "
            "is not being used for edge upper messages"
        )

    def test_no_2cells_graceful(self, cin_batch):
        """Wrapper must handle a batch without x_2, adjacency_1, incidence_2."""
        b = cin_batch.clone()
        del b.x_2
        del b.adjacency_1
        del b.incidence_2

        model = CIN(b._D, b._D, 0, hid_channels=b._D, n_layers=2)
        wrapper = CINWrapper(model, out_channels=b._D, num_cell_dimensions=2)
        out = wrapper(b)

        assert "x_0" in out
        assert "x_1" in out
        assert out.get("x_2") is None
