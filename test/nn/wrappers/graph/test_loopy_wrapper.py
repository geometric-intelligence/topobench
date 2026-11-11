"""Unit tests for the LoopyWrapper."""

import torch
from torch_geometric.data import Data

from topobench.dataloader.dataload_dataset import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.graph.loopy import Loopy
from topobench.nn.wrappers.graph.loopy_wrapper import LoopyWrapper
from topobench.transforms.data_manipulations.r_neighbourhood import (
    RNeighbourhood,
)

TRIANGLE_TAIL = ([[0, 1], [1, 2], [2, 0], [2, 3]], 4)
SQUARE_ISOLATED = ([[0, 1], [1, 2], [2, 3], [3, 0]], 5)  # node 4 isolated


def _graph(edges, num_nodes, feat, r):
    """Build a single transformed graph with a per-graph batch vector."""
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(
        x_0=torch.randn(num_nodes, feat),
        edge_index=edge_index,
        y=torch.zeros(num_nodes, dtype=torch.long),
        num_nodes=num_nodes,
    )
    data.batch_0 = torch.zeros(num_nodes, dtype=torch.long)
    return RNeighbourhood(r=r, transform_name="RNeighbourhood")(data)


def _batch(graphs, feat=8, r=2):
    """Collate several graphs the way the TopoBench dataloader does."""
    datas = [_graph(edges, n, feat, r) for edges, n in graphs]
    dataset = DataloadDataset(datas)
    return collate_fn([dataset.get(i) for i in range(len(datas))])


class TestAssemblePaths:
    """Test the graph-local to batch-global index reconstruction."""

    def test_shapes_and_transpose(self):
        batch = _batch([TRIANGLE_TAIL])
        loopy_n, loopy_a = LoopyWrapper._assemble_paths(batch)
        for order in range(3):
            if loopy_n[order].numel():
                assert loopy_n[order].shape[0] == order + 2
                assert loopy_n[order].shape == loopy_a[order].shape

    def test_single_graph_indices_unchanged(self):
        batch = _batch([TRIANGLE_TAIL])
        loopy_n, _ = LoopyWrapper._assemble_paths(batch)
        # With a single graph the offset is zero, so indices stay in range.
        assert loopy_n[1].max() < batch.x_0.shape[0]

    def test_two_graphs_offset(self):
        batch = _batch([TRIANGLE_TAIL, SQUARE_ISOLATED])
        loopy_n, _ = LoopyWrapper._assemble_paths(batch)
        # The square lives in the second graph -> its order-2 paths must
        # reference the second graph's node block (indices 4..8).
        counts = batch["loopyNcount2"]
        graph_of_path = torch.repeat_interleave(torch.arange(2), counts)
        square_paths = loopy_n[2].t()[graph_of_path == 1]
        assert square_paths.numel() > 0
        assert square_paths.min() >= 4

    def test_indices_within_own_graph(self):
        batch = _batch([TRIANGLE_TAIL, SQUARE_ISOLATED])
        loopy_n, _ = LoopyWrapper._assemble_paths(batch)
        node_counts = torch.bincount(batch.batch_0)
        node_ptr = torch.cat([node_counts.new_zeros(1), node_counts.cumsum(0)])
        for order in range(3):
            if not loopy_n[order].numel():
                continue
            counts = batch[f"loopyNcount{order}"]
            gid = torch.repeat_interleave(torch.arange(2), counts)
            glob = loopy_n[order].t()
            lo = node_ptr[gid].unsqueeze(1)
            hi = node_ptr[gid + 1].unsqueeze(1)
            assert torch.all((glob >= lo) & (glob < hi))

    def test_empty_order_handled(self):
        # A pure triangle has no order-2 (length-4) paths.
        batch = _batch([([[0, 1], [1, 2], [2, 0]], 3)])
        loopy_n, loopy_a = LoopyWrapper._assemble_paths(batch)
        assert loopy_n[2].shape[1] == 0
        assert loopy_a[2].shape[1] == 0


class TestLoopyWrapperForward:
    """Test the wrapper forward pass with a real backbone."""

    def _wrapper(self, hidden=8):
        backbone = Loopy(
            in_channels=hidden, hidden_channels=hidden, num_layers=2, r=2
        )
        return LoopyWrapper(
            backbone, out_channels=hidden, num_cell_dimensions=1
        )

    def test_output_keys(self):
        batch = _batch([TRIANGLE_TAIL, SQUARE_ISOLATED])
        out = self._wrapper()(batch)
        assert set(out.keys()) >= {"labels", "batch_0", "x_0"}

    def test_output_shape(self):
        batch = _batch([TRIANGLE_TAIL, SQUARE_ISOLATED])
        out = self._wrapper(hidden=8)(batch)
        assert out["x_0"].shape == (batch.x_0.shape[0], 8)
        assert torch.isfinite(out["x_0"]).all()

    def test_labels_and_batch_preserved(self):
        batch = _batch([TRIANGLE_TAIL])
        out = self._wrapper()(batch)
        assert torch.equal(out["batch_0"], batch.batch_0)
        assert torch.equal(out["labels"], batch.y)
