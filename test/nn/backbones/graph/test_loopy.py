"""Unit tests for the Loopy backbone."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.dataloader.dataload_dataset import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.graph.loopy import (
    ACTIVATIONS,
    CustomGINConv,
    Loopy,
    LoopyLayer,
    MLP,
    _path_propagate,
    get_activation,
)
from topobench.nn.wrappers.graph.loopy_wrapper import LoopyWrapper
from topobench.transforms.data_manipulations.r_neighbourhood import (
    RNeighbourhood,
)

TRIANGLE_TAIL = ([[0, 1], [1, 2], [2, 0], [2, 3]], 4)
SQUARE = ([[0, 1], [1, 2], [2, 3], [3, 0]], 4)


def _assembled(graphs, hidden, r=2):
    """Return ``(x, loopy_n, loopy_a, num_nodes)`` for direct layer tests."""
    datas = []
    for edges, n in graphs:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data = Data(
            x_0=torch.randn(n, hidden),
            edge_index=edge_index,
            y=torch.zeros(n, dtype=torch.long),
            num_nodes=n,
        )
        data.batch_0 = torch.zeros(n, dtype=torch.long)
        datas.append(RNeighbourhood(r=r, transform_name="RN")(data))
    dataset = DataloadDataset(datas)
    batch = collate_fn([dataset.get(i) for i in range(len(datas))])
    loopy_n, loopy_a = LoopyWrapper._assemble_paths(batch)
    return batch.x_0, loopy_n, loopy_a, batch.x_0.shape[0]


class TestGetActivation:
    """Test the activation resolver."""

    @pytest.mark.parametrize("name", sorted(ACTIVATIONS))
    def test_known(self, name):
        act = get_activation(name)
        assert isinstance(act, torch.nn.Module)
        assert act(torch.randn(3, 2)).shape == (3, 2)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            get_activation("nope")


class TestPathPropagate:
    """Test the path-neighbour convolution."""

    def test_documented_example(self):
        x = torch.tensor([[1.0], [5.0]]).unsqueeze(1)  # (2, 1, 1)
        out = _path_propagate(x)
        assert torch.equal(out.squeeze(), torch.tensor([5.0, 1.0]))

    def test_shape_preserved(self):
        x = torch.randn(4, 6, 8)
        assert _path_propagate(x).shape == x.shape

    def test_middle_node_sums_both_neighbours(self):
        x = torch.tensor([[1.0], [2.0], [4.0]]).unsqueeze(1)  # (3, 1, 1)
        out = _path_propagate(x).squeeze()
        assert out[1] == 5.0  # 1 + 4


class TestMLP:
    """Test the internal MLP."""

    def test_forward_shape(self):
        assert MLP(8, 5)(torch.randn(6, 8)).shape == (6, 5)

    @pytest.mark.parametrize("num_layers", [2, 3])
    def test_num_layers(self, num_layers):
        mlp = MLP(8, 8, num_layers=num_layers)
        assert len(mlp.lins) == num_layers

    def test_batchnorm(self):
        mlp = MLP(8, 8, norm="BatchNorm1d")
        assert isinstance(mlp.norm, torch.nn.BatchNorm1d)
        assert mlp(torch.randn(4, 8)).shape == (4, 8)

    def test_reset_parameters(self):
        MLP(8, 8, norm="BatchNorm1d").reset_parameters()


class TestCustomGINConv:
    """Test the path GIN convolution."""

    def test_forward_shape(self):
        conv = CustomGINConv(MLP(8, 8), in_channels=8, num_embeddings=4)
        x = torch.randn(3, 5, 8)  # (path_length, num_paths, channels)
        atomic = torch.randint(0, 4, (3, 5))
        assert conv(x, atomic).shape == (5, 8)

    def test_reset_parameters(self):
        CustomGINConv(MLP(8, 8), 8, 4).reset_parameters()


class TestLoopyLayer:
    """Test a single loopy layer."""

    def test_forward_shape(self):
        x, ln, la, n = _assembled([TRIANGLE_TAIL], hidden=8)
        layer = LoopyLayer(8, 8, r=2)
        assert layer(x, ln, la, n).shape == (n, 8)

    def test_shared_uses_single_conv(self):
        assert len(LoopyLayer(8, 8, r=3, shared=True).convs) == 1
        assert len(LoopyLayer(8, 8, r=3, shared=False).convs) == 3

    def test_chunk_size_invariant(self):
        x, ln, la, n = _assembled([SQUARE, TRIANGLE_TAIL], hidden=8)
        torch.manual_seed(0)
        big = LoopyLayer(8, 8, r=2, path_chunk_size=10**9).eval()
        torch.manual_seed(0)
        small = LoopyLayer(8, 8, r=2, path_chunk_size=1).eval()
        out_big = big(x, ln, la, n)
        out_small = small(x, ln, la, n)
        assert torch.allclose(out_big, out_small, atol=1e-5)

    def test_checkpoint_backward(self):
        x, ln, la, n = _assembled([SQUARE], hidden=8)
        x = x.clone().requires_grad_(True)
        layer = LoopyLayer(8, 8, r=2, path_chunk_size=1).train()
        layer(x, ln, la, n).sum().backward()
        assert x.grad is not None


class TestLoopy:
    """Test the full backbone."""

    def _run(self, graphs, hidden=8, **kw):
        x, ln, la, n = _assembled(graphs, hidden=hidden, r=kw.get("r", 2))
        model = Loopy(hidden, hidden, **kw)
        return model, model(x, None, loopy_n=ln, loopy_a=la), n

    def test_init_attributes(self):
        model = Loopy(8, 8, num_layers=3, r=2)
        assert model.out_channels == 8
        assert model.r == 2
        assert len(model.layers) == 3

    def test_forward_shape(self):
        _, out, n = self._run([TRIANGLE_TAIL])
        assert out.shape == (n, 8)
        assert torch.isfinite(out).all()

    def test_backward_all_params(self):
        x, ln, la, n = _assembled([SQUARE, TRIANGLE_TAIL], hidden=8)
        model = Loopy(8, 8, num_layers=2, r=2).train()
        model(x, None, loopy_n=ln, loopy_a=la).sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"

    @pytest.mark.parametrize("r", [1, 2, 3])
    def test_forward_different_r(self, r):
        _, out, n = self._run([SQUARE], r=r)
        assert out.shape == (n, 8)

    def test_dropout_and_kwargs_ignored(self):
        model, out, n = self._run(
            [TRIANGLE_TAIL], dropout=0.5, unused="x"
        )
        assert out.shape == (n, 8)

    def test_eval_is_deterministic(self):
        x, ln, la, n = _assembled([SQUARE], hidden=8)
        model = Loopy(8, 8, num_layers=2, r=2, dropout=0.5).eval()
        a = model(x, None, loopy_n=ln, loopy_a=la)
        b = model(x, None, loopy_n=ln, loopy_a=la)
        assert torch.allclose(a, b)

    def test_isolated_nodes(self):
        # Graph with an isolated node (no paths touch it).
        x, ln, la, n = _assembled([([[0, 1], [1, 2], [2, 0]], 5)], hidden=8)
        out = Loopy(8, 8, num_layers=1, r=2)(x, None, loopy_n=ln, loopy_a=la)
        assert out.shape == (n, 8)
        assert torch.isfinite(out).all()

    def test_end_to_end_via_wrapper(self):
        x, ln, la, n = _assembled([TRIANGLE_TAIL, SQUARE], hidden=8)
        model = Loopy(8, 8, num_layers=2, r=2)
        out = model(x, None, loopy_n=ln, loopy_a=la)
        assert out.shape == (n, 8)
