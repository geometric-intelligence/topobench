"""Unit tests for the GSN GIN+VN layer and model."""

import pytest
import torch
from torch_geometric.data import Batch, Data

# NOTE: import from the submodule, not the package. The backbones package
# auto-discovers modules and re-imports them under a synthetic name
# ("__init__.gsn"), which breaks PyG's MessagePassing signature inspector
# (it looks the class module up in sys.modules).
from topobench.nn.backbones.graph.gsn import (
    GSNGINVirtualNodeLayerV,
    GSNGINVirtualNodeModel,
    mlp_builder,
    mlp_dimension_builder,
)

# --- small shared inputs -------------------------------------------------

EDGE_INDEX = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
NUM_NODES = 3
IN = 4
GSN = 5
OUT = 6


def node_data():
    """A 3-node graph carrying node-level GSN encodings."""
    return Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        node_gsn_encodings=torch.randn(NUM_NODES, GSN),
    )


def call_node_model(model, data):
    """Invoke a node-mode GSN model the way the wrapper does.

    The GSN models take tensors (edge_index, x, gsn_embeddings, ...) directly,
    so this pulls those fields off a Data/Batch object.
    """
    return model(
        data.edge_index,
        data.x,
        data.node_gsn_encodings,
        batch=getattr(data, "batch", None),
    )


class TestMLPBuilders:
    """Tests for the MLP-construction helpers."""

    def test_dimension_builder(self):
        """A flat arch becomes consecutive (in, out) pairs."""
        assert mlp_dimension_builder([4, 8, 6]) == [(4, 8), (8, 6)]

    def test_builder_inserts_activation_between_layers(self):
        """A 2-layer MLP has 2 Linears + 1 ReLU, no trailing activation."""
        mlp = mlp_builder(mlp_dimension_builder([4, 8, 6]))
        n_linear = sum(1 for m in mlp if isinstance(m, torch.nn.Linear))
        n_relu = sum(1 for m in mlp if isinstance(m, torch.nn.ReLU))
        assert n_linear == 2
        assert n_relu == 1
        # last module is a Linear (unbounded output)
        assert isinstance(mlp[-1], torch.nn.Linear)


class TestVirtualNode:
    """Tests for the GIN+VN layer and module."""

    def test_layer_forward_with_and_without_edge_attr(self):
        """The VN layer runs with and without edge embeddings."""
        layer = GSNGINVirtualNodeLayerV(OUT, common_embedding_dim=IN)
        x = torch.randn(NUM_NODES, IN)
        assert layer(EDGE_INDEX, x).shape == (NUM_NODES, OUT)

        layer_e = GSNGINVirtualNodeLayerV(
            OUT, common_embedding_dim=IN, edge_dim=IN
        )
        edge_attr = torch.randn(EDGE_INDEX.size(1), IN)
        assert layer_e(EDGE_INDEX, x, edge_attr).shape == (NUM_NODES, OUT)

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_module_forward(self, num_layers):
        """Node-mode module maps to (N, out_channels) for any depth."""
        module = GSNGINVirtualNodeModel(
            "node", num_layers, IN, OUT, GSN, common_embedding_dim=8
        )
        assert call_node_model(module, node_data()).shape == (NUM_NODES, OUT)

    def test_module_edge_dim_path(self):
        """The edge-feature path projects and runs."""
        module = GSNGINVirtualNodeModel(
            "node", 2, IN, OUT, GSN, common_embedding_dim=8, edge_dim=2
        )
        data = node_data()
        edge_attr = torch.randn(EDGE_INDEX.size(1), 2)
        out = module(
            data.edge_index,
            data.x,
            data.node_gsn_encodings,
            edge_attr=edge_attr,
        )
        assert out.shape == (NUM_NODES, OUT)

    def test_module_no_batch_fallback(self):
        """A None batch vector is treated as a single graph."""
        module = GSNGINVirtualNodeModel(
            "node", 2, IN, OUT, GSN, common_embedding_dim=8
        )
        data = node_data()
        # batch defaults to None -> single-graph fallback
        out = module(data.edge_index, data.x, data.node_gsn_encodings)
        assert out.shape == (NUM_NODES, OUT)

    def test_edge_mode_not_implemented(self):
        """Edge mode is not implemented and raises."""
        with pytest.raises(NotImplementedError):
            GSNGINVirtualNodeModel("edge", 2, IN, OUT, GSN)

    def test_edge_attr_ignored_when_edge_dim_zero(self):
        """edge_attr present while edge_dim == 0 is tolerated (attr ignored)."""
        module = GSNGINVirtualNodeModel(
            "node", 2, IN, OUT, GSN, common_embedding_dim=8
        )
        data = node_data()
        edge_attr = torch.randn(EDGE_INDEX.size(1), 2)
        out = module(
            data.edge_index,
            data.x,
            data.node_gsn_encodings,
            edge_attr=edge_attr,
        )
        assert out.shape == (NUM_NODES, OUT)

    def test_per_graph_independence(self):
        """The virtual node stays per-graph: batching must not mix graphs."""
        torch.manual_seed(0)
        g1 = Data(
            x=torch.randn(3, IN),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            node_gsn_encodings=torch.randn(3, GSN),
        )
        g2 = Data(
            x=torch.randn(2, IN),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            node_gsn_encodings=torch.randn(2, GSN),
        )
        module = GSNGINVirtualNodeModel(
            "node", 3, IN, OUT, GSN, common_embedding_dim=8
        )
        module.eval()
        batched = call_node_model(module, Batch.from_data_list([g1, g2]))
        solo = call_node_model(module, Batch.from_data_list([g1]))
        assert torch.allclose(batched[:3], solo, atol=1e-5)


# --- helpers for the batch-norm / dropout tests --------------------------


def _count(module, cls):
    """Count direct children of ``module`` that are instances of ``cls``."""
    return sum(1 for m in module if isinstance(m, cls))


def two_graph_batch():
    """A batch of two 3-node graphs (>1 sample so BatchNorm works in train)."""
    g1 = Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        node_gsn_encodings=torch.randn(NUM_NODES, GSN),
    )
    g2 = Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        node_gsn_encodings=torch.randn(NUM_NODES, GSN),
    )
    return Batch.from_data_list([g1, g2])


class TestBatchNormAndDropout:
    """Tests for the optional BatchNorm / Dropout support."""

    def test_mlp_builder_inserts_bn_and_dropout_on_hidden_only(self):
        """BN + Dropout land on hidden layers; the output Linear stays raw."""
        # arch with two hidden layers -> three Linears
        mlp = mlp_builder(
            mlp_dimension_builder([4, 8, 8, 6]),
            batch_norm=True,
            dropout=0.3,
        )
        assert _count(mlp, torch.nn.Linear) == 3
        # one BN + one Dropout per non-final linear (i.e. per hidden layer)
        assert _count(mlp, torch.nn.BatchNorm1d) == 2
        assert _count(mlp, torch.nn.Dropout) == 2
        # nothing is appended after the final linear
        assert isinstance(mlp[-1], torch.nn.Linear)
        # BN precedes its activation; dropout follows it
        kinds = [type(m).__name__ for m in mlp]
        assert kinds[:4] == ["Linear", "BatchNorm1d", "ReLU", "Dropout"]

    def test_mlp_builder_defaults_add_nothing(self):
        """Without the flags the MLP is unchanged (no BN, no Dropout)."""
        mlp = mlp_builder(mlp_dimension_builder([4, 8, 6]))
        assert _count(mlp, torch.nn.BatchNorm1d) == 0
        assert _count(mlp, torch.nn.Dropout) == 0

    def test_zero_dropout_adds_no_dropout_layer(self):
        """dropout=0.0 must not insert Dropout modules (even with BN on)."""
        mlp = mlp_builder(
            mlp_dimension_builder([4, 8, 6]), batch_norm=True, dropout=0.0
        )
        assert _count(mlp, torch.nn.Dropout) == 0
        assert _count(mlp, torch.nn.BatchNorm1d) == 1

    def test_vn_layer_threads_flags_into_update_mlp(self):
        """The VN layer forwards BN/Dropout into its UP MLP."""
        layer = GSNGINVirtualNodeLayerV(
            OUT, common_embedding_dim=IN, batch_norm=True, dropout=0.2
        )
        assert _count(layer.UP, torch.nn.BatchNorm1d) == 1
        assert _count(layer.UP, torch.nn.Dropout) == 1

    def test_vn_module_between_layer_norms_and_g_updaters(self):
        """VN module: intermediate BN, Identity on the last, BN in G-updaters."""
        module = GSNGINVirtualNodeModel(
            "node",
            3,
            IN,
            OUT,
            GSN,
            common_embedding_dim=8,
            batch_norm=True,
            dropout=0.2,
        )
        norms = list(module._between_layer_norms)
        assert all(isinstance(n, torch.nn.BatchNorm1d) for n in norms[:-1])
        assert isinstance(norms[-1], torch.nn.Identity)
        # the virtual-node update MLPs also pick up the flags
        assert all(
            _count(mlp, torch.nn.BatchNorm1d) == 1
            for mlp in module.G_updater_MLPs
        )

    def test_vn_module_forward_train_mode_with_batch(self):
        """A BN/Dropout VN module runs in train mode on a real batch."""
        module = GSNGINVirtualNodeModel(
            "node",
            3,
            IN,
            OUT,
            GSN,
            common_embedding_dim=8,
            batch_norm=True,
            dropout=0.2,
        )
        module.train()
        out = call_node_model(module, two_graph_batch())
        assert out.shape == (2 * NUM_NODES, OUT)

    def test_dropout_active_in_train_inactive_in_eval(self):
        """Dropout perturbs outputs in train mode but is a no-op in eval."""
        torch.manual_seed(0)
        module = GSNGINVirtualNodeModel(
            "node", 3, IN, OUT, GSN, common_embedding_dim=8, dropout=0.5
        )
        batch = two_graph_batch()

        # eval: deterministic across repeated calls
        module.eval()
        assert torch.allclose(
            call_node_model(module, batch), call_node_model(module, batch)
        )

        # train: two forward passes differ because dropout masks differ
        module.train()
        assert not torch.allclose(
            call_node_model(module, batch), call_node_model(module, batch)
        )
