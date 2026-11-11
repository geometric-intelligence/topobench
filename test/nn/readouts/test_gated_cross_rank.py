"""Tests for the GatedCrossRankReadout layer."""

import pytest
import torch
import torch_geometric.data as tg_data

from topobench.nn.readouts.gated_cross_rank import GatedCrossRankReadout


class TestGatedCrossRankReadout:
    """Tests for the GatedCrossRankReadout layer."""

    @pytest.fixture
    def base_kwargs(self):
        """Fixture providing the required base parameters.

        Returns
        -------
        dict
            The base parameters for the GatedCrossRankReadout layer.
        """
        return {
            "hidden_dim": 8,
            "out_channels": 4,
            "task_level": "graph",
            "num_cell_dimensions": 3,
            "readout_name": "GatedCrossRankReadout",
        }

    @pytest.fixture
    def batch(self):
        """Fixture building a tiny 2-complex batch (triangle 0-1-2 plus edge 2-3).

        Returns
        -------
        torch_geometric.data.Data
            Batch with incidence_1 (4 nodes x 4 edges), incidence_2
            (4 edges x 1 triangle) and batch_0.
        """
        i1 = torch.tensor(
            [[0, 1, 0, 2, 1, 2, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]]
        )
        v1 = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        inc1 = torch.sparse_coo_tensor(i1, v1, (4, 4)).coalesce()
        i2 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        v2 = torch.tensor([1.0, -1.0, 1.0])
        inc2 = torch.sparse_coo_tensor(i2, v2, (4, 1)).coalesce()
        return tg_data.Data(
            incidence_1=inc1,
            incidence_2=inc2,
            batch_0=torch.zeros(4, dtype=torch.long),
        )

    @pytest.fixture
    def model_out(self):
        """Fixture providing per-rank features.

        Returns
        -------
        dict
            Model output dict with x_0, x_1, x_2.
        """
        torch.manual_seed(0)
        return {
            "x_0": torch.randn(4, 8),
            "x_1": torch.randn(4, 8),
            "x_2": torch.randn(1, 8),
        }

    def test_initialization(self, base_kwargs):
        """Test constructor: gates and per-path modules exist.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        layer = GatedCrossRankReadout(**base_kwargs)
        assert layer.gate_logits.shape == (3,)
        assert hasattr(layer, "conv_1_1")
        assert hasattr(layer, "conv_2_2") and hasattr(layer, "conv_2_1")
        assert "GatedCrossRankReadout" in repr(layer)

    def test_rank_weights_convex(self, base_kwargs):
        """Test the learned gate weights form a convex combination.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        layer = GatedCrossRankReadout(**base_kwargs)
        w = layer.rank_weights
        assert torch.allclose(w.sum(), torch.tensor(1.0))
        assert (w >= 0).all()

    def test_forward_shapes_and_logits(self, base_kwargs, batch, model_out):
        """Test forward fusion shape and graph-level logits via __call__.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        batch : torch_geometric.data.Data
            A fixture providing the batched complex.
        model_out : dict
            A fixture providing per-rank features.
        """
        layer = GatedCrossRankReadout(**base_kwargs)
        out = layer(model_out, batch)
        assert out["x_0"].shape == (4, 8)
        assert out["logits"].shape == (1, 4)

    def test_gradients_reach_gates(self, base_kwargs, batch, model_out):
        """Test gradients flow to the gate logits.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        batch : torch_geometric.data.Data
            A fixture providing the batched complex.
        model_out : dict
            A fixture providing per-rank features.
        """
        layer = GatedCrossRankReadout(**base_kwargs)
        out = layer(model_out, batch)
        out["logits"].sum().backward()
        assert layer.gate_logits.grad is not None
        assert torch.isfinite(layer.gate_logits.grad).all()

    def test_two_dimensions_only(self, base_kwargs, batch, model_out):
        """Test the readout with num_cell_dimensions=2 (no triangles used).

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        batch : torch_geometric.data.Data
            A fixture providing the batched complex.
        model_out : dict
            A fixture providing per-rank features.
        """
        kwargs = dict(base_kwargs, num_cell_dimensions=2)
        layer = GatedCrossRankReadout(**kwargs)
        assert layer.gate_logits.shape == (2,)
        out = layer(model_out, batch)
        assert out["x_0"].shape == (4, 8)

    def test_empty_top_rank(self, base_kwargs, model_out):
        """Test robustness when the complex has no triangles.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        model_out : dict
            A fixture providing per-rank features.
        """
        i1 = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        v1 = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        inc1 = torch.sparse_coo_tensor(i1, v1, (3, 2)).coalesce()
        inc2 = torch.sparse_coo_tensor(
            torch.empty(2, 0, dtype=torch.long), torch.empty(0), (2, 0)
        ).coalesce()
        batch = tg_data.Data(
            incidence_1=inc1,
            incidence_2=inc2,
            batch_0=torch.zeros(3, dtype=torch.long),
        )
        mo = {
            "x_0": torch.randn(3, 8),
            "x_1": torch.randn(2, 8),
            "x_2": torch.randn(0, 8),
        }
        layer = GatedCrossRankReadout(**base_kwargs)
        out = layer(mo, batch)
        assert out["x_0"].shape == (3, 8)
        assert torch.isfinite(out["x_0"]).all()
