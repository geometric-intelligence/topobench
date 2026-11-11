"""Unit tests for the SheafTSP wrapper."""

import torch

from topobench.nn.backbones.cell.sheaf_tsp import SheafTSP
from topobench.nn.wrappers.cell.sheaf_tsp_wrapper import SheafTSPWrapper


def _make_wrapper(data, **kwargs):
    """Build a SheafTSPWrapper around a small backbone.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Lifted complex providing ``x_1``.
    **kwargs : dict
        Extra wrapper options (``count_source``, ``petals``, ...).

    Returns
    -------
    SheafTSPWrapper
        Wrapper ready for a forward pass.
    """
    channels = data.x_1.shape[1]
    backbone = SheafTSP(
        in_channels=channels, n_layers=1, stalk_dim=2, filter_order=2
    )
    return SheafTSPWrapper(
        backbone,
        out_channels=channels,
        num_cell_dimensions=2,
        residual_connections=False,
        **kwargs,
    )


class TestSheafTSPWrapper:
    """Test the SheafTSP wrapper."""

    def test_forward_output_keys(self, sg1_clique_lifted):
        """Forward pass produces all expected keys and shapes.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CliqueLifting.
        """
        data = sg1_clique_lifted
        wrapper = _make_wrapper(data)
        out = wrapper(data)

        for key in ["labels", "batch_0", "x_0", "x_1"]:
            assert key in out
        assert out["x_1"].shape == data.x_1.shape
        assert out["x_0"].shape[0] == data.x_0.shape[0]
        assert torch.isfinite(out["x_0"]).all()

    def test_incidence_count_signal(self, sg1_clique_lifted):
        """The endogenous count t_v = |B1||B2|1 sums to 6x #2-cells.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CliqueLifting.
        """
        data = sg1_clique_lifted
        inc1 = torch.abs(data.incidence_1.coalesce())
        inc2 = torch.abs(data.incidence_2.coalesce())
        n_two_cells = inc2.shape[1]
        assert n_two_cells > 0, "Fixture must contain 2-cells"

        ones = torch.ones(n_two_cells, 1)
        t_v = torch.sparse.mm(inc1, torch.sparse.mm(inc2, ones))
        assert t_v.sum().item() == 6 * n_two_cells

        # The wrapper injects t_v through the warm channel: scaling
        # tri_warm must move column 0 of x_0 by (warm2 - warm1) * t_v.
        w1 = _make_wrapper(data, count_source="incidence", tri_warm=0.1)
        w2 = _make_wrapper(data, count_source="incidence", tri_warm=0.6)
        w2.load_state_dict(w1.state_dict())
        with torch.no_grad():
            w2.tri_embed.weight[0, 0] = 0.6
        w1.eval()
        w2.eval()
        with torch.no_grad():
            x0_a = w1(data)["x_0"]
            x0_b = w2(data)["x_0"]
        delta = x0_b[:, 0] - x0_a[:, 0]
        assert torch.allclose(delta, 0.5 * t_v.squeeze(1), atol=1e-5)
        # Other channels are untouched by the count signal
        assert torch.allclose(x0_a[:, 1:], x0_b[:, 1:], atol=1e-6)

    def test_tri_degree_transform_source(self, sg1_clique_lifted):
        """count_source='auto' prefers the TriangleDegree signal.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CliqueLifting.
        """
        data = sg1_clique_lifted.clone()
        n = data.x_0.shape[0]
        wrapper = _make_wrapper(data, count_source="auto", tri_warm=1.0)
        wrapper.eval()

        data.tri_degree = torch.zeros(n, 1)
        with torch.no_grad():
            x0_zero = wrapper(data)["x_0"]
        data.tri_degree = torch.ones(n, 1)
        with torch.no_grad():
            x0_one = wrapper(data)["x_0"]
        delta = x0_one[:, 0] - x0_zero[:, 0]
        assert torch.allclose(delta, torch.ones(n), atol=1e-5)

    def test_dirichlet_exposed_in_training_only(self, sg1_clique_lifted):
        """sheaf_dirichlet appears in training and never in eval.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CliqueLifting.
        """
        data = sg1_clique_lifted
        wrapper = _make_wrapper(data)

        wrapper.train()
        out = wrapper(data)
        assert "sheaf_dirichlet" in out
        assert torch.isfinite(out["sheaf_dirichlet"])

        wrapper.eval()
        out = wrapper(data)
        assert "sheaf_dirichlet" not in out

    def test_stream_gate_and_petals(self, sg1_clique_lifted):
        """Optional stream gate and petals branch run end to end.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with CliqueLifting.
        """
        data = sg1_clique_lifted
        wrapper = _make_wrapper(data, stream_gate=True, petals=True)
        out = wrapper(data)
        assert torch.isfinite(out["x_0"]).all()

        # Zero-init petals fusion: epoch-0 output matches the plain
        # wrapper up to the (identity-initialized) stream gate.
        plain = _make_wrapper(data)
        plain.load_state_dict(
            {
                k: v
                for k, v in wrapper.state_dict().items()
                if k in plain.state_dict()
            }
        )
        wrapper.eval()
        plain.eval()
        with torch.no_grad():
            diff = (
                (wrapper(data)["x_0"] - plain(data)["x_0"]).abs().max().item()
            )
        assert diff < 1e-5, f"Zero-init branches changed epoch 0: {diff}"

    def test_petal_edges_triangle(self):
        """Petal edges recover exactly the pairs sharing a triangle."""

        class _Batch:
            pass

        # Path graph 0-1-2-3 plus edge 0-2: one triangle (0, 1, 2).
        edges = [(0, 1), (1, 2), (2, 3), (0, 2)]
        rows, cols, vals = [], [], []
        for e, (u, v) in enumerate(edges):
            rows += [u, v]
            cols += [e, e]
            vals += [1.0, 1.0]
        batch = _Batch()
        batch.incidence_1 = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), torch.tensor(vals), (4, len(edges))
        )
        ei = SheafTSPWrapper._petal_edges(batch, 4)
        got = {tuple(p) for p in ei.t().tolist()}
        assert got == {(0, 1), (0, 2), (1, 2)}

        # Without incidence_1 the branch degrades to None.
        assert SheafTSPWrapper._petal_edges(_Batch(), 4) is None
