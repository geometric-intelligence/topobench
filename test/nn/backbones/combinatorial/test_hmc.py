"""Unit tests for the HMC backbone."""

import pytest
import torch

from topobench.nn.backbones.combinatorial.hmc import (
    HBNS,
    HBS,
    HMC,
    HMCLayer,
    sparse_row_norm,
)


def random_sparse(rows, cols, density=0.5, seed=0):
    """Create a random sparse binary matrix.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    density : float, optional
        Density of nonzero entries.
    seed : int, optional
        Random seed.

    Returns
    -------
    torch.sparse.Tensor
        Random sparse matrix in COO format.
    """
    generator = torch.Generator().manual_seed(seed)
    dense = (torch.rand(rows, cols, generator=generator) < density).float()
    return dense.to_sparse_coo()


class TestHMC:
    """Unit tests for the HMC backbone."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(0)
        self.n_0, self.n_1, self.n_2 = 6, 9, 4
        self.channels = 8
        self.x_0 = torch.randn(self.n_0, self.channels)
        self.x_1 = torch.randn(self.n_1, self.channels)
        self.x_2 = torch.randn(self.n_2, self.channels)
        self.adjacency_0 = random_sparse(self.n_0, self.n_0, seed=1)
        self.adjacency_1 = random_sparse(self.n_1, self.n_1, seed=2)
        self.coadjacency_2 = random_sparse(self.n_2, self.n_2, seed=3)
        self.incidence_1 = random_sparse(self.n_0, self.n_1, seed=4)
        self.incidence_2 = random_sparse(self.n_1, self.n_2, seed=5)

    def forward_model(self, model):
        """Run a model on the test fixtures.

        Parameters
        ----------
        model : torch.nn.Module
            HMC model to run.

        Returns
        -------
        tuple of torch.Tensor
            Model outputs.
        """
        return model(
            self.x_0,
            self.x_1,
            self.x_2,
            self.adjacency_0,
            self.adjacency_1,
            self.coadjacency_2,
            self.incidence_1,
            self.incidence_2,
        )

    def test_forward_shapes(self):
        """Test the output shapes of the HMC forward pass."""
        model = HMC(self.channels, 16, n_layers=2)
        x_0, x_1, x_2 = self.forward_model(model)
        assert x_0.shape == (self.n_0, 16)
        assert x_1.shape == (self.n_1, 16)
        assert x_2.shape == (self.n_2, 16)

    def test_single_layer(self):
        """Test HMC with a single layer."""
        model = HMC(self.channels, 16, n_layers=1)
        assert len(model.layers) == 1
        x_0, _, _ = self.forward_model(model)
        assert x_0.shape == (self.n_0, 16)

    def test_gradient_flow(self):
        """Test that gradients flow through all HMC parameters."""
        model = HMC(self.channels, 16, n_layers=1)
        x_0, x_1, x_2 = self.forward_model(model)
        (x_0.sum() + x_1.sum() + x_2.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name

    def test_equivalence_with_topomodelx(self):
        """Test numerical equivalence with the TopoModelX reference."""
        pytest.importorskip("topomodelx")
        from topomodelx.nn.combinatorial.hmc import HMC as TMXHMC

        c = self.channels
        reference = TMXHMC(
            [[[c] * 3, [c] * 3, [c] * 3], [[c] * 3, [c] * 3, [c] * 3]]
        )
        model = HMC(c, c, n_layers=2, softmax_attention=True)
        model.load_state_dict(reference.state_dict())
        reference.eval()
        model.eval()
        with torch.no_grad():
            out_ref = reference(
                self.x_0,
                self.x_1,
                self.x_2,
                self.adjacency_0,
                self.adjacency_1,
                self.coadjacency_2,
                self.incidence_1,
                self.incidence_2,
            )
            out = self.forward_model(model)
        for a, b in zip(out_ref, out, strict=True):
            assert torch.allclose(a, b, atol=1e-5)

    def test_empty_rank_two(self):
        """Test the forward pass with no 2-cells."""
        model = HMC(self.channels, 16, n_layers=2)
        x_2 = torch.zeros(0, self.channels)
        coadjacency_2 = torch.zeros(0, 0).to_sparse_coo()
        incidence_2 = torch.zeros(self.n_1, 0).to_sparse_coo()
        x_0, x_1, x_2 = model(
            self.x_0,
            self.x_1,
            x_2,
            self.adjacency_0,
            self.adjacency_1,
            coadjacency_2,
            self.incidence_1,
            incidence_2,
        )
        assert x_0.shape == (self.n_0, 16)
        assert x_1.shape == (self.n_1, 16)
        assert x_2.shape == (0, 16)

    def test_row_norm_attention(self):
        """Test the forward pass with row-normalized attention."""
        model = HMC(self.channels, 16, n_layers=1, softmax_attention=False)
        x_0, _, _ = self.forward_model(model)
        assert torch.isfinite(x_0).all()


class TestHBS:
    """Unit tests for the HBS block."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(0)
        self.n_cells = 7
        self.x = torch.randn(self.n_cells, 8)
        self.neighborhood = random_sparse(self.n_cells, self.n_cells, seed=6)

    def test_forward(self):
        """Test the forward pass of HBS."""
        block = HBS(8, 16, update_func="relu")
        out = block(self.x, self.neighborhood)
        assert out.shape == (self.n_cells, 16)

    def test_forward_no_update(self):
        """Test the forward pass without an activation."""
        block = HBS(8, 16, update_func=None)
        out = block(self.x, self.neighborhood)
        assert out.shape == (self.n_cells, 16)

    def test_m_hop(self):
        """Test HBS with two-hop attention."""
        block = HBS(8, 16, m_hop=2, update_func="sigmoid")
        assert len(block.weight) == 2
        out = block(self.x, self.neighborhood)
        assert out.shape == (self.n_cells, 16)

    def test_m_hop_matches_manual_two_hop(self):
        """Test two-hop HBS against a manual dense computation.

        The block accumulates one contribution per hop, the ``p``-th using
        the neighborhood power :math:`N^p` (whose entries count paths of
        length ``p``), its own weight matrix :math:`W_p` and its own
        attention vector :math:`a_p`.
        """
        torch.manual_seed(0)
        n_cells, in_channels, out_channels = 4, 3, 2
        block = HBS(in_channels, out_channels, m_hop=2, update_func=None)
        x = torch.randn(n_cells, in_channels)
        dense = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        out = block(x, dense.to_sparse_coo())

        expected = torch.zeros(n_cells, out_channels)
        power = torch.eye(n_cells)
        for weight, att_weight in zip(
            block.weight, block.att_weight, strict=True
        ):
            power = power @ dense
            message = x @ weight
            scores = torch.zeros(n_cells, n_cells)
            for i in range(n_cells):
                for j in range(n_cells):
                    if power[i, j] != 0:
                        pair = torch.cat([message[i], message[j]])
                        scores[i, j] = torch.nn.functional.leaky_relu(
                            pair @ att_weight.squeeze(1), negative_slope=0.2
                        )
            row_sum = scores.sum(dim=1, keepdim=True)
            row_sum = torch.where(
                row_sum == 0, torch.ones_like(row_sum), row_sum
            )
            expected = expected + ((scores / row_sum) * power) @ message

        # The two-hop power must count paths, so it is not merely binary.
        assert (power > 1).any()
        # atol=1e-5: the block computes via torch.sparse CSR kernels while
        # the reference above is dense matmul; their float32 rounding differs
        # by a few ulps and the gap varies with the CPU microarchitecture of
        # the runner. A genuine defect (wrong hop matrix, wrong weight, wrong
        # normalization) produces errors several orders of magnitude larger.
        assert torch.allclose(out, expected, atol=1e-5)

    def test_softmax(self):
        """Test HBS with softmax attention."""
        block = HBS(8, 16, softmax=True, update_func="tanh")
        out = block(self.x, self.neighborhood)
        assert out.shape == (self.n_cells, 16)

    def test_invalid_initialization(self):
        """Test that an invalid initialization raises a ValueError."""
        with pytest.raises(ValueError):
            HBS(8, 16, initialization="foo")

    def test_invalid_update_func(self):
        """Test that an invalid update function raises a ValueError."""
        block = HBS(8, 16, update_func="foo")
        with pytest.raises(ValueError):
            block(self.x, self.neighborhood)


class TestHBNS:
    """Unit tests for the HBNS block."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(0)
        self.n_source, self.n_target = 9, 6
        self.x_source = torch.randn(self.n_source, 8)
        self.x_target = torch.randn(self.n_target, 10)
        self.neighborhood = random_sparse(self.n_target, self.n_source, seed=7)

    def test_forward(self):
        """Test the forward pass of HBNS."""
        block = HBNS(8, 16, 10, 12, update_func="relu")
        out_source, out_target = block(
            self.x_source, self.x_target, self.neighborhood
        )
        assert out_source.shape == (self.n_source, 16)
        assert out_target.shape == (self.n_target, 12)

    def test_forward_no_update(self):
        """Test the forward pass without an activation."""
        block = HBNS(8, 16, 10, 12, update_func=None)
        out_source, out_target = block(
            self.x_source, self.x_target, self.neighborhood
        )
        assert out_source.shape == (self.n_source, 16)
        assert out_target.shape == (self.n_target, 12)

    @pytest.mark.parametrize("update_func", ["sigmoid", "tanh"])
    def test_update_funcs(self, update_func):
        """Test HBNS with different update functions.

        Parameters
        ----------
        update_func : str
            Update function to test.
        """
        block = HBNS(8, 16, 10, 12, update_func=update_func)
        out_source, out_target = block(
            self.x_source, self.x_target, self.neighborhood
        )
        assert out_source.shape == (self.n_source, 16)
        assert out_target.shape == (self.n_target, 12)

    def test_softmax(self):
        """Test HBNS with softmax attention."""
        block = HBNS(8, 16, 10, 12, softmax=True, update_func="relu")
        out_source, out_target = block(
            self.x_source, self.x_target, self.neighborhood
        )
        assert out_source.shape == (self.n_source, 16)
        assert out_target.shape == (self.n_target, 12)

    def test_invalid_initialization(self):
        """Test that an invalid initialization raises a ValueError."""
        with pytest.raises(ValueError):
            HBNS(8, 16, 10, 12, initialization="foo")

    def test_invalid_update_func(self):
        """Test that an invalid update function raises a ValueError."""
        block = HBNS(8, 16, 10, 12, update_func="foo")
        with pytest.raises(ValueError):
            block(self.x_source, self.x_target, self.neighborhood)

    def test_unequal_channels_numerical(self):
        """Test HBNS numerically with unequal output channel dimensions.

        Regression test for the reverse-direction attention reordering:
        the forward message concatenates the target-output portion
        (``target_out_channels`` columns from ``X_s W_s``) before the
        source-output portion (``source_out_channels`` columns from
        ``X_t W_t``), so the reverse attention vector must be split at
        ``target_out_channels``. The expected logits and row-normalized
        attentions are computed manually with dense operations and
        compared to the HBNS output.
        """
        torch.manual_seed(42)
        source_out, target_out = 3, 5
        block = HBNS(
            8,
            source_out,
            10,
            target_out,
            update_func=None,
            softmax=False,
        )
        block.eval()

        with torch.no_grad():
            s_message = self.x_source @ block.w_s  # [n_s, target_out]
            t_message = self.x_target @ block.w_t  # [n_t, source_out]
            a_s = block.att_weight[:target_out]  # pairs with s_message
            a_t = block.att_weight[target_out:]  # pairs with t_message

            n_dense = self.neighborhood.to_dense()  # [n_t, n_s]
            e_logits = torch.full_like(n_dense, 0.0)
            f_logits = torch.zeros(self.n_source, self.n_target)
            for t in range(self.n_target):
                for s in range(self.n_source):
                    if n_dense[t, s] == 0:
                        continue
                    forward = torch.cat([s_message[s], t_message[t]])
                    reverse = torch.cat([t_message[t], s_message[s]])
                    e_logits[t, s] = torch.nn.functional.leaky_relu(
                        forward @ torch.cat([a_s, a_t]).squeeze(1),
                        negative_slope=block.negative_slope,
                    )
                    f_logits[s, t] = torch.nn.functional.leaky_relu(
                        reverse @ torch.cat([a_t, a_s]).squeeze(1),
                        negative_slope=block.negative_slope,
                    )

            mask_e = n_dense != 0
            mask_f = mask_e.T
            e_att = torch.where(mask_e, e_logits, torch.zeros(()))
            f_att = torch.where(mask_f, f_logits, torch.zeros(()))
            e_row_sum = e_att.sum(dim=1, keepdim=True)
            f_row_sum = f_att.sum(dim=1, keepdim=True)
            e_row_sum[~mask_e.any(dim=1)] = 1.0
            f_row_sum[~mask_f.any(dim=1)] = 1.0
            e_att = e_att / e_row_sum
            f_att = f_att / f_row_sum

            expected_target = (e_att * n_dense) @ s_message
            expected_source = (f_att * n_dense.T) @ t_message

            out_source, out_target = block(
                self.x_source, self.x_target, self.neighborhood
            )

        # Sanity: normalization must not be trivial (rows with several
        # neighbors exist, so attention values differ from one).
        assert (mask_e.sum(dim=1) > 1).any()
        assert not torch.allclose(
            e_att[mask_e], torch.ones_like(e_att[mask_e])
        )

        assert torch.allclose(out_target, expected_target, atol=1e-5)
        assert torch.allclose(out_source, expected_source, atol=1e-5)


class TestHMCLayer:
    """Unit tests for the HMCLayer."""

    def test_forward(self):
        """Test the forward pass of a single HMCLayer."""
        torch.manual_seed(0)
        layer = HMCLayer([8] * 3, [16] * 3, [16] * 3)
        x_0 = torch.randn(6, 8)
        x_1 = torch.randn(9, 8)
        x_2 = torch.randn(4, 8)
        out_0, out_1, out_2 = layer(
            x_0,
            x_1,
            x_2,
            random_sparse(6, 6, seed=1),
            random_sparse(9, 9, seed=2),
            random_sparse(4, 4, seed=3),
            random_sparse(6, 9, seed=4),
            random_sparse(9, 4, seed=5),
        )
        assert out_0.shape == (6, 16)
        assert out_1.shape == (9, 16)
        assert out_2.shape == (4, 16)


def test_xavier_normal_initialization():
    """Test the xavier_normal initialization of HBS and HBNS."""
    HBS(8, 16, initialization="xavier_normal")
    HBNS(8, 16, 10, 12, initialization="xavier_normal")


@pytest.mark.parametrize("update_func", ["sigmoid", "tanh", None])
def test_aggregation_update_funcs(update_func):
    """Test HMCLayer aggregation with different update functions.

    Parameters
    ----------
    update_func : str or None
        Aggregation update function to test.
    """
    layer = HMCLayer(
        [8] * 3, [16] * 3, [16] * 3, update_func_aggregation=update_func
    )
    out = layer.aggregate([torch.randn(5, 16), torch.randn(5, 16)])
    assert out.shape == (5, 16)


def test_sparse_row_norm():
    """Test the sparse row normalization helper."""
    dense = torch.tensor([[1.0, 3.0], [0.0, 2.0]])
    normalized = sparse_row_norm(dense.to_sparse_coo()).to_dense()
    expected = torch.tensor([[0.25, 0.75], [0.0, 1.0]])
    assert torch.allclose(normalized, expected)


def test_sparse_row_norm_zero_row_sum():
    """Test that rows whose sum is zero stay finite (left unnormalized)."""
    dense = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    normalized = sparse_row_norm(dense.to_sparse_coo()).to_dense()
    expected = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized, expected)


def test_hbns_zero_messages_row_norm_finite():
    """Test HBNS with zero inputs and row-norm attention stays finite.

    Zero source and target features produce all-zero attention logits,
    whose row sums are zero; the normalization guard must prevent NaNs.
    """
    torch.manual_seed(0)
    block = HBNS(8, 4, 10, 6, update_func=None, softmax=False)
    x_source = torch.zeros(9, 8)
    x_target = torch.zeros(6, 10)
    neighborhood = random_sparse(6, 9, seed=7)
    out_source, out_target = block(x_source, x_target, neighborhood)
    assert torch.isfinite(out_source).all()
    assert torch.isfinite(out_target).all()
