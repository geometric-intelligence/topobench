"""Unit tests for the TopoU-Net backbone (arXiv:2605.10091)."""

import pytest
import torch

from topobench.nn.backbones.combinatorial.topounet import TopoUNet


def _toy_complex(d_feat=8):
    """Build a small combinatorial complex with ranks 0-3.

    Four nodes, four edges (01, 12, 02, 23), one triangle {0, 1, 2} and one
    global rank-3 cell containing the triangle.

    Parameters
    ----------
    d_feat : int
        Feature dimension of the rank-0 cochain.

    Returns
    -------
    tuple(dict, dict)
        Input cochains ``{0: x_0}`` and sparse consecutive incidence
        matrices ``{1: B_01, 2: B_12, 3: B_23}``.
    """
    x_0 = torch.randn(4, d_feat)
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0], [0.0]]).to_sparse()
    incidence_3 = torch.tensor([[1.0]]).to_sparse()
    return {0: x_0}, {1: incidence_1, 2: incidence_2, 3: incidence_3}


def test_invalid_rank_path():
    """Invalid encoder rank paths must be rejected at construction."""
    with pytest.raises(ValueError):
        TopoUNet(8, [0])
    with pytest.raises(ValueError):
        TopoUNet(8, [0, 1, 1])
    with pytest.raises(ValueError):
        TopoUNet(8, [0, 2, 1])


@pytest.mark.parametrize(
    "encoder_rank_path", [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
)
def test_forward_shapes(encoder_rank_path):
    """Decoder states must live in the cochain space of each path rank.

    Checks the structural compatibility of Proposition 3.4 of the paper.

    Parameters
    ----------
    encoder_rank_path : list[int]
        Encoder rank path to test.
    """
    d_feat = 8
    x_all, incidence_all = _toy_complex(d_feat)
    n_cells = {0: 4, 1: 4, 2: 1, 3: 1}
    model = TopoUNet(d_feat, encoder_rank_path)
    out = model(x_all, incidence_all)
    assert set(out.keys()) == set(encoder_rank_path)
    for rank in encoder_rank_path:
        assert out[rank].shape == (n_cells[rank], d_feat)


def test_skipped_rank_step():
    """A path step with a rank gap must use the direct incidence B_{0,2}.

    Section 3.5 of the paper: for s_{i+1} - s_i > 1 the transport uses the
    direct incidence matrix, realized as the binarized product of the
    consecutive incidence matrices.
    """
    d_feat = 8
    x_all, incidence_all = _toy_complex(d_feat)
    model = TopoUNet(d_feat, [0, 2])
    incidence_02 = model._step_incidence(incidence_all, 0, 2)
    dense = incidence_02.to_dense()
    # Nodes 0, 1, 2 belong to the triangle; values are binarized.
    assert dense.shape == (4, 1)
    assert torch.equal(dense, torch.tensor([[1.0], [1.0], [1.0], [0.0]]))
    out = model(x_all, incidence_all)
    assert out[0].shape == (4, d_feat)
    assert out[2].shape == (1, d_feat)


def test_no_skip_ablation():
    """Removing skip connections must change the decoder output.

    Reproduces the no-skip ablation of Section 4.6.3 of the paper, where the
    merge D_{s_i} = sigma((E_{s_i} + D~_{s_i}) W^m_i) is replaced by
    D_{s_i} = D~_{s_i}.
    """
    d_feat = 8
    x_all, incidence_all = _toy_complex(d_feat)
    model = TopoUNet(d_feat, [0, 1, 2]).eval()
    out_skip = model(x_all, incidence_all)
    model.use_skip = False
    out_no_skip = model(x_all, incidence_all)
    assert not torch.allclose(out_skip[0], out_no_skip[0])


def test_aggr_norm():
    """Degree normalization must average instead of sum over incident cells."""
    d_feat = 8
    x_all, incidence_all = _toy_complex(d_feat)
    torch.manual_seed(0)
    model_norm = TopoUNet(d_feat, [0, 1, 2], aggr_norm=True).eval()
    torch.manual_seed(0)
    model_raw = TopoUNet(d_feat, [0, 1, 2], aggr_norm=False).eval()
    out_norm = model_norm(x_all, incidence_all)
    out_raw = model_raw(x_all, incidence_all)
    assert not torch.allclose(out_norm[0], out_raw[0])


def test_permutation_equivariance():
    """The model must be equivariant to joint reindexing of cells.

    Checks Proposition 3.5 of the paper: permuting the cells of every rank
    (and reindexing the incidence matrices consistently) must permute the
    output cochains accordingly.
    """
    d_feat = 8
    x_all, incidence_all = _toy_complex(d_feat)
    model = TopoUNet(d_feat, [0, 1, 2]).eval()
    out = model(x_all, incidence_all)

    perm_0 = torch.randperm(4)
    perm_1 = torch.randperm(4)
    incidence_1 = incidence_all[1].to_dense()[perm_0][:, perm_1].to_sparse()
    incidence_2 = incidence_all[2].to_dense()[perm_1].to_sparse()
    out_perm = model(
        {0: x_all[0][perm_0]},
        {1: incidence_1, 2: incidence_2},
    )
    assert torch.allclose(out[0][perm_0], out_perm[0], atol=1e-5)
    assert torch.allclose(out[1][perm_1], out_perm[1], atol=1e-5)


def test_empty_rank():
    """Ranks without cells (e.g. triangle-free graphs) must be handled."""
    d_feat = 8
    x_0 = torch.randn(4, d_feat)
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    ).to_sparse()
    incidence_2 = torch.sparse_coo_tensor(size=(2, 0)).coalesce()
    model = TopoUNet(d_feat, [0, 1, 2]).eval()
    out = model({0: x_0}, {1: incidence_1, 2: incidence_2})
    assert out[2].shape == (0, d_feat)
    assert out[0].shape == (4, d_feat)
    assert torch.isfinite(out[0]).all()


def test_repr():
    """The repr must expose the configuration of the model."""
    model = TopoUNet(8, [0, 1, 2], use_skip=False)
    representation = repr(model)
    assert "TopoUNet" in representation
    assert "[0, 1, 2]" in representation
    assert "use_skip=False" in representation
