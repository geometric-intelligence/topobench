"""Unit tests for the Bi-SCNN simplicial backbone."""

import pytest
import torch

from topobench.nn.backbones.simplicial.biscnn import (
    BiSCNN,
    BiSCNNLayer,
    _hard_sign,
    _row_l1_mean,
)


def _sparse_identity(size: int) -> torch.Tensor:
    """Create a sparse identity matrix."""
    indices = torch.arange(size)
    return torch.sparse_coo_tensor(
        torch.stack((indices, indices)),
        torch.ones(size),
        (size, size),
    ).coalesce()


@pytest.fixture
def simplicial_inputs():
    """Create node, edge, face features and compatible Laplacians."""
    x_all = (
        torch.randn(5, 3),
        torch.randn(7, 4),
        torch.randn(2, 5),
    )
    laplacian_all = (
        _sparse_identity(5),
        _sparse_identity(7),
        _sparse_identity(7),
        _sparse_identity(2),
        _sparse_identity(2),
    )
    incidence_all = (
        torch.sparse_coo_tensor(size=(5, 7)),
        torch.sparse_coo_tensor(size=(7, 2)),
    )
    return x_all, laplacian_all, incidence_all


def test_hard_sign_matches_paper_approximation():
    """Hard tanh must clip values while preserving the linear central region."""
    values = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    torch.testing.assert_close(_hard_sign(values), expected)


def test_row_l1_mean():
    """Equation (23) is the row-wise L1 norm divided by feature count."""
    features = torch.tensor([[1.0, -3.0], [2.0, 4.0]])
    expected = torch.tensor([[2.0], [3.0]])
    torch.testing.assert_close(_row_l1_mean(features), expected)


def test_layer_keeps_weights_full_precision():
    """Bi-SCNN binarizes features, not trainable matrices."""
    layer = BiSCNNLayer(3, 4, use_lower=True, use_upper=True)
    assert layer.lower_weight.dtype.is_floating_point
    assert layer.upper_weight.dtype.is_floating_point
    assert layer.harmonic_weight.dtype.is_floating_point


def test_layer_outputs_magnitude_and_binary():
    """Each paper layer must return its two propagation paths."""
    layer = BiSCNNLayer(3, 4, use_lower=True, use_upper=True)
    features = torch.randn(6, 3)
    operator = _sparse_identity(6)
    magnitude, binary = layer(features, operator, operator)
    assert magnitude.shape == (6, 1)
    assert binary.shape == (6, 4)
    assert torch.all(magnitude >= 0)
    assert torch.all(binary <= 1)
    assert torch.all(binary >= -1)


def test_forward_shapes_and_finiteness(simplicial_inputs):
    """The backbone returns rank-specific TopoBench-compatible embeddings."""
    x_all, laplacian_all, incidence_all = simplicial_inputs
    model = BiSCNN(
        in_channels_all=(3, 4, 5),
        hidden_channels_all=(8, 8, 8),
        n_layers=2,
        sc_order=3,
    )
    outputs = model(x_all, laplacian_all, incidence_all)
    assert [output.shape for output in outputs] == [(5, 8), (7, 8), (2, 8)]
    assert all(torch.isfinite(output).all() for output in outputs)


def test_dense_and_sparse_operators_agree():
    """Dense and sparse Hodge multiplication must be numerically identical."""
    torch.manual_seed(7)
    layer = BiSCNNLayer(3, 2, use_lower=True, use_upper=True)
    features = torch.randn(4, 3)
    dense = torch.eye(4)
    sparse = dense.to_sparse()
    dense_outputs = layer(features, dense, dense)
    sparse_outputs = layer(features, sparse, sparse)
    for dense_output, sparse_output in zip(
        dense_outputs, sparse_outputs, strict=True
    ):
        torch.testing.assert_close(dense_output, sparse_output)


def test_sc_order_two_uses_lower_face_laplacian():
    """A two-dimensional complex has no rank-two upper Hodge component."""
    model = BiSCNN(
        in_channels_all=(2, 2, 2),
        hidden_channels_all=(3, 3, 3),
        n_layers=1,
        sc_order=2,
    )
    x_all = (torch.randn(4, 2), torch.randn(5, 2), torch.randn(1, 2))
    laplacian_all = (
        _sparse_identity(4),
        _sparse_identity(5),
        _sparse_identity(5),
        _sparse_identity(1),
    )
    outputs = model(x_all, laplacian_all)
    assert [output.shape for output in outputs] == [(4, 3), (5, 3), (1, 3)]


def test_gradients_flow_through_hard_tanh(simplicial_inputs):
    """The hard-tanh proxy permits ordinary gradient-based optimization."""
    x_all, laplacian_all, incidence_all = simplicial_inputs
    x_all = tuple(features.requires_grad_() for features in x_all)
    model = BiSCNN(
        in_channels_all=(3, 4, 5),
        hidden_channels_all=(6, 6, 6),
        n_layers=2,
        sc_order=3,
    )
    loss = sum(
        output.square().mean()
        for output in model(x_all, laplacian_all, incidence_all)
    )
    loss.backward()
    assert all(features.grad is not None for features in x_all)
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "in_channels_all": (3, 4),
                "hidden_channels_all": (5, 5, 5),
            },
            "node, edge, and face",
        ),
        (
            {
                "in_channels_all": (3, 4, 5),
                "hidden_channels_all": (5, 5, 5),
                "n_layers": 0,
            },
            "at least one",
        ),
        (
            {
                "in_channels_all": (3, 4, 5),
                "hidden_channels_all": (5, 5, 5),
                "sc_order": 1,
            },
            "at least two",
        ),
    ],
)
def test_invalid_model_arguments(kwargs, message):
    """Invalid architecture arguments must fail clearly."""
    with pytest.raises(ValueError, match=message):
        BiSCNN(**kwargs)


def test_invalid_laplacian_count(simplicial_inputs):
    """The model rejects malformed TopoBench Laplacian tuples."""
    x_all, laplacian_all, _ = simplicial_inputs
    model = BiSCNN(
        in_channels_all=(3, 4, 5),
        hidden_channels_all=(5, 5, 5),
        sc_order=3,
    )
    with pytest.raises(ValueError, match="expected 5 Laplacians"):
        model(x_all, laplacian_all[:-1])
