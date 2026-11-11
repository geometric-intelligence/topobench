"""Tests for learned copresheaf map families."""

import pytest
import torch

from topobench.nn.layers.copresheaf import (
    BaseCopresheafMap,
    create_copresheaf_map,
)


@pytest.mark.parametrize(
    "map_type",
    ["identity", "diagonal", "full", "shared_local", "outer_product"],
)
def test_copresheaf_maps_have_expected_shape_and_finite_gradients(map_type):
    """Every catalogue map is batched, multi-head, and differentiable."""
    source = torch.randn(7, 8, requires_grad=True)
    target = torch.randn(7, 8, requires_grad=True)
    transport = create_copresheaf_map(
        map_type, channels=8, heads=2, stalk_dimension=4
    )

    matrices = transport(source, target)

    assert isinstance(transport, BaseCopresheafMap)
    assert matrices.shape == (7, 2, 4, 4)
    assert torch.isfinite(matrices).all()
    if map_type == "identity":
        assert not matrices.requires_grad
    else:
        matrices.sum().backward()
        assert source.grad is not None


def test_identity_map_returns_exact_identity():
    """The identity ablation must not alter transported stalk values."""
    transport = create_copresheaf_map(
        "identity", channels=6, heads=3, stalk_dimension=2
    )
    matrices = transport(torch.randn(5, 6), torch.randn(5, 6))
    expected = torch.eye(2).expand(5, 3, 2, 2)
    torch.testing.assert_close(matrices, expected)


def test_diagonal_map_has_no_off_diagonal_entries():
    """The GCN/SAGE map family remains diagonal after learning."""
    transport = create_copresheaf_map(
        "diagonal", channels=4, heads=1, stalk_dimension=4
    )
    matrices = transport(torch.randn(3, 4), torch.randn(3, 4))
    off_diagonal = matrices - torch.diag_embed(
        matrices.diagonal(dim1=-2, dim2=-1)
    )
    torch.testing.assert_close(off_diagonal, torch.zeros_like(off_diagonal))


def test_invalid_map_configuration_has_actionable_error():
    """Invalid family names and incompatible stalk dimensions fail early."""
    with pytest.raises(ValueError, match="unknown copresheaf map"):
        create_copresheaf_map("mystery", channels=4)
    with pytest.raises(ValueError, match=r"heads \* stalk_dimension"):
        create_copresheaf_map(
            "identity", channels=7, heads=2, stalk_dimension=4
        )
