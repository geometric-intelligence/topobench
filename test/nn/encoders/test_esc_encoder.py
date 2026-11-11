import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.encoders.esc_encoder import ESCFeatureEncoder


def _valid_data() -> Data:
    return Data(
        x=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        batch_0=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        esc_code_id=torch.tensor([0, 5, 386, 12], dtype=torch.long),
        esc_code_count=torch.tensor([1.0, 2.0, 3.0, 1.0], dtype=torch.float32),
        esc_nnz_per_edge=torch.tensor([2, 1, 0, 1], dtype=torch.long),
    )


def test_missing_cached_field_has_actionable_error():
    data = _valid_data()
    del data.esc_code_id
    encoder = ESCFeatureEncoder(in_channels=[3], out_channels=64)

    with pytest.raises(
        ValueError,
        match=r"missing cached ESC field.*transforms=model_defaults/esc_gnn",
    ):
        encoder(data)

    assert data.get("x_0") is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "esc_code_id",
            torch.tensor([0, 5, 386, 12], dtype=torch.int32),
            "esc_code_id must have dtype torch.long",
        ),
        (
            "esc_code_count",
            torch.tensor([1.0, 2.0, 3.0, 1.0], dtype=torch.float64),
            "esc_code_count must have dtype torch.float32",
        ),
        (
            "esc_nnz_per_edge",
            torch.tensor([2, 1, 1], dtype=torch.long),
            r"len\(esc_nnz_per_edge\)",
        ),
        (
            "esc_nnz_per_edge",
            torch.tensor([2, 1, 0, 0], dtype=torch.long),
            r"sum\(esc_nnz_per_edge\)",
        ),
        (
            "esc_nnz_per_edge",
            torch.tensor([2, 1, -1, 2], dtype=torch.long),
            "nonnegative integers",
        ),
        (
            "esc_code_count",
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "equal length",
        ),
        (
            "esc_code_id",
            torch.tensor([0, 5, 387, 12], dtype=torch.long),
            r"values must lie in \[0, 387\)",
        ),
        (
            "esc_code_count",
            torch.tensor([1.0, 2.5, 3.0, 1.0], dtype=torch.float32),
            "positive integer-valued counts",
        ),
        (
            "esc_code_count",
            torch.tensor([1.0, float("inf"), 3.0, 1.0]),
            "finite values",
        ),
    ],
)
def test_rejects_malformed_cached_fields(field, value, message):
    data = _valid_data()
    data[field] = value
    encoder = ESCFeatureEncoder(in_channels=[3], out_channels=64)

    with pytest.raises(ValueError, match=message):
        encoder(data)
