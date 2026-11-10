"""Tests for TRAWL's sequence layers."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from topobench.nn.backbones.combinatorial.trawl_utils.layers import (
    PureTorchMambaBlock,
    SISABlock,
    SISALayer,
    WalkGraphLayer,
    sequence_cumsum,
)

FIXTURE = Path(__file__).parent / "fixtures" / "trawl_reference.npz"


def test_mamba_scan_parity():
    parallel = PureTorchMambaBlock(8, d_state=4)
    sequential = copy.deepcopy(parallel)
    sequential.scan = "sequential"
    x = torch.randn(2, 7, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    first, second = parallel(x), sequential(y)
    torch.testing.assert_close(first, second)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)


def test_frozen_sisa_layer():
    """SISA matches frozen outputs of the original implementation."""
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        net = SISABlock(16, n_heads=2, d_ssm=4).eval()
        net.load_state_dict(
            {
                key.removeprefix("sisa::"): torch.from_numpy(fixture[key])
                for key in fixture.files
                if key.startswith("sisa::")
            }
        )
        actual = net(torch.from_numpy(fixture["sisa_input"]))
        torch.testing.assert_close(
            actual,
            torch.from_numpy(fixture["sisa_output"]),
            atol=2e-6,
            rtol=2e-6,
        )
        actual.square().mean().backward()
        parameters = dict(net.named_parameters())
        for key in fixture.files:
            if key.startswith("sisa_grad::"):
                torch.testing.assert_close(
                    parameters[key.split("::", 1)[1]].grad,
                    torch.from_numpy(fixture[key]),
                    atol=2e-6,
                    rtol=2e-5,
                )
        torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3).step()
        for key in fixture.files:
            if key.startswith("sisa_step::"):
                torch.testing.assert_close(
                    parameters[key.split("::", 1)[1]],
                    torch.from_numpy(fixture[key]),
                    atol=2e-6,
                    rtol=2e-5,
                )


def test_sisa_wide_decay_matches_and_stays_finite():
    torch.manual_seed(0)
    layer = SISALayer(32, n_heads=4, d_ssm=4, attention_dropout=0.0).eval()
    x = torch.randn(2, 32, 32)
    with torch.no_grad():
        factored = layer(x)
        layer.MAX_FACTORED_SPAN = -1.0
        explicit = layer(x)
    torch.testing.assert_close(factored, explicit, atol=1e-5, rtol=1e-5)
    layer.MAX_FACTORED_SPAN = 60.0
    nn.init.constant_(layer.alpha_proj.bias, 20.0)
    layer.train()
    x.requires_grad_(True)
    output = layer(x)
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("decay_bias", [-5.0, 0.0, 0.5, 2.0])
def test_sisa_float16_attention_stays_finite(decay_bias):
    """Learned decays that overflow float16 use the explicit path."""
    torch.manual_seed(0)
    layer = SISALayer(32, n_heads=4, d_ssm=4, attention_dropout=0.0).eval()
    nn.init.constant_(layer.alpha_proj.bias, decay_bias)
    x = torch.randn(4, 32, 32)
    with torch.no_grad():
        expected = layer(x)
        actual = layer.half()(x.half()).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_sequence_cumsum_matches_cumsum():
    x = torch.randn(2, 4, 32, 3)
    torch.testing.assert_close(sequence_cumsum(x, 2), torch.cumsum(x, 2))
    if torch.cuda.is_available():
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            result = sequence_cumsum(x.cuda(), 2)
        finally:
            torch.use_deterministic_algorithms(previous)
        torch.testing.assert_close(result.cpu(), torch.cumsum(x, 2))


def test_graph_adapter_never_crosses_walks():
    class SumNeighbors(nn.Module):
        def forward(self, x, edge_index):
            return torch.zeros_like(x).index_add(
                0, edge_index[1], x[edge_index[0]]
            )

    layer = WalkGraphLayer(SumNeighbors(), bidirectional=False)
    x = torch.arange(6.0).reshape(2, 3, 1)
    torch.testing.assert_close(
        layer(x).flatten(), torch.tensor([0.0, 0.0, 1.0, 0.0, 3.0, 4.0])
    )
