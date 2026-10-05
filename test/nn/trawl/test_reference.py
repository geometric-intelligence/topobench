"""Numerical parity of SISA against frozen outputs of the original implementation."""

from pathlib import Path

import numpy as np
import torch

from topobench.nn.backbones.combinatorial.trawl import SISABlock


def test_frozen_sisa_layer():
    with np.load(
        Path(__file__).parent / "fixtures/reference.npz", allow_pickle=False
    ) as fixture:
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
