"""CUDA parity, mixed precision and compilation of the TRAWL backbone."""

import copy
import os
import sys

import pytest
import torch

from test._utils.trawl import collate, network, prepare
from topobench.nn.backbones.combinatorial.trawl_utils.layers import make_layer

# cuBLAS reads its workspace configuration when first used in a process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
requires_compiler = pytest.mark.skipif(
    not torch.cuda.is_available() or sys.platform == "win32",
    reason="Linux CUDA compiler validation",
)


@requires_cuda
@pytest.mark.parametrize("architecture", ["mamba", "sisa", "hybrid"])
def test_cuda_parity_and_amp_step(architecture):
    torch.manual_seed(42)
    graphs = [prepare(), prepare()]
    cpu = network(
        graphs,
        architecture=architecture,
        walk_refresh="fixed",
        checkpoint_layers=True,
    ).eval()
    gpu = copy.deepcopy(cpu).cuda()
    expected = cpu(collate(graphs))["graph_embedding"]
    actual = gpu(collate(graphs).to("cuda"))["graph_embedding"]
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, left), (_, right) in zip(
        cpu.named_parameters(), gpu.named_parameters(), strict=True
    ):
        if left.grad is not None:
            assert right.grad is not None, name
            torch.testing.assert_close(
                right.grad.cpu(), left.grad, atol=5e-4, rtol=5e-3
            )
    gpu.train()
    optimizer = torch.optim.Adam(gpu.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    before = next(gpu.backbone.encoders.parameters()).detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = (
            gpu(collate(graphs).to("cuda"))["graph_embedding"].square().mean()
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss)
    gradients = [p.grad for p in gpu.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, next(gpu.backbone.encoders.parameters()))


@requires_cuda
def test_official_mamba_backend():
    pytest.importorskip("mamba_ssm")
    graphs = [prepare(), prepare()]
    model = network(
        graphs,
        hidden_dim=32,
        depth=1,
        architecture="mamba",
        layer_options={
            "mamba": {
                "backend": "mamba_ssm",
                "d_state": 16,
                "scan": "parallel",
            }
        },
        walks={"k": 3, "length": 8},
        checkpoint_layers=True,
    )
    model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for precision in [torch.float32, torch.float16]:
        optimizer.zero_grad(set_to_none=True)
        before = next(model.backbone.encoders.parameters()).detach().clone()
        with torch.autocast(
            "cuda", dtype=precision, enabled=precision != torch.float32
        ):
            result = model(collate(graphs).to("cuda"))["graph_embedding"]
            loss = result.square().mean()
        loss.backward()
        assert result.shape[0] == len(graphs) and torch.isfinite(loss)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
        assert not torch.equal(
            before, next(model.backbone.encoders.parameters())
        )


@requires_compiler
@pytest.mark.parametrize("kind", ["mamba", "sisa"])
def test_compiled_cuda_layer(kind):
    torch.manual_seed(431)
    eager = make_layer({"kind": kind}, 32).cuda().eval()
    compiled = torch.compile(copy.deepcopy(eager), fullgraph=True)
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = eager(x), compiled(y)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=2e-4, rtol=2e-3)


@requires_compiler
def test_compiled_backbone_layers():
    graphs = [prepare(), prepare()]
    eager = network(
        graphs,
        hidden_dim=32,
        depth=1,
        architecture="mamba",
        walks={"k": 3, "length": 8},
    )
    eager.cuda().eval()
    compiled = copy.deepcopy(eager)
    for layer in compiled.compilable_layers():
        layer.compile(fullgraph=True)
    expected = eager(collate(graphs).to("cuda"))["graph_embedding"]
    actual = compiled(collate(graphs).to("cuda"))["graph_embedding"]
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(
        eager.parameters(), compiled.parameters(), strict=True
    ):
        if left.grad is not None:
            torch.testing.assert_close(
                right.grad, left.grad, atol=2e-4, rtol=2e-3
            )
