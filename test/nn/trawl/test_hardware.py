"""Device parity, mixed precision and bounded sparse-graph validation."""

import copy
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from topobench.data.utils.trawl.encodings import (
    guided_transition,
    positional_encodings,
)
from topobench.data.utils.trawl.sampling import WalkSampler
from topobench.nn.backbones.combinatorial.trawl import TRAWL, make_layer

from .test_trawl import collate, prepare


@pytest.mark.parametrize("occurrence_pooling", ["mean", "attention"])
@pytest.mark.parametrize("walk_scope", ["union", "separate"])
def test_cpu_autocast_accumulation(occurrence_pooling, walk_scope):
    graphs = [prepare(), prepare()]
    model = TRAWL(
        hidden_dim=16,
        depth=2,
        architecture="mlp",
        walks={"k": 3, "length": 5},
        walk_scope=walk_scope,
        occurrence_pooling=occurrence_pooling,
        graph_readout="cells",
    )
    model.initialize(graphs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(collate(graphs))
        loss = output["graph_embedding"].square().mean()
    assert torch.isfinite(loss)
    assert output["x_0"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("architecture", ["mamba", "sisa", "hybrid"])
def test_cuda_parity_and_amp_step(architecture):
    torch.manual_seed(42)
    options = dict(
        hidden_dim=16,
        depth=2,
        architecture=architecture,
        walks={"k": 3, "length": 5},
        walk_refresh="fixed",
        checkpoint_layers=True,
    )
    graphs = [prepare(), prepare()]
    cpu = TRAWL(**options)
    cpu.initialize(graphs)
    cpu.eval()
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
    before = next(gpu.encoders.parameters()).detach().clone()
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
    assert not torch.equal(before, next(gpu.encoders.parameters()))


def test_large_sparse_graph_and_spectral_limit():
    n = 10000
    rows = np.arange(n)
    matrix = sp.coo_matrix(
        (
            np.ones(2 * n),
            (
                np.concatenate([rows, rows]),
                np.concatenate([(rows + 1) % n, (rows - 1) % n]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    encoding = positional_encodings(matrix, rw_steps=2, rw_samples=2)
    assert encoding.shape == (n, 4)
    assert np.isfinite(encoding).all()
    paths = WalkSampler(0)(matrix, k=32, length=32)
    assert paths.shape == (32, 32)
    assert ((paths >= 0) & (paths < n)).all()
    with pytest.raises(ValueError, match="dense_limit"):
        positional_encodings(matrix, rw_steps=0, laplacian_dim=8)
    with pytest.raises(ValueError, match="dense_limit"):
        guided_transition(matrix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_official_mamba_backend():
    pytest.importorskip("mamba_ssm")
    options = dict(
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
    graphs = [prepare(), prepare()]
    model = TRAWL(**options)
    model.initialize(graphs)
    model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for precision in [torch.float32, torch.float16]:
        optimizer.zero_grad(set_to_none=True)
        before = next(model.encoders.parameters()).detach().clone()
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
        assert not torch.equal(before, next(model.encoders.parameters()))


@pytest.mark.skipif(
    not torch.cuda.is_available() or sys.platform == "win32",
    reason="Linux CUDA compiler validation",
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or sys.platform == "win32",
    reason="Linux CUDA compiler validation",
)
def test_compiled_backbone_layers():
    graphs = [prepare(), prepare()]
    eager = TRAWL(
        hidden_dim=32,
        depth=1,
        architecture="mamba",
        walks={"k": 3, "length": 8},
    )
    eager.initialize(graphs)
    eager.cuda().eval()
    compiled = copy.deepcopy(eager)
    for layer in compiled.encoders[0]:
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
