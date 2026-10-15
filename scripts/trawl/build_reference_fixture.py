"""Regenerate the SISA reference fixture from a trusted local TRAWL checkout.

Usage: python scripts/trawl/build_reference_fixture.py --source-root ../
Only selected mathematical classes/functions are executed; original training
scripts are never imported. The generated fixture has no pickle objects.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def definitions(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    return "\n\n".join(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in names
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test/nn/trawl/fixtures/reference.npz"),
    )
    args = parser.parse_args()
    sisa_source = (
        args.source_root
        / "snapshots/proteins_peak224_richfeat_hybrid_learnable_gate/trawl_proteins.py"
    )
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    values = {}
    exec(
        definitions(
            sisa_source,
            {"build_rope_cache", "apply_rope", "SISALayer", "SISABlock"},
        ),
        namespace,
    )
    torch.manual_seed(782)
    sisa = namespace["SISABlock"](16, n_heads=2, d_ssm=4).eval()
    tokens = torch.randn(2, 7, 16)
    values["sisa_input"] = tokens.numpy()
    with torch.no_grad():
        values["sisa_output"] = sisa(tokens).numpy()
    for key, tensor in sisa.state_dict().items():
        values["sisa::" + key] = tensor.numpy().copy()
    sisa(tokens).square().mean().backward()
    for key, parameter in sisa.named_parameters():
        if parameter.grad is not None:
            values["sisa_grad::" + key] = parameter.grad.numpy().copy()
    torch.optim.Adam(sisa.parameters(), lr=1e-4, weight_decay=1e-3).step()
    for key, parameter in sisa.named_parameters():
        if parameter.grad is not None:
            values["sisa_step::" + key] = parameter.detach().numpy().copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    metadata = {
        "sources": {
            str(path.relative_to(args.source_root)).replace(
                "\\", "/"
            ): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [sisa_source]
        },
        "torch": torch.__version__,
        "seed": [782],
        "scope": "SISA block forward, gradient and Adam-step parity with the original implementation.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
