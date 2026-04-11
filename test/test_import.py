"""Tests for package import behavior."""

import subprocess
import sys


def test_import_without_sparse():
    """Verify topobench imports without torch_sparse/scatter/cluster.

    Runs in a subprocess that blocks the sparse imports via sys.modules
    to avoid modifying the test environment.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "sys.modules['torch_sparse'] = None;"
            "sys.modules['torch_scatter'] = None;"
            "sys.modules['torch_cluster'] = None;"
            "import topobench;"
            "from topobench.nn.backbones.graph import BACKBONE_CLASSES;"
            "from topobench.nn.backbones.hypergraph import BACKBONE_CLASSES;"
            "print('ok')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import topobench failed without sparse packages:\n{result.stderr}"
    )
    assert "ok" in result.stdout
