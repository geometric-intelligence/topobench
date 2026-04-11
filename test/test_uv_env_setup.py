"""Tests for uv_env_setup.sh error handling."""

import subprocess
from pathlib import Path


def test_source_invalid_platform_does_not_kill_shell():
    """Verify sourcing with invalid platform does not kill the shell.

    If the script uses bare `exit 1`, sourcing it kills the shell
    (subshell exits). If it uses `return 1`, the subshell survives
    and the sentinel `STILL_ALIVE` is printed.
    """
    script = Path(__file__).parents[1] / "uv_env_setup.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {script} INVALID 2>/dev/null; echo STILL_ALIVE",
        ],
        capture_output=True,
        text=True,
    )
    assert "STILL_ALIVE" in result.stdout, (
        "Shell was killed by `exit 1` instead of `return 1`"
    )
