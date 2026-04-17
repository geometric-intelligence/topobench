"""Unit tests for the tutorials."""

import glob
import subprocess
import tempfile

import pytest


def _exec_tutorial(path):
    """Execute a tutorial notebook.

    Parameters
    ----------
    path : str
        The path to the tutorial.
    """
    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1000",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        file_name,
        path,
    ]
    subprocess.check_call(args)


paths = sorted(glob.glob("tutorials/*.ipynb"))


@pytest.mark.parametrize("path", paths)
def test_tutorial(path):
    """Run the test of the tutorials.

    Parameters
    ----------
    path : str
        The path to the tutorial.
    """
    _exec_tutorial(path)
