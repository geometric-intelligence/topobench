"""TRAWL test session setup."""

import os

# cuBLAS reads its workspace configuration when first used in a process, so
# deterministic-mode tests need it before any earlier test touches CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
