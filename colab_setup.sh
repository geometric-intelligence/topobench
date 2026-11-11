#!/usr/bin/env bash
# ==============================================================================
# TopoBench — Colab bootstrap script
# ==============================================================================
# Clones a branch of TopoBench and installs it in an isolated Python 3.11 venv
# (the project requires Python >=3.11,<3.12; Colab's own kernel may be newer).
#
# Usage (paste as the first Colab cell):
#   %%bash
#   BRANCH=dphgnn_gf bash -c "$(curl -fsSL <raw-url-to-this-file>)"
# or, after uploading/copy-pasting this file into Colab:
#   !bash colab_setup.sh
#
# Env vars you can override:
#   REPO_URL  (default: https://github.com/yeli-falk/topobench.git)
#   BRANCH    (default: dphgnn_gf)
#   DEST      (default: /content/topobench)
# ==============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/yeli-falk/topobench.git}"
BRANCH="${BRANCH:-dphgnn_gf}"
DEST="${DEST:-/content/topobench}"

echo "==> Cloning ${REPO_URL} @ ${BRANCH} into ${DEST}"
if [ -d "${DEST}/.git" ]; then
    git -C "${DEST}" fetch origin "${BRANCH}"
    git -C "${DEST}" checkout "${BRANCH}"
    git -C "${DEST}" reset --hard "origin/${BRANCH}"
else
    git clone --branch "${BRANCH}" "${REPO_URL}" "${DEST}"
fi
cd "${DEST}"

echo "==> Installing uv"
pip install -q -U uv

echo "==> Creating an isolated Python 3.11 venv at ${DEST}/.venv"
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Syncing dependencies (torch/PyG default to the cu121 index, matching Colab GPUs)"
uv sync --all-extras

echo "==> Registering the venv as a Jupyter kernel"
python -m ipykernel install --user --name=topobench --display-name "Python 3.11 (topobench)"

echo ""
echo "==> Setup complete."
python -c "import torch; print(f'Torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "NOTE: Colab's hosted runtime cannot switch its notebook kernel to a custom"
echo "venv, so run heavy commands through the venv's interpreter explicitly, e.g.:"
echo "  !${DEST}/.venv/bin/python -m topobench model=hypergraph/dphgnn"
echo "  !${DEST}/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \\"
echo "      ${DEST}/2026_tdl_challenge/run_evaluation.ipynb"
