#!/bin/bash -l

# ==============================================================================
# 🛠️  TopoBench Environment Setup Script (Py3.11 + Dynamic CUDA)
# ==============================================================================
# usage: bash uv_env_setup.sh [cpu|cu118|cu121|cu128]
# ==============================================================================

PLATFORM="${1:-cpu}"

# Visual Header
echo ""
echo "======================================================="
echo "🚀 Initializing TopoBench Environment ($PLATFORM)"
echo "======================================================="

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
case "$PLATFORM" in
    cpu|cu118|cu121|cu128)
        TARGET_INDEX="pytorch-${PLATFORM}"
        ;;
    *)
        echo "❌ Error: Invalid platform '$PLATFORM'. Use: cpu, cu118, cu121, or cu128."
        return 1 2>/dev/null || exit 1
        ;;
esac

echo "⚙️  Updating pyproject.toml..."

# Update the torch source index for Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/index = \"pytorch-[a-z0-9]*\", marker = \"sys_platform == 'linux'/index = \"${TARGET_INDEX}\", marker = \"sys_platform == 'linux'/g" pyproject.toml
else
    sed -i "s/index = \"pytorch-[a-z0-9]*\", marker = \"sys_platform == 'linux'/index = \"${TARGET_INDEX}\", marker = \"sys_platform == 'linux'/g" pyproject.toml
fi

echo "✅ Set Torch Index to: ${TARGET_INDEX}"

# ------------------------------------------------------------------------------
# Sync
# ------------------------------------------------------------------------------
echo ""
echo "🧹 Cleaning old lockfile..."
rm -f uv.lock

# Dry-run to detect which torch version will be installed
TORCH_VER=$(uv sync --dry-run --python 3.11 2>&1 \
    | grep '+ torch==' | sed 's/.*+ torch==//')
if [ -z "$TORCH_VER" ]; then
    # Fallback: read from existing venv (dry-run reports nothing if already installed)
    TORCH_VER=$(.venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null)
fi
if [ -z "$TORCH_VER" ]; then
    echo "❌ Error: Could not detect torch version."
    return 1 2>/dev/null || exit 1
fi
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}.html"
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s|find-links = \[\".*\"\]|find-links = [\"${PYG_URL}\"]|g" pyproject.toml
else
    sed -i "s|find-links = \[\".*\"\]|find-links = [\"${PYG_URL}\"]|g" pyproject.toml
fi
echo "✅ Set PyG Links to : ${PYG_URL} (torch ${TORCH_VER})"

echo "📦 Syncing Environment (Python 3.11)..."
if ! uv sync --python 3.11 --all-extras; then
    echo "❌ uv sync failed."
    return 1 2>/dev/null || exit 1
fi

# ------------------------------------------------------------------------------
# Finalize
# ------------------------------------------------------------------------------
source .venv/bin/activate
echo ""
echo "🔧 Configuring Git Hooks..."
uv pip install pre-commit
pre-commit install

echo ""
echo "======================================================="
echo "🎉 Setup Complete!"
echo "======================================================="
python -c "import sys; import torch; print(f'✅ Python Ver    : {sys.version.split()[0]}'); print(f'✅ Torch Version : {torch.__version__}'); print(f'✅ CUDA Available: {torch.cuda.is_available()}'); print(f'✅ CUDA Version  : {torch.version.cuda}')"
echo "======================================================="
