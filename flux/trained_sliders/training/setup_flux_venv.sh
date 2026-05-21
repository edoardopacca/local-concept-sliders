#!/usr/bin/env bash
# ============================================================================
# Create a Python venv for Flux Concept Sliders training (bf16, no
# quantisation).
#
# Strategy:
#   - Python 3.10+ (required by recent diffusers + peft + bitsandbytes)
#   - Torch 2.4+ with CUDA 12.4 wheels (Ampere A100 / Hopper compatible)
#   - Latest diffusers (Flux pipeline)
#   - transformers 4.44+ (T5 + CLIP for Flux)
#   - peft + accelerate (LoRA training)
#   - bitsandbytes (optional, for future NF4 experiments)
#
# After installation, `pip freeze` is saved to requirements-flux.lock for
# exact reproducibility.
#
# Usage:
#   bash flux/trained_sliders/training/setup_flux_venv.sh
#
# Override VENV_PATH and REPO_DIR below if your local setup differs.
# ============================================================================
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/venvs/flux-sliders}"
REPO_DIR="${REPO_DIR:-$HOME/local-concept-sliders}"
LOCK_FILE="$REPO_DIR/flux/trained_sliders/training/requirements-flux.lock"

echo "=== Flux sliders venv setup ==="
echo "Venv path:    $VENV_PATH"
echo "Repo dir:     $REPO_DIR"
echo "Lock file:    $LOCK_FILE"
echo ""

# -----------------------------------------------------------------------------
# 1) Sanity check: Python 3.10+ available
# -----------------------------------------------------------------------------
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/6] python3 --version: $PY_VER"

MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]; }; then
    echo ""
    echo "WARNING: Python $PY_VER detected, need >= 3.9 for diffusers + Flux."
    echo "Try:"
    echo "  module load miniconda3"
    echo "  conda create -n py311 python=3.11 -y && conda activate py311"
    echo "Then re-run this script."
    exit 1
fi
if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -eq 9 ]; then
    echo "      Note: Python 3.9 works with diffusers>=0.30 but is not the"
    echo "      ideal target. Prefer Python 3.10/3.11 if available."
fi

# -----------------------------------------------------------------------------
# 2) Create venv
# -----------------------------------------------------------------------------
if [ -d "$VENV_PATH" ]; then
    echo "[2/6] Venv already exists at $VENV_PATH - removing and recreating"
    rm -rf "$VENV_PATH"
fi
mkdir -p "$(dirname "$VENV_PATH")"
python3 -m venv "$VENV_PATH"
echo "[2/6] Venv created at $VENV_PATH"

# -----------------------------------------------------------------------------
# 3) Activate + upgrade pip
# -----------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel
echo "[3/6] pip upgraded: $(pip --version)"

# -----------------------------------------------------------------------------
# 4) Install PyTorch with CUDA 12.4 (Ampere + Hopper compatible)
# -----------------------------------------------------------------------------
echo "[4/6] Installing torch 2.4+ with CUDA 12.4 ..."
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 torchvision==0.19.1

# -----------------------------------------------------------------------------
# 5) Install the remaining dependencies
# -----------------------------------------------------------------------------
echo "[5/6] Installing diffusers / transformers / peft / accelerate + utils ..."
pip install \
    "diffusers>=0.31.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.34.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.44.0" \
    safetensors \
    sentencepiece \
    protobuf \
    einops \
    opencv-python \
    lpips \
    wandb \
    tqdm \
    pyyaml \
    matplotlib \
    pandas \
    scipy \
    pillow \
    jupyter \
    ipykernel \
    ipywidgets

# -----------------------------------------------------------------------------
# 6) Freeze into the lock file for reproducibility
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "$LOCK_FILE")"
{
    echo "# ============================================================================"
    echo "# requirements-flux.lock"
    echo "# ============================================================================"
    echo "# Snapshot of the flux-sliders venv at $(date -Iseconds)"
    echo "#"
    echo "# Python: $(python --version 2>&1 | cut -d' ' -f2)"
    echo "# Venv:   $VENV_PATH"
    echo "# CUDA:   torch $(python -c 'import torch; print(torch.__version__)') (cuda available: $(python -c 'import torch; print(torch.cuda.is_available())'))"
    echo "#"
    echo "# To recreate:"
    echo "#   source \$NEW_VENV/bin/activate"
    echo "#   pip install -r requirements-flux.lock"
    echo "# ============================================================================"
    pip freeze
} > "$LOCK_FILE"

echo ""
echo "[6/6] Lock file saved at $LOCK_FILE"
echo ""
echo "=== Verification ==="
python -c "
import torch, diffusers, transformers, peft, accelerate
print(f'torch        {torch.__version__}, cuda avail: {torch.cuda.is_available()}')
print(f'diffusers    {diffusers.__version__}')
print(f'transformers {transformers.__version__}')
print(f'peft         {peft.__version__}')
print(f'accelerate   {accelerate.__version__}')
try:
    import bitsandbytes
    print(f'bitsandbytes {bitsandbytes.__version__}')
except Exception as e:
    print(f'bitsandbytes NOT OK: {e}')
"

echo ""
echo "=== Done ==="
echo "To activate the venv in the future:"
echo "  source $VENV_PATH/bin/activate"
