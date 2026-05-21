#!/usr/bin/env bash
# ============================================================================
# Create the conda environment `sliders_flux` for Flux Concept Sliders +
# the LoRAShop-based pipeline.
#
# Target hardware: NVIDIA H100 / H200 (Hopper, SM 9.0)
# CUDA:            12.4  (cu124 wheels, requires CUDA driver >= 12.0)
# Python:          3.11
# PyTorch:         2.4.1
#
# The resulting environment covers every Flux task in the repo:
#   - Concept Sliders training (flux/trained_sliders/training/)
#   - Mask-guided generation / editing (flux/tasks/masked_lora/)
#   - LoRAShop-style multi-slider pipeline (flux/tasks/shop_concept/)
#   - CLIP + LPIPS evaluation (metrics/)
#
# Usage:
#   bash flux/trained_sliders/training/setup_sliders_flux_conda.sh
#
# After installation, point `tools/set_slurms.sh` at this environment:
#   activate_flux_env() {
#       source /path/to/conda/etc/profile.d/conda.sh
#       conda activate sliders_flux
#   }
# ============================================================================
set -euo pipefail

ENV_NAME="sliders_flux"
PYTHON_VERSION="3.11"

# ---------------------------------------------------------------------------
# Detect conda / mamba
# ---------------------------------------------------------------------------
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
    echo "[setup] Using mamba (faster)"
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
    echo "[setup] Using conda"
else
    echo "[setup] conda not in PATH — trying module load ..."
    if module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null; then
        CONDA_CMD="conda"
        echo "[setup] Loaded via module load"
    else
        echo ""
        echo "ERROR: conda not found."
        echo "  Try: module avail 2>&1 | grep -i conda"
        echo "  Then: module load <module-name>"
        echo "  Then re-run this script."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Determine the conda base path
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: unable to determine the conda base path."
    exit 1
fi
echo "[setup] conda base: $CONDA_BASE"

# ---------------------------------------------------------------------------
# Remove the env if it already exists
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Env '${ENV_NAME}' exists — removing and recreating"
    conda env remove -n "$ENV_NAME" -y
fi

# ---------------------------------------------------------------------------
# 1) Create env with Python 3.11
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} ..."
$CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo "[1/5] OK"

# Activate
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "[setup] Env activated: $(python --version)"

pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2) PyTorch 2.4.1 with CUDA 12.4 (H100 / H200 compatible)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Installing PyTorch 2.4.1 + CUDA 12.4 ..."
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 \
    torchvision==0.19.1

# ---------------------------------------------------------------------------
# 3) diffusers stack (training + LoRAShop-style + masked_lora)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installing diffusers / transformers / peft / accelerate ..."
pip install \
    "diffusers==0.36.0" \
    "transformers==4.57.6" \
    "accelerate==1.10.1" \
    "peft==0.17.1" \
    "bitsandbytes>=0.44.0" \
    safetensors \
    sentencepiece \
    protobuf \
    einops

# ---------------------------------------------------------------------------
# 4) LoRAShop deps + metrics + utilities
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Installing LoRAShop deps + metrics (CLIP/LPIPS) + utilities ..."
pip install \
    lpips \
    opencv-python \
    scipy \
    matplotlib \
    pandas \
    numpy \
    Pillow \
    tqdm \
    pyyaml \
    wandb \
    datasets \
    ftfy \
    scikit-learn \
    requests \
    prodigyopt \
    dadaptation \
    jupyter \
    ipykernel \
    ipywidgets

# ---------------------------------------------------------------------------
# 5) Lock file
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="$SCRIPT_DIR/requirements-sliders-flux-conda.lock"

echo ""
echo "[5/5] Saving lock file at $LOCK_FILE ..."
{
    echo "# ============================================================================"
    echo "# requirements-sliders-flux-conda.lock"
    echo "# ============================================================================"
    echo "# Snapshot of conda env '${ENV_NAME}' captured at $(date -Iseconds)"
    echo "#"
    echo "# Python: $(python --version 2>&1 | cut -d' ' -f2)"
    echo "# Env:    $ENV_NAME"
    echo "# CUDA:   torch $(python -c 'import torch; print(torch.__version__)') (cuda: $(python -c 'import torch; print(torch.cuda.is_available())'))"
    echo "# GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
    echo "#"
    echo "# To recreate (same hardware):"
    echo "#   conda create -n sliders_flux python=3.11 -y"
    echo "#   conda activate sliders_flux"
    echo "#   pip install -r requirements-sliders-flux-conda.lock"
    echo "# ============================================================================"
    pip freeze
} > "$LOCK_FILE"
echo "[5/5] OK"

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
echo ""
echo "=== Installation check ==="
python -c "
import torch, diffusers, transformers, peft, accelerate, lpips, cv2
print(f'torch        {torch.__version__}')
print(f'  cuda avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device     : {torch.cuda.get_device_name(0)}')
    print(f'  cuda vers  : {torch.version.cuda}')
print(f'diffusers    {diffusers.__version__}')
print(f'transformers {transformers.__version__}')
print(f'peft         {peft.__version__}')
print(f'accelerate   {accelerate.__version__}')
print(f'lpips        {lpips.__version__}')
print(f'opencv       {cv2.__version__}')
try:
    import bitsandbytes as bnb
    print(f'bitsandbytes {bnb.__version__}')
except Exception as e:
    print(f'bitsandbytes NOT OK: {e}')
"

echo ""
echo "=== Setup completed ==="
echo ""
echo "Update tools/set_slurms.sh with:"
echo ""
echo "  activate_flux_env() {"
echo "      source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "      conda activate ${ENV_NAME}"
echo "  }"
echo ""
echo "To activate manually:"
echo "  source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
