#!/usr/bin/env bash
# ============================================================================
# setup_flux_venv.sh
# ============================================================================
# Crea da zero il venv Python per Flux Concept Sliders (bf16, no quantizzazione).
#
# Strategia:
#   - Python 3.10+ (requisito diffusers recente + peft + bitsandbytes moderni)
#   - Torch 2.4+ con CUDA 12.4 wheels (compat Ampere A100 e futuro Hopper)
#   - diffusers ultima (Flux pipeline ufficiale)
#   - transformers 4.44+ (T5 + CLIP per Flux)
#   - peft + accelerate (LoRA training)
#   - bitsandbytes (opzionale, per eventuali esperimenti NF4 futuri)
#
# Post-setup salva `pip freeze` in FLUX_train/requirements-flux.lock per
# riproducibilita' esatta (come fatto per SDXL).
#
# Uso:
#   bash flux/trained_sliders/training/setup_flux_venv.sh
#
# ============================================================================
set -euo pipefail

VENV_PATH="$HOME/Linux4HPC/venvs/flux-sliders"
REPO_DIR="$HOME/Linux4HPC/sliders_demo/local-concept-sliders"
LOCK_FILE="$REPO_DIR/flux/trained_sliders/training/requirements-flux.lock"

echo "=== Flux sliders venv setup ==="
echo "Venv path:    $VENV_PATH"
echo "Repo dir:     $REPO_DIR"
echo "Lock file:    $LOCK_FILE"
echo ""

# -----------------------------------------------------------------------------
# 1) Sanity check: Python 3.10+ disponibile
# -----------------------------------------------------------------------------
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/6] python3 --version: $PY_VER"

MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]; }; then
    echo ""
    echo "ATTENZIONE: Python $PY_VER rilevato, serve >= 3.9 per diffusers+Flux."
    echo "Prova:"
    echo "  module load miniconda3"
    echo "  conda create -n py311 python=3.11 -y && conda activate py311"
    echo "Poi rilancia questo script."
    exit 1
fi
if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -eq 9 ]; then
    echo "      Nota: Python 3.9 funziona con diffusers>=0.30 ma non e' il target"
    echo "      ideale. Se in futuro servisse 3.10+ usa 'module load miniconda3'."
fi

# -----------------------------------------------------------------------------
# 2) Crea venv
# -----------------------------------------------------------------------------
if [ -d "$VENV_PATH" ]; then
    echo "[2/6] Venv esistente in $VENV_PATH - lo cancello e ricreo"
    rm -rf "$VENV_PATH"
fi
mkdir -p "$(dirname "$VENV_PATH")"
python3 -m venv "$VENV_PATH"
echo "[2/6] Venv creato in $VENV_PATH"

# -----------------------------------------------------------------------------
# 3) Attiva + upgrade pip
# -----------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel
echo "[3/6] pip aggiornato: $(pip --version)"

# -----------------------------------------------------------------------------
# 4) Installa PyTorch con CUDA 12.4 (Ampere + Hopper compat)
# -----------------------------------------------------------------------------
echo "[4/6] Installo torch 2.4+ con CUDA 12.4..."
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 torchvision==0.19.1

# -----------------------------------------------------------------------------
# 5) Installa resto delle dipendenze
# -----------------------------------------------------------------------------
echo "[5/6] Installo diffusers/transformers/peft/accelerate + notebook deps..."
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
# 6) Freeze in lock file per riproducibilita'
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "$LOCK_FILE")"
{
    echo "# ============================================================================"
    echo "# requirements-flux.lock"
    echo "# ============================================================================"
    echo "# Snapshot del venv flux-sliders al $(date -Iseconds)"
    echo "#"
    echo "# Python: $(python --version 2>&1 | cut -d' ' -f2)"
    echo "# Venv:   $VENV_PATH"
    echo "# CUDA:   torch $(python -c 'import torch; print(torch.__version__)') (cuda disponibile: $(python -c 'import torch; print(torch.cuda.is_available())'))"
    echo "#"
    echo "# Per ricreare:"
    echo "#   source \$NEW_VENV/bin/activate"
    echo "#   pip install -r requirements-flux.lock"
    echo "# ============================================================================"
    pip freeze
} > "$LOCK_FILE"

echo ""
echo "[6/6] Lock file salvato in $LOCK_FILE"
echo ""
echo "=== Verifica ==="
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
echo "Per attivare il venv in futuro:"
echo "  source $VENV_PATH/bin/activate"
