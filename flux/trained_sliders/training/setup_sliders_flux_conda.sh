#!/usr/bin/env bash
# ============================================================================
# setup_sliders_flux_conda.sh
# ============================================================================
# Crea da zero il conda env `sliders_flux` per Flux Concept Sliders + LoRAShop.
#
# Target hardware: NVIDIA H100 / H200 (Hopper, SM 9.0)
# CUDA:           12.4  (wheel cu124, supportato da CUDA driver >= 12.0)
# Python:         3.11
# PyTorch:        2.4.1
#
# Contiene tutto il necessario per:
#   - Training Flux concept sliders (flux/trained_sliders/training/)
#   - Generazione / masked LoRA edit (flux/tasks/masked_lora/)
#   - LoRAShop multi-slider (flux/tasks/shop_concept/)
#   - Evaluation CLIP + LPIPS (metrics/)
#
# Uso:
#   bash flux/trained_sliders/training/setup_sliders_flux_conda.sh
#
# Dopo il setup aggiorna tools/set_slurms.sh con:
#   activate_flux_env() {
#       source /path/to/conda/etc/profile.d/conda.sh
#       conda activate sliders_flux
#   }
# ============================================================================
set -euo pipefail

ENV_NAME="sliders_flux"
PYTHON_VERSION="3.11"

# ---------------------------------------------------------------------------
# Rileva conda / mamba
# ---------------------------------------------------------------------------
if command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
    echo "[setup] Usando mamba (più veloce)"
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
    echo "[setup] Usando conda"
else
    # Prova a caricare via module
    echo "[setup] conda non in PATH — provo module load ..."
    if module load miniconda3 2>/dev/null || module load anaconda3 2>/dev/null; then
        CONDA_CMD="conda"
        echo "[setup] Caricato via module load"
    else
        echo ""
        echo "ERROR: conda non trovato."
        echo "  Prova: module avail 2>&1 | grep -i conda"
        echo "  Poi:   module load <nome-modulo>"
        echo "  Poi rilancia questo script."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Individua il path base di conda per init
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: impossibile determinare conda base path."
    exit 1
fi
echo "[setup] conda base: $CONDA_BASE"

# ---------------------------------------------------------------------------
# Rimuovi env esistente se presente
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Env '${ENV_NAME}' esistente — lo rimuovo e ricreo"
    conda env remove -n "$ENV_NAME" -y
fi

# ---------------------------------------------------------------------------
# 1) Crea env con Python 3.11
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Creo conda env '${ENV_NAME}' con Python ${PYTHON_VERSION}..."
$CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo "[1/5] OK"

# Attiva l'env
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "[setup] Env attivato: $(python --version)"

# Aggiorna pip
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2) PyTorch 2.4.1 con CUDA 12.4 (H100 / H200 compatibile)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Installo PyTorch 2.4.1 + CUDA 12.4..."
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 \
    torchvision==0.19.1

# ---------------------------------------------------------------------------
# 3) Diffusers stack (training + LoRAShop + masked_lora)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installo diffusers / transformers / peft / accelerate..."
pip install \
    "diffusers>=0.31.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.34.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.44.0" \
    safetensors \
    sentencepiece \
    protobuf \
    einops

# ---------------------------------------------------------------------------
# 4) Dipendenze LoRAShop + metriche + utility
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Installo LoRAShop deps + metrics (CLIP/LPIPS) + utility..."
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
    openai \
    anthropic \
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
echo "[5/5] Salvo lock file in $LOCK_FILE..."
{
    echo "# ============================================================================"
    echo "# requirements-sliders-flux-conda.lock"
    echo "# ============================================================================"
    echo "# Snapshot conda env '${ENV_NAME}' creato il $(date -Iseconds)"
    echo "#"
    echo "# Python: $(python --version 2>&1 | cut -d' ' -f2)"
    echo "# Env:    $ENV_NAME"
    echo "# CUDA:   torch $(python -c 'import torch; print(torch.__version__)') (cuda: $(python -c 'import torch; print(torch.cuda.is_available())'))"
    echo "# GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
    echo "#"
    echo "# Per ricreare (stesso hardware):"
    echo "#   conda create -n sliders_flux python=3.11 -y"
    echo "#   conda activate sliders_flux"
    echo "#   pip install -r requirements-sliders-flux-conda.lock"
    echo "# ============================================================================"
    pip freeze
} > "$LOCK_FILE"
echo "[5/5] OK"

# ---------------------------------------------------------------------------
# Verifica finale
# ---------------------------------------------------------------------------
echo ""
echo "=== Verifica installazione ==="
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
echo "=== Setup completato ==="
echo ""
echo "Aggiorna tools/set_slurms.sh con:"
echo ""
echo "  activate_flux_env() {"
echo "      source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "      conda activate ${ENV_NAME}"
echo "  }"
echo ""
echo "Per attivare manualmente:"
echo "  source ${CONDA_BASE}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
