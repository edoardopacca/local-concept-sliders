#!/bin/bash
#SBATCH --job-name=sdxl_baseline
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=sdxl/tasks/baseline/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/baseline/logs/%x_%j.err

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p sdxl/tasks/baseline/logs sdxl/tasks/baseline/outputs

# --- CUSTOMIZE ---
SLIDER="sdxl/trained_sliders/sliders/smiling.pt"
PROMPT="a portrait photo of a person, studio lighting, 50mm"
RUN_NAME="smiling_demo_$(date +%Y%m%d_%H%M)"
SAVE_PATH="sdxl/tasks/baseline/outputs/${RUN_NAME}"
SEED=42

# Slider rank/alpha must match the values used at training time.
RANK=4
ALPHA=1.0

# Bidirectional scale sweep: -2 = strong reverse, 0 = baseline (no LoRA), 2 = strong forward
python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
    --prompt "${PROMPT}" \
    --slider "${SLIDER}" \
    --save_path "${SAVE_PATH}" \
    --scales -2 -1 0 1 2 \
    --start_noise 700 \
    --seed "${SEED}" \
    --rank "${RANK}" \
    --alpha "${ALPHA}" \
    --steps 50

echo "[done] Output saved to ${SAVE_PATH}"
