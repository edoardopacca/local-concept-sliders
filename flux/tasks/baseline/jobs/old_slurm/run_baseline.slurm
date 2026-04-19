#!/bin/bash
#SBATCH --job-name=flux_baseline
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p flux/tasks/baseline/logs flux/tasks/baseline/outputs

# --- CUSTOMIZE ---
# Path to the trained slider directory (output of train_flux_slider.py)
LORA_DIR="flux/trained_sliders/sliders/<slider_name>"

RUN_NAME="baseline_demo_$(date +%Y%m%d_%H%M)"
SAVE_DIR="flux/tasks/baseline/outputs/${RUN_NAME}"

# Slider rank/alpha must match the values used at training time.
RANK=16
ALPHA=1.0

# Bidirectional scale sweep
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --lora_dirs "${LORA_DIR}" \
    --lora_scales -2 -1 0 1 2 \
    --prompt "a portrait photo of a person, studio lighting" \
    --seeds 42 \
    --save_dir "${SAVE_DIR}" \
    --height 1024 --width 1024 \
    --steps 30 \
    --guidance_scale 3.5 \
    --rank "${RANK}" \
    --alpha "${ALPHA}" \
    --train_method xattn

echo "[done] Output saved to ${SAVE_DIR}"
