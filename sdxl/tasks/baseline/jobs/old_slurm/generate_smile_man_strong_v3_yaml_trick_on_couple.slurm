#!/bin/bash
#SBATCH --job-name=gen_v3_yaml_trick_couple
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

# Inference grid for v3_yaml_trick — same seed / prompt / scale sweep
# as the v2_guidance4 and v3_preserve_woman couple runs. Three grids
# will be directly comparable cell-by-cell:
#
#   v2_guidance4        → baseline: smile on man, smile leaks onto woman
#   v3_yaml_trick       → preservation via YAML trick (cheap, upstream)
#   v3_preserve_woman   → preservation via explicit two-term loss (2× cost)
#
# All three use the same man enhance entries, the same 1000 iterations,
# the same lr/optimizer/precision. The only variable is HOW the
# "preserve woman" signal is injected into the training loop.

SLIDER_NAME=smile_man_strong_v3_yaml_trick_alpha1.0_rank4_noxattn
SLIDER_DIR=sdxl/trained_sliders/training/outputs/${SLIDER_NAME}
# Prefer the final checkpoint; fall back to the per_steps=500 checkpoint
# if the training job hit its walltime before completing.
if [ -f "${SLIDER_DIR}/${SLIDER_NAME}_last.safetensors" ]; then
    SLIDER="${SLIDER_DIR}/${SLIDER_NAME}_last.safetensors"
    echo "Using final checkpoint: ${SLIDER}"
else
    SLIDER="${SLIDER_DIR}/${SLIDER_NAME}_500steps.safetensors"
    echo "Final checkpoint missing — falling back to 500-steps: ${SLIDER}"
fi
SAVE_ROOT=sdxl/trained_sliders/training/outputs/generate_${SLIDER_NAME}_on_couple

PROMPT="a photo of a man and a woman, shoulders up, looking at the camera, 4k"
SEED=42

python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
    --prompt "${PROMPT}" \
    --slider "${SLIDER}" \
    --save_path "${SAVE_ROOT}" \
    --scales 0 1 2 3 4 \
    --seed "${SEED}" \
    --rank 4 \
    --alpha 1.0 \
    --steps 50
