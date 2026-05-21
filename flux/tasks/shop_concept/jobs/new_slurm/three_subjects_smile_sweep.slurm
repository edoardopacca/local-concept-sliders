#!/bin/bash
#SBATCH --job-name=three_subjects_smile
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# three_subjects_smile_sweep.slurm
# =============================================================================
# Generate 3 images with the mask-aware pipeline (Flux + per-subject
# multi-LoRA) using the same smile slider with different scales for each
# of the 3 subjects.
#
# Prompt: scene with 3 subjects facing the camera (man / woman / child).
# Slider: smile_flux_v5_symmetric (rank=16, xattn, alpha=1).
#
# 3 images:
#   1. baseline_zero      -> scales: man=0   woman=0   child=0   (slider OFF everywhere)
#   2. minus1_zero_plus1  -> scales: man=-1  woman=0   child=+1  (one serious, one neutral, one smiling)
#   3. minus2_zero_minus2 -> scales: man=-2  woman=0   child=-2  (first and third heavily serious)
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p flux/tasks/shop_concept/logs

# -----------------------------------------------------------------------------
# Shared parameters
# -----------------------------------------------------------------------------
# Path to the final slider checkpoint (training completed).
# To test intermediate checkpoints replace with one of:
#   .../flux-smile_flux_v5_symmetric_step100/slider_0.pt   (early)
#   .../flux-smile_flux_v5_symmetric_step200/slider_0.pt
#   .../flux-smile_flux_v5_symmetric_step300/slider_0.pt
#   .../flux-smile_flux_v5_symmetric_step400/slider_0.pt   (late)
SLIDER="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"

# Main prompt (the 3 targets must be literal substrings)
PROMPT="a photograph of a man, a woman, and a child standing side by side, looking at the camera, in a sunny park with trees and grass in the background, natural light, sharp focus, high detail"

# Prompt substrings identifying the 3 subjects.
TARGETS=(man woman child)

SEED=42
RUN_NAME="three_subjects_smile_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 3 subjects smile sweep ==="
echo "=============================================="
echo "prompt  : $PROMPT"
echo "targets : ${TARGETS[*]}"
echo "seed    : $SEED"
echo "out_dir : $OUT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Helper function: launches one generation with 3 scale (one per subject)
# -----------------------------------------------------------------------------
run_one() {
    local NAME="$1"
    local S1="$2" S2="$3" S3="$4"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : scales = ${S1} / ${S2} / ${S3}"
    echo "  out: $OUT_PATH"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "$SLIDER" "$SLIDER" "$SLIDER" \
        --target_prompt "${TARGETS[0]}" "${TARGETS[1]}" "${TARGETS[2]}" \
        --lora_scales "$S1" "$S2" "$S3" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

# -----------------------------------------------------------------------------
# The 3 generations
# -----------------------------------------------------------------------------
run_one "01_baseline_zero"        0   0   0
run_one "02_minus1_zero_plus1"   -1   0   1
run_one "03_minus2_zero_minus2"  -2   0  -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
