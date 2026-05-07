#!/bin/bash
#SBATCH --job-name=three_subjects_smile_HQ
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# three_subjects_smile_sweep_HQ.slurm — versione HIGH QUALITY del sweep
# =============================================================================
# Differenze vs three_subjects_smile_sweep.slurm:
#   - 1024x1024 (era 512x512) → soggetti hanno ~3x più pixel, qualità decente
#   - 50 inference steps (era 30) → più tempo ma più dettaglio
#   - Time wall: 1h30 (era 45min) per le 3 generazioni
#
# 3 immagini, scales:
#   1. baseline_zero      → 0 / 0 / 0
#   2. minus1_zero_plus1  → -1 / 0 / +1
#   3. minus2_zero_minus2 → -2 / 0 / -2
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
# Parametri
# -----------------------------------------------------------------------------
SLIDER="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"

PROMPT="a photograph of a man, a woman, and a child standing side by side, looking at the camera, in a sunny park with trees and grass in the background, natural light, sharp focus, high detail"

TARGETS=(man woman child)

SEED=42
RUN_NAME="three_subjects_smile_HQ_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

# HQ params
STEPS=50
HEIGHT=1024
WIDTH=1024
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 3 subjects smile HQ      ==="
echo "=============================================="
echo "prompt    : $PROMPT"
echo "targets   : ${TARGETS[*]}"
echo "seed      : $SEED"
echo "resolution: ${HEIGHT}x${WIDTH}, ${STEPS} steps"
echo "out_dir   : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local S1="$2" S2="$3" S3="$4"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : scales = ${S1} / ${S2} / ${S3}"
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

run_one "01_baseline_zero"        0   0   0
run_one "02_minus1_zero_plus1"   -1   0   1
run_one "03_minus2_zero_minus2"  -2   0  -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
