#!/bin/bash
#SBATCH --job-name=two_subjects_vangogh
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# Same setup as the smile sweep: a man + woman couple at 512x512. The
# Van Gogh style slider is applied to BOTH subjects with the same scale,
# swept from 0 to 5.
#
# 6 images:
#   01_scale_0_0.png  -> man=0  woman=0  (baseline, slider OFF)
#   02_scale_1_0.png  -> man=1  woman=0
#   03_scale_2_0.png  -> man=2  woman=0
#   04_scale_3_0.png  -> man=3  woman=0
#   05_scale_4_0.png  -> man=4  woman=0
#   06_scale_5_0.png  -> man=5  woman=0
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
# Parameters
# -----------------------------------------------------------------------------
SLIDER="flux/trained_sliders/sliders/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1/slider_0.pt"

PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"

TARGETS=(man woman)

SEED=42
RUN_NAME="two_subjects_vangogh_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 2 subjects vangogh sweep ==="
echo "=============================================="
echo "prompt    : $PROMPT"
echo "targets   : ${TARGETS[*]}"
echo "slider    : Van Gogh"
echo "seed      : $SEED"
echo "out_dir   : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local S1="$2" S2="$3"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : scales = ${S1} / ${S2}"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "$SLIDER" "$SLIDER" \
        --target_prompt "${TARGETS[0]}" "${TARGETS[1]}" \
        --lora_scales "$S1" "$S2" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

run_one "01_scale_0_0"   0   0
run_one "02_scale_1_0"   1   0
run_one "03_scale_2_0"   2   0
run_one "04_scale_3_0"   3   0
run_one "05_scale_4_0"   4   0
run_one "06_scale_5_0"   5   0

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
