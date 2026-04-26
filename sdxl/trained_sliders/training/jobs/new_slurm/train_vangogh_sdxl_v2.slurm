#!/bin/bash
#SBATCH --job-name=vangogh_sdxl_v2
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=sdxl/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=sdxl/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_vangogh_sdxl_v2.slurm
# =============================================================================
# Style slider Van Gogh globale — v2.
#
# DIFF rispetto a v1 (train_vangogh_sky_sdxl_v1.slurm):
#   - guidance_scale 4 nei prompt (era 1)
#   - target/neutral neutri (non paesaggi pittorici)
#   - stile globale Van Gogh, non limitato al cielo
#   - test su prompt neutri: "a landscape", "a portrait of a person",
#     "a wheat field", "a village street"
#   - scale: -2 0 2 (meno ridondante di -2 -1 0 1 2)
#
# Path slider output:
#   sdxl/trained_sliders/sliders/vangogh_sdxl_v2_alpha1.0_rank16_noxattn/
#     vangogh_sdxl_v2_alpha1.0_rank16_noxattn_last.safetensors
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "================================================================"
echo "=== FASE 1: TRAINING vangogh_sdxl_v2                         ==="
echo "================================================================"
nvidia-smi
echo ""

python sdxl/trained_sliders/training/scripts/train.py \
    --config_file sdxl/trained_sliders/training/configs/vangogh_sdxl_v2.yaml \
    --name vangogh_sdxl_v2 \
    --rank 16 \
    --alpha 1.0 \
    --device 0

echo ""
echo "=== Training completato ==="
SLIDER_DIR="sdxl/trained_sliders/sliders/vangogh_sdxl_v2_alpha1.0_rank16_noxattn"
SLIDER="${SLIDER_DIR}/vangogh_sdxl_v2_alpha1.0_rank16_noxattn_last.safetensors"
ls -lh "$SLIDER_DIR"

echo ""
echo "================================================================"
echo "=== FASE 2: TEST POST-TRAINING                                ==="
echo "================================================================"
# Prompt neutrali (no descrittori pittorici nel prompt di test).
# scale -2 0 2, start_noise 700, seed 42.

TEST_ROOT="sdxl/tasks/baseline/outputs/vangogh_sdxl_v2_test"

run_test() {
    local PROMPT="$1"
    local SUBDIR="$2"
    local OUT="${TEST_ROOT}/${SUBDIR}"
    mkdir -p "$OUT"
    echo ""
    echo "--- Test: '${PROMPT}' → ${OUT} ---"
    python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
        --prompt "$PROMPT" \
        --slider "$SLIDER" \
        --save_path "$OUT" \
        --scales -2 0 2 \
        --start_noise 700 \
        --rank 16 \
        --alpha 1.0 \
        --seed 42 \
        --steps 50
}

run_test "a landscape"              "landscape"
run_test "a portrait of a person"   "portrait"
run_test "a wheat field"            "wheat_field"
run_test "a village street"         "village_street"

echo ""
echo "================================================================"
echo "=== JOB COMPLETATO                                            ==="
echo "=== Output slider: ${SLIDER_DIR}                              ==="
echo "=== Output test:   ${TEST_ROOT}                               ==="
echo "================================================================"
ls -lh "$TEST_ROOT"
