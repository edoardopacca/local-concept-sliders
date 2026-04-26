#!/bin/bash
#SBATCH --job-name=furlength_pet_v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=sdxl/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=sdxl/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_furlength_pet_sdxl_v1.slurm
# =============================================================================
# Fur length slider for dogs and cats (general, no subject localization).
#
# Setup:
#   - 10 entries: 5 dog types + 5 cat types
#   - positive=long fur/fluffy, unconditional=short fur/shorthaired
#   - rank 4, alpha 1.0, noxattn
#   - guidance_scale 4, resolution 1024, 1000 iterations
#   - start_noise 750
#
# Test: scale -2 -1 0 1 2, seed 42
#   - scale < 0: shorter fur / scale > 0: longer fur
#
# Output slider:
#   sdxl/trained_sliders/sliders/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn/
#     furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors
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
echo "=== FASE 1: TRAINING furlength_pet_sdxl_v1                   ==="
echo "================================================================"
nvidia-smi
echo ""

python sdxl/trained_sliders/training/scripts/train.py \
    --config_file sdxl/trained_sliders/training/configs/furlength_pet_sdxl_v1.yaml \
    --name furlength_pet_sdxl_v1 \
    --rank 4 \
    --alpha 1.0 \
    --device 0

echo ""
echo "=== Training completato ==="
SLIDER_DIR="sdxl/trained_sliders/sliders/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn"
SLIDER="${SLIDER_DIR}/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"
ls -lh "$SLIDER_DIR"

echo ""
echo "================================================================"
echo "=== FASE 2: TEST POST-TRAINING                                ==="
echo "================================================================"

TEST_ROOT="sdxl/tasks/baseline/outputs/furlength_pet_sdxl_v1_test"

run_test() {
    local PROMPT="$1"
    local SUBDIR="$2"
    local OUT="${TEST_ROOT}/${SUBDIR}"
    mkdir -p "$OUT"
    echo "--- Test: '${PROMPT}' -> ${OUT} ---"
    python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
        --prompt "$PROMPT" \
        --slider "$SLIDER" \
        --save_path "$OUT" \
        --scales -2 -1 0 1 2 \
        --start_noise 750 \
        --rank 4 \
        --alpha 1.0 \
        --seed 42 \
        --steps 50
}

run_test "a photorealistic portrait of a dog, sharp focus, studio lighting"     "dog"
run_test "a photorealistic portrait of a cat, sharp focus, studio lighting"     "cat"
run_test "a photorealistic portrait of a golden retriever, sharp focus"         "golden_retriever"
run_test "a photorealistic portrait of a dog and a cat, sharp focus"            "dog_and_cat"

echo ""
echo "================================================================"
echo "=== JOB COMPLETATO                                            ==="
echo "=== Output slider: ${SLIDER_DIR}                              ==="
echo "=== Output test:   ${TEST_ROOT}                               ==="
echo "================================================================"
ls -lh "$TEST_ROOT"
