#!/bin/bash
#SBATCH --job-name=multi_masked_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# run_multi_masked_phase3.slurm
# =============================================================================
# FASE 3 — Applica masked LoRA per tutti e 6 gli esperimenti.
#
# Prerequisiti per ogni run dir:
#   base.png + init_latents.pt  (da fase 1)
#   mask.png                    (da fase 2 locale con SAM)
#
# Slider utilizzati:
#   age_person    → age_person_sdxl_v1_alpha1.0_rank4_noxattn
#   curlyhair     → curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn
#   daynight      → daynight_sdxl_v1_alpha1.0_rank4_noxattn
#   furlength     → furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn
#   smile_person  → smile_person_sdxl_v1_alpha1.0_rank4_noxattn
#   painterly     → painterly_sdxl_v1_alpha1.0_rank4_noxattn
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
SLIDERS_ROOT="sdxl/trained_sliders/sliders"

echo "================================================================"
echo "=== FASE 3: MULTI MASKED EDIT (6 experiments)                ==="
echo "================================================================"
nvidia-smi
echo ""

run_edit() {
    local RUN_ID="$1"
    local SLIDER_PATH="$2"
    local OUTPUT_PREFIX="$3"
    local RANK="${4:-4}"

    local RUN_DIR="sdxl/tasks/masked_lora/runs/${RUN_ID}"
    echo "==== ${RUN_ID} ===="
    ls -lh "${RUN_DIR}/"

    for SCALE in 1.5 2.0 3.0; do
        echo "--- Scale ${SCALE} ---"
        python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
            --run_dir "$RUN_DIR" \
            --slider_path "$SLIDER_PATH" \
            --slider_scale "$SCALE" \
            --rank "$RANK" \
            --start_noise 750 \
            --model_id "$MODEL_ID" \
            --device cuda \
            --dtype float16 \
            --mask_name mask.png \
            --output_name "edited_${OUTPUT_PREFIX}_s${SCALE}.png"
    done
    echo ""
}

run_edit "masked_age_person_01" \
    "${SLIDERS_ROOT}/age_person_sdxl_v1_alpha1.0_rank4_noxattn/age_person_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors" \
    "age"

run_edit "masked_curlyhair_01" \
    "${SLIDERS_ROOT}/curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn/curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors" \
    "curly"

run_edit "masked_daynight_01" \
    "${SLIDERS_ROOT}/daynight_sdxl_v1_alpha1.0_rank4_noxattn/daynight_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors" \
    "daynight"

run_edit "masked_furlength_01" \
    "${SLIDERS_ROOT}/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn/furlength_pet_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors" \
    "furlength"

run_edit "masked_smile_person_01" \
    "${SLIDERS_ROOT}/smile_person_sdxl_v1_alpha1.0_rank4_noxattn/smile_person_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors" \
    "smile"

run_edit "masked_painterly_01" \
    "${SLIDERS_ROOT}/painterly_sdxl_v1_alpha1.0_rank4_noxattn/painterly_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors" \
    "painterly"

echo "================================================================"
echo "=== FASE 3 COMPLETATA                                         ==="
echo "================================================================"
for RUN in masked_age_person_01 masked_curlyhair_01 masked_daynight_01 \
           masked_furlength_01 masked_smile_person_01 masked_painterly_01; do
    echo "--- ${RUN} ---" && ls -lh "sdxl/tasks/masked_lora/runs/${RUN}/"
done
