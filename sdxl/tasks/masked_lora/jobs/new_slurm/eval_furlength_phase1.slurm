#!/bin/bash
#SBATCH --job-name=eval_furlength_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_furlength_phase1.slurm
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione furlength.
#
# Slider usato in fase 3: furlength_pet_sdxl_v1.
# Semi: stessi numeri 1001–1020 dei concept person per uniformità.
# Prompt: cani e gatti, masker = animale target in fase 2.
#
# Output: sdxl/tasks/masked_lora/runs/eval_furlength_{01..20}/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
OUTPUT_ROOT="sdxl/tasks/masked_lora/runs"
NEG="painting, illustration, cartoon, blurry, low quality, watermark, text, signature, human, person"

mkdir -p "$OUTPUT_ROOT" sdxl/tasks/masked_lora/logs

echo "================================================================"
echo "=== FASE 1: eval_furlength  (20 immagini base)              ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    1001 1002 1003 1004 1005
    1006 1007 1008 1009 1010
    1011 1012 1013 1014 1015
    1016 1017 1018 1019 1020
)

declare -a PROMPTS=(
    "a golden retriever and a labrador, photorealistic, 4k"
    "a golden retriever and a cat, photorealistic, 4k"
    "a husky and a border collie, photorealistic, 4k"
    "a poodle and a golden retriever, photorealistic, 4k"
    "a labrador and a cat, photorealistic, 4k"
    "two dogs in a field, photorealistic, 4k"
    "a golden retriever and a persian cat, photorealistic, 4k"
    "a husky and a siamese cat, photorealistic, 4k"
    "a border collie and a maine coon cat, photorealistic, 4k"
    "a poodle and a domestic cat, photorealistic, 4k"
    "a golden retriever and a labrador sitting together, photorealistic, 4k"
    "a husky and a cat on a couch, photorealistic, 4k"
    "two dogs in a park, photorealistic, 4k"
    "a labrador and a border collie, photorealistic, 4k"
    "a golden retriever and a cat on grass, photorealistic, 4k"
    "a poodle and a persian cat, photorealistic, 4k"
    "a golden retriever and a husky, photorealistic, 4k"
    "a labrador and a maine coon cat, photorealistic, 4k"
    "a border collie and a siamese cat, photorealistic, 4k"
    "two dogs and a cat, photorealistic, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_furlength_${RUN_NUM}"
    SEED="${SEEDS[$i]}"
    PROMPT="${PROMPTS[$i]}"

    echo "--- ${RUN_ID}  seed=${SEED} ---"
    echo "    ${PROMPT}"

    python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
        --prompt        "$PROMPT" \
        --negative_prompt "$NEG" \
        --seed          "$SEED" \
        --steps         50 \
        --guidance_scale 7.5 \
        --height        1024 \
        --width         1024 \
        --run_id        "$RUN_ID" \
        --output_root   "$OUTPUT_ROOT" \
        --model_id      "$MODEL_ID" \
        --device        cuda \
        --dtype         float16
    echo ""
done

echo "================================================================"
echo "=== FASE 1 COMPLETATA: eval_furlength                       ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_furlength_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_furlength_* | wc -l)"
