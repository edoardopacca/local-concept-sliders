#!/bin/bash
#SBATCH --job-name=sel_fur_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# eval_furlength_phase1.slurm  (selectivity task)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di selettività "furlength".
#
# Ogni immagine contiene UN CANE e UN GATTO (almeno):
#   - TARGET:     il cane  (furlength_dog slider deve allungare il suo pelo)
#   - NON-TARGET: il gatto (furlength_dog slider non deve toccarlo)
#
# Confronto: furlength_dog (specifico) vs furlength_pet (generale)
#
# Seeds: 6041–6060
# Output: sdxl/tasks/selectivity/runs/eval_furlength_{01..20}/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
OUTPUT_ROOT="sdxl/tasks/selectivity/runs"
NEG="painting, illustration, cartoon, blurry, low quality, watermark, text, signature"

mkdir -p "$OUTPUT_ROOT" sdxl/tasks/selectivity/logs

echo "================================================================"
echo "=== FASE 1: selectivity/furlength  (20 immagini base)       ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    6041 6042 6043 6044 6045
    6046 6047 6048 6049 6050
    6051 6052 6053 6054 6055
    6056 6057 6058 6059 6060
)

declare -a PROMPTS=(
    "a dog and a cat together, photorealistic photo, 4k"
    "a golden retriever and a cat, photorealistic photo, 4k"
    "a labrador and a tabby cat, photorealistic photo, 4k"
    "a small dog and a large cat, photorealistic photo, 4k"
    "a puppy and a kitten, photorealistic photo, 4k"
    "a dog and a cat sitting on a sofa, photorealistic photo, 4k"
    "a dog and a cat in a garden, photorealistic photo, 4k"
    "a fluffy dog and a cat, photorealistic photo, 4k"
    "a border collie and a cat, photorealistic photo, 4k"
    "a beagle and a cat, photorealistic photo, 4k"
    "a dog and a cat on a white background, photorealistic photo, 4k"
    "a dog and a cat playing together, photorealistic photo, 4k"
    "a german shepherd and a cat, photorealistic photo, 4k"
    "a poodle and a cat, photorealistic photo, 4k"
    "a dog and a cat outdoors, photorealistic photo, 4k"
    "a husky and a cat, photorealistic photo, 4k"
    "a dog and a cat resting together, photorealistic photo, 4k"
    "a small white dog and an orange cat, photorealistic photo, 4k"
    "a dog and a black cat, photorealistic photo, 4k"
    "a dog and a cat side by side, photorealistic photo, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_furlength_${RUN_NUM}"
    SEED="${SEEDS[$i]}"
    PROMPT="${PROMPTS[$i]}"

    echo "--- ${RUN_ID}  seed=${SEED} ---"
    echo "    ${PROMPT}"

    python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
        --prompt          "$PROMPT" \
        --negative_prompt "$NEG" \
        --seed            "$SEED" \
        --steps           50 \
        --guidance_scale  7.5 \
        --height          1024 \
        --width           1024 \
        --run_id          "$RUN_ID" \
        --output_root     "$OUTPUT_ROOT" \
        --model_id        "$MODEL_ID" \
        --device          cuda \
        --dtype           float16
    echo ""
done

echo "================================================================"
echo "=== FASE 1 COMPLETATA: selectivity/furlength                ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_furlength_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_furlength_* | wc -l)"
