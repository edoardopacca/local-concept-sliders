#!/bin/bash
#SBATCH --job-name=flux_eval_fur_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_furlength_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione furlength.
# Fase 2 (SAM) in locale: cliccare sul corpo di UN animale (il primo/più
# visibile se ce ne sono più di uno).
#
# Output: flux/tasks/masked_lora/runs/eval_furlength_{01..20}/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_ROOT="flux/tasks/masked_lora/runs"
mkdir -p "$OUTPUT_ROOT" flux/tasks/masked_lora/logs

echo "================================================================"
echo "=== FASE 1: eval_furlength Flux  (20 immagini base)         ==="
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

    python flux/tasks/masked_lora/scripts/01_generate_base.py \
        --prompt        "$PROMPT" \
        --seed          "$SEED" \
        --steps         30 \
        --guidance_scale 3.5 \
        --max_sequence_length 256 \
        --height        1024 \
        --width         1024 \
        --run_id        "$RUN_ID" \
        --output_root   "$OUTPUT_ROOT" \
        --model_id      "black-forest-labs/FLUX.1-dev" \
        --device        cuda
    echo ""
done

echo "================================================================"
echo "=== FASE 1 COMPLETATA: eval_furlength Flux                  ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_furlength_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_furlength_* | wc -l)"
