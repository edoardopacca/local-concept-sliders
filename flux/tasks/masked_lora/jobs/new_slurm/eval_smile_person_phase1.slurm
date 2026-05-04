#!/bin/bash
#SBATCH --job-name=flux_eval_smile_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_smile_person_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione smile_person.
# Stessi prompt e semi di age_person (slider gender-specific per donne).
# Fase 2 (SAM) in locale: cliccare sul viso di una qualsiasi donna.
#
# Output: flux/tasks/masked_lora/runs/eval_smile_person_{01..20}/
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
echo "=== FASE 1: eval_smile_person Flux  (20 immagini base)      ==="
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
    "two women, photorealistic portrait, 4k"
    "a woman and a man, photorealistic portrait, 4k"
    "two women and a man, photorealistic portrait, 4k"
    "a woman and two men, photorealistic portrait, 4k"
    "three women, photorealistic portrait, 4k"
    "two white women, photorealistic portrait, 4k"
    "a black woman and a white man, photorealistic portrait, 4k"
    "two asian women, photorealistic portrait, 4k"
    "a hispanic woman and a man, photorealistic portrait, 4k"
    "an indian woman and a man, photorealistic portrait, 4k"
    "a mother and daughter, photorealistic portrait, 4k"
    "two women sitting together, photorealistic portrait, 4k"
    "a woman and an elderly man, photorealistic portrait, 4k"
    "two women outdoor, photorealistic portrait, 4k"
    "a woman and a man, studio lighting, photorealistic portrait, 4k"
    "two women, neutral background, photorealistic portrait, 4k"
    "a young woman and an older woman, photorealistic portrait, 4k"
    "two women and two men, photorealistic portrait, 4k"
    "a woman and a man, close-up, photorealistic portrait, 4k"
    "three people including two women, photorealistic portrait, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_smile_person_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: eval_smile_person Flux               ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_smile_person_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_smile_person_* | wc -l)"
