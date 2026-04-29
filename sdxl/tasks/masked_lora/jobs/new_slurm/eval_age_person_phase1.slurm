#!/bin/bash
#SBATCH --job-name=eval_age_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_age_person_phase1.slurm
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione age_person.
#
# Slider usato in fase 3: age_woman_sdxl_v1 (gender-specific, applicato solo
# su immagini con almeno una donna).  Tutti i prompt contengono almeno una donna.
#
# Nota: curlyhair e smile_person usano gli STESSI prompt e semi → le immagini
# base sono pixel-identiche; i run dir sono separati per chiarezza.
#
# Prerequisiti:
#   - SDXL base model in cache HF
#   - mkdir -p sdxl/tasks/masked_lora/runs  sdxl/tasks/masked_lora/logs
#
# Output: sdxl/tasks/masked_lora/runs/eval_age_person_{01..20}/
#   base.png  init_latents.pt  metadata.json
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
OUTPUT_ROOT="sdxl/tasks/masked_lora/runs"
NEG="painting, illustration, cartoon, blurry, low quality, watermark, text, signature"

mkdir -p "$OUTPUT_ROOT" sdxl/tasks/masked_lora/logs

echo "================================================================"
echo "=== FASE 1: eval_age_person  (20 immagini base)             ==="
echo "================================================================"
nvidia-smi
echo ""

# 20 (seed, prompt) — donne in tutti i prompt per lo slider age_woman
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
    RUN_ID="eval_age_person_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: eval_age_person                      ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_age_person_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_age_person_* | wc -l)"
