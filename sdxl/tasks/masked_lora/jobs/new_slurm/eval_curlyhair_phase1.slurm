#!/bin/bash
#SBATCH --job-name=eval_curly_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_curlyhair_phase1.slurm
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione curlyhair.
#
# Slider usato in fase 3: curlyhair_man_sdxl_v1 (gender-specific per uomini).
# Prompt con almeno un uomo in ogni immagine (speculare ai prompt di age/smile
# che usano donne). Semi identici 1001–1020 per uniformità.
#
# Output: sdxl/tasks/masked_lora/runs/eval_curlyhair_{01..20}/
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
echo "=== FASE 1: eval_curlyhair  (20 immagini base)              ==="
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
    "two men, photorealistic portrait, 4k"
    "a man and a woman, photorealistic portrait, 4k"
    "two men and a woman, photorealistic portrait, 4k"
    "a man and two women, photorealistic portrait, 4k"
    "three men, photorealistic portrait, 4k"
    "two white men, photorealistic portrait, 4k"
    "a black man and a white woman, upper body, visible hair, photorealistic portrait, 4k"
    "two asian men, photorealistic portrait, 4k"
    "a hispanic man and a woman, photorealistic portrait, 4k"
    "an indian man and a woman, photorealistic portrait, 4k"
    "a father and son, photorealistic portrait, 4k"
    "two men sitting together, photorealistic portrait, 4k"
    "a man and an elderly woman, photorealistic portrait, 4k"
    "two men outdoor, photorealistic portrait, 4k"
    "a man and a woman, studio lighting, photorealistic portrait, 4k"
    "two men, neutral background, photorealistic portrait, 4k"
    "a young man and an older man, photorealistic portrait, 4k"
    "two men and two women, photorealistic portrait, 4k"
    "a man and a woman, upper body, photorealistic portrait, 4k"
    "three people including two men, photorealistic portrait, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_curlyhair_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: eval_curlyhair                       ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_curlyhair_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_curlyhair_* | wc -l)"
