#!/bin/bash
#SBATCH --job-name=eval_paint_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_painterly_phase1.slurm
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione painterly.
#
# Slider usato in fase 3: painterly_sdxl_v1.
# I prompt descrivono soggetti concreti SENZA stile: la base è fotorealistica,
# lo slider di pittura viene applicato solo sul soggetto in fase 3.
# Ogni immagine ha un soggetto singolo in primo piano facilmente selezionabile
# con SAM in fase 2 (es. edificio, barca, mulino, albero...).
# Mask target in fase 2: il SOGGETTO PRINCIPALE in primo piano.
#
# Output: sdxl/tasks/masked_lora/runs/eval_painterly_{01..20}/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
OUTPUT_ROOT="sdxl/tasks/masked_lora/runs"
NEG="painting, illustration, cartoon, oil painting, watercolor, blurry, low quality, watermark, text, signature"

mkdir -p "$OUTPUT_ROOT" sdxl/tasks/masked_lora/logs

echo "================================================================"
echo "=== FASE 1: eval_painterly  (20 immagini base)              ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    4001 4002 4003 4004 4005
    4006 4007 4008 4009 4010
    4011 4012 4013 4014 4015
    4016 4017 4018 4019 4020
)

# Prompt: soggetto reale senza indicazione di stile — la base deve essere
# fotorealistica; lo slider applica la texture pittorica solo sul soggetto.
declare -a PROMPTS=(
    "two buildings on a street, photorealistic, 4k"
    "a sailing boat on the sea, photorealistic, 4k"
    "a windmill in a field, photorealistic, 4k"
    "a lighthouse on a cliff, photorealistic, 4k"
    "a red barn in a meadow, photorealistic, 4k"
    "a stone bridge over a river, photorealistic, 4k"
    "a wooden cabin in the forest, photorealistic, 4k"
    "a church in a village square, photorealistic, 4k"
    "a tree in a field, photorealistic, 4k"
    "a bicycle leaning against a wall, photorealistic, 4k"
    "a park bench under a tree, photorealistic, 4k"
    "a row of flowers in a garden, photorealistic, 4k"
    "a small house with a garden, photorealistic, 4k"
    "a fountain in a square, photorealistic, 4k"
    "a mountain with a lake in front, photorealistic, 4k"
    "a boat on a canal, photorealistic, 4k"
    "a field of sunflowers, photorealistic, 4k"
    "a waterfall in a forest, photorealistic, 4k"
    "an arch bridge, photorealistic, 4k"
    "a clock tower in a city, photorealistic, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_painterly_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: eval_painterly                       ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_painterly_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_painterly_* | wc -l)"
