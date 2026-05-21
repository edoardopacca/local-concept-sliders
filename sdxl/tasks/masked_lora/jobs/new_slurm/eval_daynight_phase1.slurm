#!/bin/bash
#SBATCH --job-name=eval_daynight_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_daynight_phase1.slurm
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione daynight.
#
# Slider used in phase 3: daynight_sdxl_v1, scale 3.0 (optimal for the concept).
# Tutti i paesaggi hanno cielo visibile → facile da selezionare con SAM in fase 2.
# Mask target in fase 2: regione del CIELO.
#
# Output: sdxl/tasks/masked_lora/runs/eval_daynight_{01..20}/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
OUTPUT_ROOT="sdxl/tasks/masked_lora/runs"
NEG="painting, illustration, cartoon, blurry, low quality, watermark, text, signature, night, dark"

mkdir -p "$OUTPUT_ROOT" sdxl/tasks/masked_lora/logs

echo "================================================================"
echo "=== FASE 1: eval_daynight  (20 immagini base)               ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    3001 3002 3003 3004 3005
    3006 3007 3008 3009 3010
    3011 3012 3013 3014 3015
    3016 3017 3018 3019 3020
)

declare -a PROMPTS=(
    "a river valley, photorealistic, 4k"
    "a mountain landscape, photorealistic, 4k"
    "a beach with ocean, photorealistic, 4k"
    "a countryside with fields, photorealistic, 4k"
    "a lake surrounded by trees, photorealistic, 4k"
    "a desert landscape, photorealistic, 4k"
    "a grassy meadow, photorealistic, 4k"
    "a coastal cliff, photorealistic, 4k"
    "a forest clearing under a wide open sky, photorealistic, 4k"
    "a valley with a river, photorealistic, 4k"
    "a mountain with snow peaks, photorealistic, 4k"
    "a tropical beach under a wide open sky, photorealistic, 4k"
    "rolling hills under a wide open sky, photorealistic, 4k"
    "a rocky mountain, photorealistic, 4k"
    "a calm lake, photorealistic, 4k"
    "a vineyard landscape, photorealistic, 4k"
    "a canyon landscape, photorealistic, 4k"
    "a rice field landscape, photorealistic, 4k"
    "an open plain, photorealistic, 4k"
    "a savanna landscape, photorealistic, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_daynight_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETED: eval_daynight                        ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_daynight_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_daynight_* | wc -l)"
