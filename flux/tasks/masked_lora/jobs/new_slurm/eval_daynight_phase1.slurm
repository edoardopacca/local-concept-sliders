#!/bin/bash
#SBATCH --job-name=flux_eval_day_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_daynight_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione daynight.
# Fase 2 (SAM) in locale: cliccare sulla regione del CIELO.
#
# Output: flux/tasks/masked_lora/runs/eval_daynight_{01..20}/
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
echo "=== FASE 1: eval_daynight Flux  (20 immagini base)          ==="
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
    "a forest clearing, photorealistic, 4k"
    "a valley with a river, photorealistic, 4k"
    "a mountain with snow peaks, photorealistic, 4k"
    "a tropical beach, photorealistic, 4k"
    "rolling hills, photorealistic, 4k"
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
echo "=== FASE 1 COMPLETED: eval_daynight Flux                   ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_daynight_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_daynight_* | wc -l)"
