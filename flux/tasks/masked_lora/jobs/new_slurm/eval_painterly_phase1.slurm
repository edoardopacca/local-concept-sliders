#!/bin/bash
#SBATCH --job-name=flux_eval_paint_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_painterly_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione painterly.
# Fase 2 (SAM) in locale: cliccare sulla regione del CIELO (tutte le immagini
# hanno cielo visibile per design).
#
# Output: flux/tasks/masked_lora/runs/eval_painterly_{01..20}/
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
echo "=== FASE 1: eval_painterly Flux  (20 immagini base)         ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    4001 4002 4003 4004 4005
    4006 4007 4008 4009 4010
    4011 4012 4013 4014 4015
    4016 4017 4018 4019 4020
)

declare -a PROMPTS=(
    "two buildings on a street under a wide open sky, photorealistic, 4k"
    "a sailing boat on the sea, photorealistic, 4k"
    "a windmill in a field, photorealistic, 4k"
    "a lighthouse on a cliff, photorealistic, 4k"
    "a red barn in a meadow, photorealistic, 4k"
    "a stone bridge over a river, photorealistic, 4k"
    "a wooden cabin at the edge of a forest under a wide open sky, photorealistic, 4k"
    "a church in a village square, photorealistic, 4k"
    "a tree in a field, photorealistic, 4k"
    "a bicycle on a country road under a wide open sky, photorealistic, 4k"
    "a park bench under a tree, photorealistic, 4k"
    "a row of flowers in an open field under a wide open sky, photorealistic, 4k"
    "a small house with a garden, photorealistic, 4k"
    "a fountain in an open square under a wide open sky, photorealistic, 4k"
    "a mountain with a lake in front, photorealistic, 4k"
    "a boat on a canal, photorealistic, 4k"
    "a field of sunflowers under a wide open sky, photorealistic, 4k"
    "a waterfall on a rocky hillside under a wide open sky, photorealistic, 4k"
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
echo "=== FASE 1 COMPLETATA: eval_painterly Flux                  ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_painterly_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_painterly_* | wc -l)"
