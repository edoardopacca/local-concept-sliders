#!/bin/bash
#SBATCH --job-name=flux_eval_curly_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_curlyhair_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione curlyhair.
# Fase 2 (SAM) in locale: cliccare ESATTAMENTE sui capelli di un uomo.
#
# Output: flux/tasks/masked_lora/runs/eval_curlyhair_{01..20}/
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
echo "=== FASE 1: eval_curlyhair Flux  (20 immagini base)         ==="
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
echo "=== FASE 1 COMPLETATA: eval_curlyhair Flux                  ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_curlyhair_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_curlyhair_* | wc -l)"
