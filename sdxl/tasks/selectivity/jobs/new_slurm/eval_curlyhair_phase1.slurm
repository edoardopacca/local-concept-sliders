#!/bin/bash
#SBATCH --job-name=sel_curly_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# eval_curlyhair_phase1.slurm  (selectivity task)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di selettività "curlyhair".
#
# Ogni immagine contiene UN UOMO e UNA DONNA (almeno):
#   - TARGET:     l'uomo    (curlyhair_man slider deve arricciargli i capelli)
#   - NON-TARGET: la donna  (curlyhair_man slider non deve toccarla)
#
# Confronto: curlyhair_man (specifico) vs curlyhair_person (generale)
#
# Seeds: 6021–6040
# Output: sdxl/tasks/selectivity/runs/eval_curlyhair_{01..20}/
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
echo "=== FASE 1: selectivity/curlyhair  (20 immagini base)       ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    6021 6022 6023 6024 6025
    6026 6027 6028 6029 6030
    6031 6032 6033 6034 6035
    6036 6037 6038 6039 6040
)

declare -a PROMPTS=(
    "a man and a woman, photorealistic portrait, 4k"
    "a man and two women, photorealistic portrait, 4k"
    "a young man and a woman, photorealistic portrait, 4k"
    "two men and a woman, photorealistic portrait, 4k"
    "a black man and a woman, photorealistic portrait, 4k"
    "a white man and a woman, photorealistic portrait, 4k"
    "an asian man and a woman, photorealistic portrait, 4k"
    "a man and a woman, studio lighting, photorealistic portrait, 4k"
    "a man and a woman, outdoor, photorealistic portrait, 4k"
    "a man and a woman, close-up, photorealistic portrait, 4k"
    "two men and two women, photorealistic portrait, 4k"
    "a young man and an older woman, photorealistic portrait, 4k"
    "a man and a woman, neutral background, photorealistic portrait, 4k"
    "a hispanic man and a woman, photorealistic portrait, 4k"
    "an indian man and a woman, photorealistic portrait, 4k"
    "a man and a woman, professional photo, 4k"
    "a man with a beard and a woman, photorealistic portrait, 4k"
    "a man and a woman with glasses, photorealistic portrait, 4k"
    "a man and a woman, warm lighting, photorealistic portrait, 4k"
    "a man and a woman looking at each other, photorealistic portrait, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_curlyhair_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: selectivity/curlyhair                ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_curlyhair_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_curlyhair_* | wc -l)"
