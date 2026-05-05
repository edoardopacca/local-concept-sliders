#!/bin/bash
#SBATCH --job-name=sel_smile_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# eval_smile_phase1.slurm  (selectivity task)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di selettività "smile".
#
# Ogni immagine contiene UNA DONNA e UN UOMO (almeno):
#   - TARGET:     la donna  (smile_woman slider deve farla sorridere)
#   - NON-TARGET: l'uomo    (smile_woman slider non deve toccarlo)
#
# Confronto: smile_woman (specifico) vs smile_person (generale)
#
# Seeds: 6061–6080  (semi diversi da age per immagini diverse)
# Output: sdxl/tasks/selectivity/runs/eval_smile_{01..20}/
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
echo "=== FASE 1: selectivity/smile  (20 immagini base)           ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    6061 6062 6063 6064 6065
    6066 6067 6068 6069 6070
    6071 6072 6073 6074 6075
    6076 6077 6078 6079 6080
)

declare -a PROMPTS=(
    "a woman and a man, photorealistic portrait, 4k"
    "a woman and two men, photorealistic portrait, 4k"
    "a young woman and an older man, photorealistic portrait, 4k"
    "two women and a man, photorealistic portrait, 4k"
    "a black woman and a white man, photorealistic portrait, 4k"
    "a white woman and a black man, photorealistic portrait, 4k"
    "an asian woman and a man, photorealistic portrait, 4k"
    "a woman and a man, studio lighting, photorealistic portrait, 4k"
    "a woman and a man, outdoor, photorealistic portrait, 4k"
    "a woman and a man, close-up, photorealistic portrait, 4k"
    "two women and two men, photorealistic portrait, 4k"
    "a young woman and an elderly man, photorealistic portrait, 4k"
    "a woman and a man, neutral background, photorealistic portrait, 4k"
    "a hispanic woman and a man, photorealistic portrait, 4k"
    "an indian woman and a man, photorealistic portrait, 4k"
    "a woman and a man, professional photo, 4k"
    "a blonde woman and a man, photorealistic portrait, 4k"
    "a woman with glasses and a man, photorealistic portrait, 4k"
    "a woman and a man, warm lighting, photorealistic portrait, 4k"
    "a woman and a man facing each other, photorealistic portrait, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_smile_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: selectivity/smile                    ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_smile_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_smile_* | wc -l)"
