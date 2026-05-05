#!/bin/bash
#SBATCH --job-name=sel_age_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# eval_age_phase1.slurm  (selectivity task)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di selettività "age".
#
# Ogni immagine contiene UNA DONNA e UN UOMO (almeno):
#   - TARGET:     la donna  (age_woman slider deve invecchiarla)
#   - NON-TARGET: l'uomo    (age_woman slider non deve toccarlo)
#
# Confronto: age_woman (specifico) vs age_person (generale)
#
# Seeds: 6001–6020  (separati da masked_lora per evitare sovrapposizioni)
# Output: sdxl/tasks/selectivity/runs/eval_age_{01..20}/
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
echo "=== FASE 1: selectivity/age  (20 immagini base)             ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    6001 6002 6003 6004 6005
    6006 6007 6008 6009 6010
    6011 6012 6013 6014 6015
    6016 6017 6018 6019 6020
)

declare -a PROMPTS=(
    "a woman and a man, photorealistic portrait, 4k"
    "two women and a man, photorealistic portrait, 4k"
    "a young woman and an older man, photorealistic portrait, 4k"
    "a woman and two men, photorealistic portrait, 4k"
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
    "a woman and a man looking at each other, photorealistic portrait, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_age_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETATA: selectivity/age                      ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_age_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_age_* | wc -l)"
