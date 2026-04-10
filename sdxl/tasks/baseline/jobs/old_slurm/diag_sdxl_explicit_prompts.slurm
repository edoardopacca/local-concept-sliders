#!/bin/bash
#SBATCH --job-name=diag_sdxl_explicit_prompts
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=sdxl/tasks/baseline/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/baseline/logs/%x_%j.err

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# DIAGNOSTICO: può SDXL da solo separare "uomo sorride" da "donna neutra"
# solo tramite prompt, senza bisogno di slider?
#
# Se la risposta è SÌ → il CLIP leak che abbiamo osservato nel setup
#   v2_guidance4 dipende dal prompt specifico "a man and a woman, smiling"
#   in cui "smiling" bindinga globalmente. Cambiando sintassi si evita.
#   → il concept slider potrebbe non servire, o serve per scopi diversi.
#
# Se la risposta è NO → CLIP non fa attribute binding neanche con prompt
#   esplicito e ben strutturato. Il leak è una proprietà strutturale di
#   CLIP+SDXL, non un artefatto del nostro setup. Motiva v4 come unica
#   via pulita per ottenere separazione.
#
# Scale=0 → LoRA disattivata, vedi SDXL puro. Il file slider passato è
# ininfluente ma richiesto dall'API dello script.
#
# 4 varianti di prompt × 2 seed = 8 immagini, ~5-10 min di runtime totale.

SLIDER="sdxl/trained_sliders/training/outputs/smile_man_strong_v3_yaml_trick_alpha1.0_rank4_noxattn/smile_man_strong_v3_yaml_trick_alpha1.0_rank4_noxattn_last.safetensors"
SAVE_ROOT=sdxl/trained_sliders/training/outputs/diag_sdxl_explicit_prompts

declare -a PROMPTS=(
    "Two people, shoulders up, facing camera. Smiling man. Neutral expression woman."
    "a man smiling next to a woman with neutral expression, photograph"
    "a smiling man and a serious woman, portrait, shoulders up"
    "two people portrait: a happy man on the left, a calm woman on the right"
)

declare -a TAGS=(
    "v1_user_original"
    "v2_next_to_with"
    "v3_smiling_serious"
    "v4_spatial_left_right"
)

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    TAG="${TAGS[$i]}"
    for SEED in 42 7; do
        echo ""
        echo "=== [${TAG} / seed=${SEED}] ==="
        echo "    prompt: ${PROMPT}"
        python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
            --prompt "${PROMPT}" \
            --slider "${SLIDER}" \
            --save_path "${SAVE_ROOT}/${TAG}_seed${SEED}" \
            --scales 0 \
            --seed "${SEED}" \
            --rank 4 --alpha 1.0 --steps 50
    done
done

echo ""
echo "=== FINITO ==="
echo "Output in: ${SAVE_ROOT}"
ls -R "${SAVE_ROOT}"
