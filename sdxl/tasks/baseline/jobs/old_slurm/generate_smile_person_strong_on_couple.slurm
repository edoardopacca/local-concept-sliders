#!/bin/bash
#SBATCH --job-name=smile_person_strong_couple
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

# Counterpart to generate_smile_man_strong_on_couple.slurm.
# Same prompt, same seed, same scales, same start_noise — only the slider
# weights differ (smile_person_strong vs smile_man_strong). Any difference
# in the resulting images is therefore attributable to the anchor token
# used at training time. This is the localization test's control arm.
SLIDER=sdxl/trained_sliders/training/outputs/smile_person_strong_alpha1.0_rank4_noxattn/smile_person_strong_alpha1.0_rank4_noxattn_last.safetensors
SAVE_PATH=sdxl/trained_sliders/training/outputs/smile_person_strong_alpha1.0_rank4_noxattn/generations_couple

PROMPT="a photo of a man and a woman, shoulders up, looking at the camera, 4k"
SEED=42

python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
    --prompt "${PROMPT}" \
    --slider "${SLIDER}" \
    --save_path "${SAVE_PATH}" \
    --scales 0 1 2 3 4 \
    --start_noise 700 \
    --seed "${SEED}" \
    --rank 4 \
    --alpha 1.0 \
    --steps 50
