#!/bin/bash
#SBATCH --job-name=smile_man_lowscales
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
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

# Fine-grained POSITIVE scale sweep on the ORIGINAL smile_man slider, to check
# whether there is a visible modulation window at small magnitudes that we
# missed with the coarse `-2 -1 0 1 2` grid. If the LoRA is off-manifold at
# large scales (the suspected failure mode with guidance_scale=4 at training
# time), a clean low-scale response would still be visible in this range.
SLIDER=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/smile_man_alpha1.0_rank4_noxattn_last.safetensors
SAVE_PATH=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/generations_single_lowscales

PROMPT="a portrait photo of a man, studio lighting, 50mm"
SEED=42

python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
    --prompt "${PROMPT}" \
    --slider "${SLIDER}" \
    --save_path "${SAVE_PATH}" \
    --scales 0 0.25 0.5 0.75 1.0 \
    --start_noise 700 \
    --seed "${SEED}" \
    --rank 4 \
    --alpha 1.0 \
    --steps 50
