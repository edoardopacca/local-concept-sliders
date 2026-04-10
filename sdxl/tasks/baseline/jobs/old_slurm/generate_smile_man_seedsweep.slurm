#!/bin/bash
#SBATCH --job-name=smile_man_seedsweep
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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

# Cross-seed robustness check: same prompt, same scale sweep, different seeds.
# If the slider effect is real, at scale=0 vs scale=2 you should see consistent
# directional change across all seeds (smile -> more smile). If the effect is
# seed-dependent or random, only some seeds will show it: that tells us the
# LoRA has a noisy direction that fires only on certain layouts.
SLIDER=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/smile_man_alpha1.0_rank4_noxattn_last.safetensors
PROMPT="a portrait photo of a man, studio lighting, 50mm"

for SEED in 7 42 77 123 2024; do
    SAVE_PATH=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/generations_single_seed${SEED}
    python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
        --prompt "${PROMPT}" \
        --slider "${SLIDER}" \
        --save_path "${SAVE_PATH}" \
        --scales 0 1 2 \
        --start_noise 700 \
        --seed "${SEED}" \
        --rank 4 \
        --alpha 1.0 \
        --steps 50
done
