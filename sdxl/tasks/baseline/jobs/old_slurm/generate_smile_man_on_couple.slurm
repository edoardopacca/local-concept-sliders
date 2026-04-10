#!/bin/bash
#SBATCH --job-name=smile_man_couple
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

# ============================================================================
# KEY LOCALIZATION TEST.
# Scene with BOTH a man and a woman, framed the same way, looking at camera.
# The slider was trained with target="man" — if anchoring works, only the man's
# expression should change across scales; the woman should stay unchanged.
# If the woman's expression tracks the scale too, the LoRA has learned a global
# "smile" direction and the subject anchoring failed. That would be the
# empirical motivation for Phase 2 (preservation set / Eq. 8).
# ============================================================================

SLIDER=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/smile_man_alpha1.0_rank4_noxattn_last.safetensors
SAVE_PATH=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/generations_couple

# Simple prompt: both subjects, same framing, neutral setting.
# Shoulders up + looking at camera keeps the two faces comparable side-by-side.
PROMPT="a photo of a man and a woman, shoulders up, looking at the camera, 4k"

# Fixed seed — if the composition doesn't place both subjects clearly,
# rerun with a different seed (e.g. 7, 13, 77, 123, 2024) until you find
# one where both faces are clearly visible. Same seed across scales means
# the layout is identical, so any difference IS the slider's effect.
SEED=42

python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
    --prompt "${PROMPT}" \
    --slider "${SLIDER}" \
    --save_path "${SAVE_PATH}" \
    --scales -2 -1 0 1 2 \
    --start_noise 700 \
    --seed "${SEED}" \
    --rank 4 \
    --alpha 1.0 \
    --steps 50

# --- How to read the grid ---
#   scale=0        : baseline, should show both faces neutral/natural
#   scale=+1, +2   : ONLY the man should smile more; woman unchanged
#   scale=-1, -2   : ONLY the man should frown / turn neutral; woman unchanged
#
# If the woman's expression mirrors the man's, anchoring did not localize.
# If the man's expression changes while the woman's is visually stable,
# the local-concept-slider hypothesis is empirically supported on this seed.
