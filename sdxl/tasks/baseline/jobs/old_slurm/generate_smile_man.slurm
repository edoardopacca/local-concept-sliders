#!/bin/bash
#SBATCH --job-name=smile_man_gen
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

# Slider weights path: produced by sdxl/trained_sliders/training/jobs/train_smile_man.slurm
# The training loop appends _alpha<alpha>_rank<rank>_<training_method> to the save name,
# so with alpha=1.0, rank=4, training_method=noxattn the file is:
SLIDER=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/smile_man_alpha1.0_rank4_noxattn_last.safetensors

# Output folder for this run: comparison grid + per-scale PNGs.
SAVE_PATH=sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/generations_man

# Default test: single male subject. Expect slider to modulate the smile.
PROMPT="a portrait photo of a man, studio lighting, 50mm"

# Seed fixed for reproducibility. Change to explore different layouts.
SEED=42

# Bidirectional scale sweep: negatives probe the "frown" direction,
# positives probe "smile". 0 is the exact baseline (no LoRA effect).
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

# --- Localization test ideas (run after the first one works) ---
# To verify the slider is spatially anchored to "man" and NOT to generic subjects,
# rerun this job with different --prompt / --save_path combos, e.g.:
#
#   PROMPT="a portrait photo of a woman, studio lighting, 50mm"
#   SAVE_PATH=.../generations_woman_control
#       => if slider is well-localized to "man", the woman should NOT smile more.
#
#   PROMPT="a man and a woman sitting at a cafe"
#   SAVE_PATH=.../generations_mixed_scene
#       => the key test — only the man's expression should change with scale.
