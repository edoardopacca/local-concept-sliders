#!/bin/bash
#SBATCH --job-name=smile_man_strong_v2_guidance4_couple
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

# Four-way comparison on the same couple prompt / seed / scales / start_noise:
#   - reference       → sdxl/trained_sliders/sliders/smiling.pt                          (person, 10 entries, gs_eff=1)
#   - v1              → smile_man_strong_last.safetensors                 (man,    5 entries,  gs=1)
#   - v2              → smile_man_strong_v2_last.safetensors              (man,   10 entries,  gs=1)
#   - v2_guidance4    → smile_man_strong_v2_guidance4_last.safetensors    (man,   10 entries,  gs=4) ← this
#
# Since the first three comparisons share scales 0..4, we generate the gs=4
# variant at the same scale sweep for visual alignment. If the ΔW is truly
# ~4× larger, the slider will saturate faster along the sweep and the
# per-scale step-size of visual change will be ~4× what v2 shows.
SLIDER=sdxl/trained_sliders/training/outputs/smile_man_strong_v2_guidance4_alpha1.0_rank4_noxattn/smile_man_strong_v2_guidance4_alpha1.0_rank4_noxattn_last.safetensors
SAVE_PATH=sdxl/trained_sliders/training/outputs/smile_man_strong_v2_guidance4_alpha1.0_rank4_noxattn/generations_couple

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
