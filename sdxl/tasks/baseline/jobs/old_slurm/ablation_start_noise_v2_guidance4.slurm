#!/bin/bash
#SBATCH --job-name=ablation_start_noise_v2_guidance4
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

# Ablation on start_noise, using v2_guidance4 (our strongest slider —
# gs=4 during training, produces teeth visible on the man at scale=4).
#
# v2_guidance4 has confirmed strong smile direction in feature space but
# leaks onto the woman with equal intensity — stronger slider, stronger
# leak. Localization cannot be fixed by changing --scale (it's the
# magnitude, not the locus). The remaining inference-time lever is
# --start_noise: by restricting the LoRA to fire only in the late phase
# of denoising (low t), we force it to act on an already-differentiated
# latent — where "man face" and "woman face" regions have distinct
# activations. In principle the ΔW, which is aligned with "smile on
# man-trained features", should have more grip on the man's region and
# less on the woman's.
#
# Code check (generate_with_sliders.py line 236):
#   if t > start_noise: LoRA off else on.
# → LoRA active at t in [0, start_noise]. Lower start_noise = slider
# only in late (fine-detail) denoising phase.
#
# Sweep: 5 values. Same seed/prompt/scale so differences across panels
# are purely attributable to start_noise. Scale=4 to be in the regime
# where the teeth effect is clearly visible on the man.
SLIDER=sdxl/trained_sliders/training/outputs/smile_man_strong_v2_guidance4_alpha1.0_rank4_noxattn/smile_man_strong_v2_guidance4_alpha1.0_rank4_noxattn_last.safetensors
SAVE_ROOT=sdxl/trained_sliders/training/outputs/ablation_start_noise_v2_guidance4

PROMPT="a photo of a man and a woman, shoulders up, looking at the camera, 4k"
SEED=42

for SN in 900 700 500 300 100; do
  echo "=== start_noise=${SN} ==="
  python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
      --prompt "${PROMPT}" \
      --slider "${SLIDER}" \
      --save_path "${SAVE_ROOT}/sn${SN}" \
      --scales 0 2 4 \
      --start_noise "${SN}" \
      --seed "${SEED}" \
      --rank 4 \
      --alpha 1.0 \
      --steps 50
done
