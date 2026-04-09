#!/bin/bash
#SBATCH --job-name=smile_man_strong_v3_preserve_woman
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=sdxl/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=sdxl/trained_sliders/training/logs/%x_%j.err

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# v3_preserve_woman — explicit two-term loss:
#   loss = loss_enhance_on_man + lambda_pres * loss_preservation_on_woman
#
# The man-enhance term is identical to v2_guidance4 (same prompts YAML,
# gs=4 per entry). The preservation term penalizes any ΔW applied to
# "woman" UNet activations at every timestep: at iteration step i, we
# sample one woman entry, roll the diffusion forward to a random t with
# LoRA on, then compare the noise prediction at that state with LoRA on
# vs LoRA off — the MSE between the two is what the optimizer fights.
#
# Hypothesis: the CLIP attribute leak on v2_guidance4 (smile bleeds onto
# the woman's face, equal intensity, symmetric wrt gender) is caused by
# the LoRA's ΔW being nonzero on woman-activated UNet features, because
# during v2_guidance4 training nothing told the LoRA "don't touch
# woman". Adding an explicit second loss term that measures exactly
# that quantity and asks it to be zero should — in principle — collapse
# the leak: the LoRA is now Pareto-optimizing "enhance smile on man
# features AND leave woman features alone".
#
# Cost: ~2× v2_guidance4 per iteration (extra denoise trajectory + 2
# extra forward passes on UNet). Same 1000 iterations → walltime budget
# 2h (v2_guidance4 was ~45 min so 2h is comfortable for 2× cost).
# Mem bumped 16G → 24G to be safe under SDXL bf16 + double activations.
#
# lambda_pres = 1.0 as a neutral first value. If the slider under-fires
# on man (enhance term dominated), lower to 0.5. If woman still leaks,
# raise to 2.0 or 5.0. Sweep after this first run.

python sdxl/trained_sliders/training/scripts/train_with_preservation.py \
    --config_file sdxl/trained_sliders/training/configs/smile_man_strong_v3_preserve_woman.yaml \
    --lambda_pres 1.0 \
    --rank 4 \
    --alpha 1.0
