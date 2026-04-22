#!/bin/bash
#SBATCH --job-name=shop_eyes_hair_hi
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# test_eyes_and_hair_highscale.slurm
# =============================================================================
# Test con scale AGGRESSIVE per big_eyes e blue_hair: a scale=1.0 l'effetto
# era troppo debole. Proviamo 2.0, 3.0, 5.0 (come fatto con van Gogh) per
# vedere se serve solo tarare la scale o se il training e' davvero debole.
#
# 4 scale (0, 2, 3, 5) x 2 slider = 8 img, circa 15-20 min.
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p flux/tasks/shop_concept/logs shop_concept/outputs shop_concept/_peft_cache

echo "=== Pre-flight ==="
nvidia-smi
echo ""

PROMPT="a portrait of a woman, studio photography, neutral expression"

# -----------------------------------------------------------------------------
# Sweep 1: big_eyes
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[1/2] BIG_EYES SWEEP (high scale)"
echo "============================================================"
python -u -m flux.tasks.shop_concept.scripts.sweep \
    --slider_path "flux/trained_sliders/training/outputs/big_eyes_flux_v1_symmetric_rank16_xattn_alpha1/flux-big_eyes_flux_v1_symmetric/slider_0.pt" \
    --target_prompt "woman" \
    --prompt "$PROMPT" \
    --seeds 42 \
    --scales 0.0 2.0 3.0 5.0 \
    --output_dir "shop_concept/outputs/test_big_eyes_v1_highscale" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache"

echo ""
echo "============================================================"
echo "[2/2] BLUE_HAIR SWEEP (high scale)"
echo "============================================================"
python -u -m flux.tasks.shop_concept.scripts.sweep \
    --slider_path "flux/trained_sliders/training/outputs/blue_hair_flux_v1_symmetric_rank16_xattn_alpha1/flux-blue_hair_flux_v1_symmetric/slider_0.pt" \
    --target_prompt "woman" \
    --prompt "$PROMPT" \
    --seeds 42 \
    --scales 0.0 2.0 3.0 5.0 \
    --output_dir "shop_concept/outputs/test_blue_hair_v1_highscale" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache"

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/test_big_eyes_v1_highscale shop_concept/outputs/test_blue_hair_v1_highscale
