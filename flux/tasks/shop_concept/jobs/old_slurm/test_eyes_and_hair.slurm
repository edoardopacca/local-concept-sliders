#!/bin/bash
#SBATCH --job-name=shop_eyes_hair_test
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# test_eyes_and_hair.slurm
# =============================================================================
# Test veloce dei due nuovi slider trainati (big_eyes + blue_hair) su un
# singolo prompt con una donna in primo piano. Scale negative, zero e
# positive per vedere la direzione e la magnitude.
#
# Due sweep in sequenza nello stesso job (Flux si ricarica ma evitiamo la
# coda 1-job/utente):
#   1) big_eyes:  scale -1.0, 0.0, 1.0 -> 3 img + mask dump
#   2) blue_hair: scale -1.0, 0.0, 1.0 -> 3 img + mask dump
# Totale ~6 img, circa 10-15 min.
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
echo "[1/2] BIG_EYES SWEEP"
echo "============================================================"
python -u -m flux.tasks.shop_concept.scripts.sweep \
    --slider_path "flux/trained_sliders/training/outputs/big_eyes_flux_v1_symmetric_rank16_xattn_alpha1/flux-big_eyes_flux_v1_symmetric/slider_0.pt" \
    --target_prompt "woman" \
    --prompt "$PROMPT" \
    --seeds 42 \
    --scales -1.0 0.0 1.0 \
    --output_dir "shop_concept/outputs/test_big_eyes_v1" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache" \
    --dump_masks

echo ""
# -----------------------------------------------------------------------------
# Sweep 2: blue_hair
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[2/2] BLUE_HAIR SWEEP"
echo "============================================================"
python -u -m flux.tasks.shop_concept.scripts.sweep \
    --slider_path "flux/trained_sliders/training/outputs/blue_hair_flux_v1_symmetric_rank16_xattn_alpha1/flux-blue_hair_flux_v1_symmetric/slider_0.pt" \
    --target_prompt "woman" \
    --prompt "$PROMPT" \
    --seeds 42 \
    --scales -1.0 0.0 1.0 \
    --output_dir "shop_concept/outputs/test_blue_hair_v1" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache" \
    --dump_masks

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/test_big_eyes_v1 shop_concept/outputs/test_blue_hair_v1
