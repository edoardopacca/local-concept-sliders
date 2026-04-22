#!/bin/bash
#SBATCH --job-name=shop_bluehair_v2
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# test_blue_hair_v2.slurm
# =============================================================================
# Test blue_hair v2 (disambiguanti realistic/dyed/brown) con scale aggressive.
# Stessi seed/prompt/scale di test_eyes_and_hair_highscale (v1) per confronto
# diretto.
#
# Se v2 fa capelli blu (e non asian): fix funziona.
# Se v2 fa ancora asian: il bias di Flux e' strutturale, servono fix piu'
# pesanti (rank piu' alto, lr diverso, altri colori rari).
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

python -u -m flux.tasks.shop_concept.scripts.sweep \
    --slider_path "flux/trained_sliders/training/outputs/blue_hair_flux_v2_symmetric_rank16_xattn_alpha1/flux-blue_hair_flux_v2_symmetric/slider_0.pt" \
    --target_prompt "woman" \
    --prompt "a portrait of a woman, studio photography, neutral expression" \
    --seeds 42 \
    --scales 0.0 2.0 3.0 5.0 \
    --output_dir "shop_concept/outputs/test_blue_hair_v2_highscale" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache"

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/test_blue_hair_v2_highscale
