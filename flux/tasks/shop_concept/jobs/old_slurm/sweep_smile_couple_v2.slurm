#!/bin/bash
#SBATCH --job-name=shop_smile_sweep_v2
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# sweep_smile_couple_v2.slurm
# =============================================================================
# Sweep 4 scale x 2 seed dello slider "smile symmetric" sulla coppia
# (mask solo su "man"). 8 PNG totali.
#
# Differenza vs sweep_smile_couple.slurm v1: usa shop_concept/sweep.py che
# carica Flux UNA sola volta invece che 8 (fix model-load overhead).
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
    --slider_path "flux/trained_sliders/training/outputs/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt" \
    --target_prompt "man" \
    --prompt "a man and a woman facing the camera, studio portrait, neutral expression" \
    --seeds 42 123 \
    --scales 0.0 0.3 0.7 1.0 \
    --output_dir "shop_concept/outputs/sweep_smile_couple" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache"

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/sweep_smile_couple
