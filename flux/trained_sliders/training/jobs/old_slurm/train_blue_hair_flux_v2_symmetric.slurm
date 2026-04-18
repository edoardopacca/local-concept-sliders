#!/bin/bash
#SBATCH --job-name=flux_bluehair_v2_sym
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_blue_hair_flux_v2_symmetric.slurm
# =============================================================================
# Blue-hair slider v2 - fix del shortcut learning di v1 (imparava asian
# appearance invece del colore capelli). Stessi iperparam di v1, cambia
# SOLO il prompt design (disambiguanti realistic/dyed/brown).
#
#   - 10 entries: 5 male ethnicities + 5 female ethnicities
#   - positive      "hair dyed blue, bright blue hair color, realistic photograph, natural skin"
#   - unconditional "natural brown hair, realistic photograph, natural skin"
#   - guidance_scale 4, rank 16, xattn, alpha 1, 500 step, lr 0.002
#
# Costo atteso: ~1h (come v1).
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

python -c "import yaml" 2>/dev/null || pip install pyyaml --quiet

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pre-flight ==="
nvidia-smi
echo ""

OUT_DIR="flux/trained_sliders/training/outputs/blue_hair_flux_v2_symmetric_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name blue_hair_flux_v2_symmetric \
    --prompts_yaml flux/trained_sliders/training/prompts/old_prompt/blue_hair_flux_v2_symmetric.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
