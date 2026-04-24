#!/bin/bash
#SBATCH --job-name=train_age_flux_v1_sym
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_age_flux_v1_symmetric.slurm
# =============================================================================
# Training concept slider AGE su Flux.1-dev, setting simmetrico:
#   - 10 entries (5 male + 5 female)
#   - TUTTE enhance (nessun YAML trick di preservation)
#   - L'effetto età si applica a qualunque soggetto
#
# Hyperparameters identici a smile_flux_v5_symmetric per apples-to-apples.
# Output: flux/trained_sliders/sliders/age_flux_v1_symmetric_rank16_xattn_alpha1/
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

OUT_DIR="flux/trained_sliders/sliders/age_flux_v1_symmetric_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name age_flux_v1_symmetric \
    --prompts_yaml flux/trained_sliders/training/prompts/new_prompt/age_flux_v1_symmetric.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
