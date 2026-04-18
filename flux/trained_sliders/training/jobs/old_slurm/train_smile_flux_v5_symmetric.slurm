#!/bin/bash
#SBATCH --job-name=flux_smile_v5_sym
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_smile_flux_v5_symmetric.slurm
# =============================================================================
# Smile slider simmetrico su Flux, setting paper originale:
#   - 10 entries: 5 male ethnicities + 5 female ethnicities
#   - TUTTE enhance (nessun YAML trick)
#   - guidance_scale 4 (match paper), rank 16, xattn, alpha 1
#
# Baseline "expected behavior": slider di sorriso che va su tutti.
# Serve come riferimento per distinguere:
#   A) LoRA+xattn su Flux impara smile (si', con setting simmetrico)
#   B) asymmetric setting (v3/v4) fallisce per colpa del binding
#
# Costo atteso: ~5-6 s/step -> 500 step ~45-55 min + load ~7 min = ~1h.
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

OUT_DIR="flux/trained_sliders/training/outputs/smile_flux_v5_symmetric_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name smile_flux_v5_symmetric \
    --prompts_yaml flux/trained_sliders/training/prompts/old_prompt/smile_flux_v5_symmetric.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
