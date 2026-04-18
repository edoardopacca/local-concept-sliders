#!/bin/bash
#SBATCH --job-name=flux_age_man_v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_age_man_flux_v1_yaml_trick.slurm
# =============================================================================
# Concept slider age (old vs young) su uomo, con donna in preservation
# YAML-trick. Analogo esatto di v3 smile ma con attributo differente.
#
# Scopo: distinguere se il leakage osservato su smile v3/v4 e'
#   A) attribute-specific (solo smile ha problemi di binding)
#   B) structural (qualsiasi slider asymmetric su Flux leakka)
#
# Setup IDENTICO a v3 smile:
#   - 5 enhance man + 5 preservation woman (YAML trick)
#   - 500 max_train_steps, rank 16, xattn, alpha 1
#   - lr 0.002, warmup 200
#   - save_every 100 -> checkpoint a 100/200/300/400/final
#   - guidance_scale 2 (matchiamo v3/v4)
#
# Costo atteso: ~5-6 s/step -> 500 step ~45-55 min + load ~7 min = ~1h.
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

# safety pyyaml
python -c "import yaml" 2>/dev/null || pip install pyyaml --quiet

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pre-flight ==="
nvidia-smi
echo ""

OUT_DIR="flux/trained_sliders/training/outputs/age_man_flux_v1_yaml_trick_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name age_man_flux_v1_yaml_trick \
    --prompts_yaml flux/trained_sliders/training/prompts/old_prompt/age_man_flux_v1_yaml_trick.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
