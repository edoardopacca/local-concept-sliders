#!/bin/bash
#SBATCH --job-name=flux_pittoric_v1c
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

python -c "import yaml" 2>/dev/null || pip install pyyaml --quiet

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pre-flight ==="
nvidia-smi
echo ""

OUT_DIR="flux/trained_sliders/training/outputs/pittoric_style_flux_v1_custom_rank4_noxattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 1000 \
    --num_inference_steps 50 \
    --output_dir "$OUT_DIR" \
    --slider_name pittoric_style_flux_v1_custom \
    --prompts_yaml flux/trained_sliders/training/prompts/new_prompt/pittoric_style_flux_v1_custom.yaml \
    --rank 4 \
    --alpha 1 \
    --train_method noxattn \
    --lr 0.0002 \
    --lr_warmup_steps 100 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
