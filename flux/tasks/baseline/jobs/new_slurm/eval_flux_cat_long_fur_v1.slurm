#!/bin/bash
#SBATCH --job-name=eval_flux_cat_longfur_v1
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR="flux/tasks/baseline/outputs/cat_long_fur_v1_eval"
mkdir -p "$SAVE_DIR"

LORA_DIR="flux/trained_sliders/training/outputs/cat_long_fur_flux_v1_symmetric_rank16_xattn_alpha1/flux-cat_long_fur_flux_v1_symmetric_step100"

echo "=== Pre-flight ==="
nvidia-smi
echo "LoRA: $LORA_DIR"
echo "Save dir: $SAVE_DIR"
echo ""

python -u flux/tasks/baseline/scripts/generate_flux_slider.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --lora_dirs "$LORA_DIR" \
  --lora_scales 0 0.2 0.4 0.6 0.8 \
  --prompt "a dog and a cat, realistic photograph" \
  --seeds 42 \
  --save_dir "$SAVE_DIR" \
  --height 512 \
  --width 512 \
  --steps 30 \
  --guidance_scale 3.5 \
  --max_sequence_length 512 \
  --rank 16 \
  --alpha 1.0 \
  --train_method xattn \
  --cuda_device 0
  --skip_slider_timestep_till 10

echo ""
echo "=== DONE ==="
find "$SAVE_DIR" -type f | sort
