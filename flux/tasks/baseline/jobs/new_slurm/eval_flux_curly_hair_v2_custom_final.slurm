#!/bin/bash
#SBATCH --job-name=eval_curly_v2c
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
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

SAVE_DIR="flux/tasks/baseline/outputs/curly_hair_flux_v2_custom_final_eval"
mkdir -p "$SAVE_DIR"

LORA_DIR="flux/trained_sliders/training/outputs/curly_hair_flux_v2_custom_rank16_noxattn_alpha1/flux-curly_hair_flux_v2_custom"

echo "=== Pre-flight ==="
nvidia-smi
echo "LoRA dir: $LORA_DIR"
echo "Save dir: $SAVE_DIR"
echo ""

python -u flux/tasks/baseline/scripts/generate_flux_slider.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --lora_dirs "$LORA_DIR" \
  --lora_scales 0 0.5 1.0 2.0 3.0 4.0 \
  --prompt "a portrait of a man, realistic photograph" \
  --seeds 42 \
  --save_dir "$SAVE_DIR" \
  --height 1024 \
  --width 1024 \
  --steps 50 \
  --guidance_scale 3.5 \
  --max_sequence_length 512 \
  --rank 16 \
  --alpha 1.0 \
  --train_method noxattn \
  --cuda_device 0

echo ""
echo "=== DONE ==="
find "$SAVE_DIR" -type f | sort
