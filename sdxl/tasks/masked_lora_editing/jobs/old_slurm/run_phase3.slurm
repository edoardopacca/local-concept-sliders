#!/bin/bash
#SBATCH --job-name=fert_real_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/masked_lora_editing/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora_editing/logs/%x_%j.err

set -euxo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p sdxl/tasks/masked_lora_editing/logs

# ---- CONFIG ----
RUN_ID="real_edit_001"
SLIDER_PATH="sdxl/trained_sliders/sliders/smiling_sd14.pt"
MASK_NAME="mask_target.png"

python sdxl/tasks/masked_lora_editing/scripts/03_masked_edit_real.py \
  --run_dir "sdxl/tasks/masked_lora_editing/runs/${RUN_ID}" \
  --slider_path "$SLIDER_PATH" \
  --slider_scale 8.0 \
  --rank 4 \
  --start_noise 900 \
  --mask_name "$MASK_NAME" \
  --output_name edited_target_only_strong.png \
  --device cuda \
  --dtype float32 \
  --local_files_only \
  --skip_metrics
