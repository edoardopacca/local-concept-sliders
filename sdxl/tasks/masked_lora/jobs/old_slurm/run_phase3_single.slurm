#!/bin/bash
#SBATCH --job-name=fert_p3_one
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

set -euo pipefail
source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env
export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${RUN_ID:?RUN_ID not set}"
: "${SLIDER_PATH:?SLIDER_PATH not set}"

python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
  --run_dir "sdxl/tasks/masked_lora/runs/${RUN_ID}" \
  --slider_path "$SLIDER_PATH" \
  --slider_scale 2.0 \
  --rank 4 \
  --start_noise 700 \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --device cuda \
  --dtype float16 \
  --mask_name mask_target.png \
  --output_name edited_target_only.png \
  --skip_metrics
