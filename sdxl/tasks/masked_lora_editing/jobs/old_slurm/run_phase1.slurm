#!/bin/bash
#SBATCH --job-name=fert_real_p1
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

mkdir -p sdxl/tasks/masked_lora_editing/logs sdxl/tasks/masked_lora_editing/runs

# ---- CONFIG ----
RUN_ID="real_edit_001"
IMAGE="sdxl/tasks/masked_lora_editing/input/real_image.jpg"
PROMPT="photo of a person"

python sdxl/tasks/masked_lora_editing/scripts/01_invert_real.py \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --run_id "$RUN_ID" \
  --output_root sdxl/tasks/masked_lora_editing/runs \
  --steps 50 \
  --guidance_scale 7.5 \
  --device cuda \
  --dtype float32 \
  --local_files_only
