#!/bin/bash
#SBATCH --job-name=fert_real_sdxl_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
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
RUN_ID="real_edit_sdxl_001"
IMAGE="sdxl/tasks/masked_lora_editing/input/real_image.jpg"
PROMPT="photo of a person"
MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

python sdxl/tasks/masked_lora_editing/scripts/01_prepare_real_sdxl.py \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --run_id "$RUN_ID" \
  --output_root sdxl/tasks/masked_lora_editing/runs \
  --model_id "$MODEL_ID" \
  --height 1024 \
  --width 1024 \
  --steps 50 \
  --guidance_scale 5.0 \
  --seed 1234 \
  --device cuda \
  --dtype float16 \
  --local_files_only
