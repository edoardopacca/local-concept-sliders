#!/bin/bash
#SBATCH --job-name=fert_tight_inv
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/real_editing/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/real_editing/logs/%x_%j.err

set -euxo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub

mkdir -p real_editing/logs real_editing/runs

# ---- CONFIG (override with env vars if needed) ----
RUN_ID="${RUN_ID:-try_tight_001}"
IMAGE="${IMAGE:-real_editing/try.jpeg}"
PROMPT="${PROMPT:-two people standing together}"
MODEL_ID="${MODEL_ID:-$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b}"

# Tight inversion config
NUM_GD_STEPS="${NUM_GD_STEPS:-3}"
GD_STEP_SIZE="${GD_STEP_SIZE:-0.001}"
IPA_SCALE="${IPA_SCALE:-0.4}"
IPA_WEIGHT_NAME="${IPA_WEIGHT_NAME:-ip-adapter-plus_sdxl_vit-h.safetensors}"

python sdxl/tasks/real_editing/scripts/invert_real_image.py \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --run_id "$RUN_ID" \
  --output_root real_editing/runs \
  --model_type sdxl \
  --model_id "$MODEL_ID" \
  --backend tight_inversion \
  --steps 50 \
  --guidance_scale 1.0 \
  --seed 1234 \
  --num_gd_steps "$NUM_GD_STEPS" \
  --gd_step_size "$GD_STEP_SIZE" \
  --ipa_scale "$IPA_SCALE" \
  --dtype float16 \
  --device cuda \
