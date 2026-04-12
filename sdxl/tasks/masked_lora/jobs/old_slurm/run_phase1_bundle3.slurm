#!/bin/bash
#SBATCH --job-name=fert_p1_bundle3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:30:00
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

PROMPT="two people, shoulder-up portrait, realistic photo, 4k"
NEG="full body, blurry, duplicate face, low quality"

python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
  --prompt "$PROMPT" --negative_prompt "$NEG" --seed 1234 \
  --steps 50 --guidance_scale 5.0 --height 1024 --width 1024 \
  --run_id exp01 --output_root sdxl/tasks/masked_lora/runs \
  --model_id $FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --device cuda --dtype float16

python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
  --prompt "$PROMPT" --negative_prompt "$NEG" --seed 1234 \
  --steps 50 --guidance_scale 5.0 --height 1024 --width 1024 \
  --run_id exp02 --output_root sdxl/tasks/masked_lora/runs \
  --model_id $FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --device cuda --dtype float16

python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
  --prompt "$PROMPT" --negative_prompt "$NEG" --seed 1234 \
  --steps 50 --guidance_scale 5.0 --height 1024 --width 1024 \
  --run_id exp03 --output_root sdxl/tasks/masked_lora/runs \
  --model_id $FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --device cuda --dtype float16
