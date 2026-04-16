#!/bin/bash
#SBATCH --job-name=fert_tight_edit
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

mkdir -p real_editing/logs

# ---- CONFIG (override with env vars if needed) ----
RUN_ID="${RUN_ID:-try_tight_001}"
MASK_FROM_RUN_ID="${MASK_FROM_RUN_ID:-try_001}"   # run that already has SAM mask
MASK_NAME="${MASK_NAME:-mask_target.png}"
SLIDER_PATH="${SLIDER_PATH:-sdxl/trained_sliders/sliders/smiling.pt}"
MODEL_ID="${MODEL_ID:-$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b}"

# Conservative stability-oriented defaults (avoid ROI collapse/black holes)
SLIDER_SCALE="${SLIDER_SCALE:-0.4}"
START_NOISE="${START_NOISE:-350}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.0}"
IMG2IMG_STRENGTH="${IMG2IMG_STRENGTH:-0.22}"
FEATHER_RADIUS="${FEATHER_RADIUS:-16}"
OUTPUT_NAME="${OUTPUT_NAME:-edited_tight_001_v2.png}"

# Ensure mask exists in target run dir
if [ ! -f "real_editing/runs/${RUN_ID}/${MASK_NAME}" ]; then
  cp "real_editing/runs/${MASK_FROM_RUN_ID}/${MASK_NAME}" "real_editing/runs/${RUN_ID}/"
fi
if [ -f "real_editing/runs/${MASK_FROM_RUN_ID}/mask_meta.json" ] && [ ! -f "real_editing/runs/${RUN_ID}/mask_meta.json" ]; then
  cp "real_editing/runs/${MASK_FROM_RUN_ID}/mask_meta.json" "real_editing/runs/${RUN_ID}/"
fi

python sdxl/tasks/real_editing/scripts/edit_real_image_masked.py \
  --run_dir "real_editing/runs/${RUN_ID}" \
  --slider_path "$SLIDER_PATH" \
  --slider_scale "$SLIDER_SCALE" \
  --rank 4 \
  --start_noise "$START_NOISE" \
  --mask_name "$MASK_NAME" \
  --output_name "$OUTPUT_NAME" \
  --guidance_scale "$GUIDANCE_SCALE" \
  --img2img_strength "$IMG2IMG_STRENGTH" \
  --feather_radius "$FEATHER_RADIUS" \
  --pixel_composite \
  --device cuda \
  --dtype float32 \
  --local_files_only \
  --model_id "$MODEL_ID" \
  --skip_metrics
