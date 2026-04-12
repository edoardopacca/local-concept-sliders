#!/bin/bash
#SBATCH --job-name=fert_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

set -euxo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
# Se le metriche CLIP falliscono al primo run (modello non in cache), commenta queste due righe:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p sdxl/tasks/masked_lora/logs

# ---- Prova: stesso RUN_ID della fase 1; slider come exp_generation/run_generation.sh ----
RUN_ID="fert_test_007"
SLIDER_PATH="sdxl/trained_sliders/sliders/age_sdxl.pt"
SLIDER_SCALE=2.0
START_NOISE=700

python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
  --run_dir "sdxl/tasks/masked_lora/runs/${RUN_ID}" \
  --slider_path "$SLIDER_PATH" \
  --slider_scale "$SLIDER_SCALE" \
  --rank 4 \
  --start_noise "$START_NOISE" \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --device cuda \
  --dtype float16 \
  --mask_name mask.png \
  --output_name edited.png
