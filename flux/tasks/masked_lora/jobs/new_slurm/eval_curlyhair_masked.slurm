#!/bin/bash
#SBATCH --job-name=eval_curly_masked
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Masked LoRA eval per curlyhair — scale 5, 6, 8 → slider_1, slider_2, slider_3
# Input:  flux/tasks/masked_lora/runs/eval_curlyhair_{01..20}/edited_curly_s{5,6,8}.0.png
# Output: metrics/results_flux_masked/curlyhair/eval_{results,aggregate}.{csv,json}

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

mkdir -p flux/tasks/masked_lora/logs

echo "=== Masked LoRA eval — curlyhair — scale 5,6,8 ==="
nvidia-smi
echo ""

python metrics/eval_masked.py \
    --concept     curlyhair \
    --edit_prefix curly \
    --scales      5 6 8 \
    --runs_root   "$FERT_REPO/flux/tasks/masked_lora/runs" \
    --output_dir  "metrics/results_flux_masked" \
    --device      cuda

echo ""
echo "=== DONE — metrics/results_flux_masked/curlyhair/ ==="
