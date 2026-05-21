#!/bin/bash
#SBATCH --job-name=eval_painterly_masked_flux
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Masked LoRA eval per painterly (Flux) — scale 1, 1.5, 2
# Input:  flux/tasks/masked_lora/runs/eval_painterly_{01..20}/edited_painterly_s{1.0,1.5,2.0}.png
# Output: metrics/results_flux_masked/painterly/eval_{results,aggregate}.{csv,json}
#
# Note: runs 01-03 have additional scales 7/8/10 but those are absent in runs 04-20,
# so we restrict to the 3 scales present in all 20 runs.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

mkdir -p flux/tasks/masked_lora/logs

echo "=== Masked LoRA eval — painterly (Flux) — scale 1, 1.5, 2 ==="
nvidia-smi
echo ""

python metrics/eval_masked.py \
    --concept     painterly \
    --scales      1 1.5 2 \
    --runs_root   "$FERT_REPO/flux/tasks/masked_lora/runs" \
    --output_dir  "metrics/results_flux_masked" \
    --device      cuda

echo ""
echo "=== DONE — metrics/results_flux_masked/painterly/ ==="
