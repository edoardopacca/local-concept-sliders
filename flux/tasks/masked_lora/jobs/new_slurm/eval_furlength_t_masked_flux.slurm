#!/bin/bash
#SBATCH --job-name=eval_furlength_t_masked_flux
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Masked LoRA eval — furlength trial_v1 (Flux) — scale 1, 2, 3
# Input:  flux/tasks/masked_lora/runs/eval_furlength_{01..20}/edited_furlength_t_s{1,2,3}.png
# Output: metrics/results_flux_masked/furlength_t/eval_{results,aggregate}.{csv,json}

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

mkdir -p flux/tasks/masked_lora/logs metrics/results_flux_masked

echo "=== Masked LoRA eval — furlength_t (Flux) — scale 1, 2, 3 ==="
nvidia-smi
echo ""

# Remove previous outputs
rm -rf metrics/results_flux_masked/furlength metrics/results_flux_masked/furlength_t

python metrics/eval_masked.py \
    --concept     furlength \
    --edit_prefix furlength_t \
    --scales      1 2 3 \
    --runs_root   "$FERT_REPO/flux/tasks/masked_lora/runs" \
    --output_dir  "metrics/results_flux_masked" \
    --device      cuda

# Rename the subdir for consistency (in case a furlength_f variant is added later).
mv metrics/results_flux_masked/furlength metrics/results_flux_masked/furlength_t

echo ""
echo "=== DONE — metrics/results_flux_masked/furlength_t/ ==="
ls -la metrics/results_flux_masked/furlength_t/
