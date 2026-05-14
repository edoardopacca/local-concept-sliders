#!/bin/bash
#SBATCH --job-name=eval_age_masked_flux
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Masked LoRA eval per age_person (Flux) — scale 0.5, 1, 1.5
# Input:  flux/tasks/masked_lora/runs/eval_age_person_{01..20}/edited_age_s{0.5,1.0,1.5}.png
# Output: metrics/results_flux_masked/age_person/eval_{results,aggregate}.{csv,json}

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

mkdir -p flux/tasks/masked_lora/logs

echo "=== Masked LoRA eval — age_person (Flux) — scale 0.5, 1, 1.5 ==="
nvidia-smi
echo ""

python metrics/eval_masked.py \
    --concept     age_person \
    --scales      0.5 1 1.5 \
    --runs_root   "$FERT_REPO/flux/tasks/masked_lora/runs" \
    --output_dir  "metrics/results_flux_masked" \
    --device      cuda

echo ""
echo "=== DONE — metrics/results_flux_masked/age_person/ ==="
