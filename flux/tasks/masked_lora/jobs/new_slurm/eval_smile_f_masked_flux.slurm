#!/bin/bash
#SBATCH --job-name=eval_smile_f_masked_flux
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Masked LoRA eval — smile_person federico_v2 (Flux) — scale 1, 2, 3
# Input:  flux/tasks/masked_lora/runs/eval_smile_person_{01..20}/edited_smile_f_s{1,2,3}.png
# Output: metrics/results_flux_masked/smile_person_f/eval_{results,aggregate}.{csv,json}

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

mkdir -p flux/tasks/masked_lora/logs metrics/results_flux_masked

echo "=== Masked LoRA eval — smile_f (Flux) — scale 1, 2, 3 ==="
nvidia-smi
echo ""

# Remove previous outputs (both smile_person base and smile_person_f)
rm -rf metrics/results_flux_masked/smile_person metrics/results_flux_masked/smile_person_f

python metrics/eval_masked.py \
    --concept     smile_person \
    --edit_prefix smile_f \
    --scales      1 2 3 \
    --runs_root   "$FERT_REPO/flux/tasks/masked_lora/runs" \
    --output_dir  "metrics/results_flux_masked" \
    --device      cuda

# rinomina subdir di output per distinguere da smile_t
mv metrics/results_flux_masked/smile_person metrics/results_flux_masked/smile_person_f

echo ""
echo "=== DONE — metrics/results_flux_masked/smile_person_f/ ==="
ls -la metrics/results_flux_masked/smile_person_f/
