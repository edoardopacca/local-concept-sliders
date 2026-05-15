#!/bin/bash
#SBATCH --job-name=eval_smile_t_masked_flux
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Masked LoRA eval — smile_person trial_v1 (Flux) — scale 1, 1.5, 2
# Input:  flux/tasks/masked_lora/runs/eval_smile_person_{01..20}/edited_smile_t_s{1,1.5,2}.png
# Output: metrics/results_flux_masked/smile_person_t/eval_{results,aggregate}.{csv,json}

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

mkdir -p flux/tasks/masked_lora/logs metrics/results_flux_masked

echo "=== Masked LoRA eval — smile_t (Flux) — scale 1, 1.5, 2 ==="
nvidia-smi
echo ""

# rimuovi output precedenti (sia smile_person base che smile_person_t)
rm -rf metrics/results_flux_masked/smile_person metrics/results_flux_masked/smile_person_t

python metrics/eval_masked.py \
    --concept     smile_person \
    --edit_prefix smile_t \
    --scales      1 1.5 2 \
    --runs_root   "$FERT_REPO/flux/tasks/masked_lora/runs" \
    --output_dir  "metrics/results_flux_masked" \
    --device      cuda

# rinomina subdir di output per distinguere da smile_f
mv metrics/results_flux_masked/smile_person metrics/results_flux_masked/smile_person_t

echo ""
echo "=== DONE — metrics/results_flux_masked/smile_person_t/ ==="
ls -la metrics/results_flux_masked/smile_person_t/
