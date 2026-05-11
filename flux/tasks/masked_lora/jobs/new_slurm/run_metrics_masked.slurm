#!/bin/bash
#SBATCH --job-name=metrics_masked
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUNS_ROOT="$FERT_REPO/flux/tasks/masked_lora/runs"

mkdir -p flux/tasks/masked_lora/logs

echo "================================================================"
echo "=== METRICHE MASKED LORA — tutti i concept                 ==="
echo "================================================================"
nvidia-smi
echo ""

for C in age_person curlyhair furlength painterly smile_person; do
    echo "--- [metrics] concept=${C} ---"
    python metrics/eval_masked.py \
        --concept   "${C}" \
        --runs_root "${RUNS_ROOT}" \
        --device    cuda
    echo ""
done

echo "================================================================"
echo "=== COMPLETATO — risultati in metrics/results/              ==="
echo "================================================================"
