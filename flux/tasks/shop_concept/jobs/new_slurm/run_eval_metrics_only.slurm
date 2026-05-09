#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# run_eval_metrics_only.slurm — calcola solo le metriche sulle runs già generate
# Usa quando le immagini in runs/eval_* esistono già ma le metriche sono fallite
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

RUNS_ROOT="$FERT_REPO/flux/tasks/shop_concept/runs"

echo "================================================================"
echo "=== EVAL LORASHOP: compute metrics only ==="
echo "================================================================"
nvidia-smi || true
echo ""

for CONCEPT in smile_woman curlyhair_man age_woman furlength_dog; do
    echo "--- [metrics] concept=${CONCEPT} ---"
    python metrics/eval_lorashop.py \
        --concept "${CONCEPT}" \
        --runs_root "${RUNS_ROOT}" \
        --device cuda
done

echo ""
echo "================================================================"
echo "=== COMPLETATO ==="
echo "  Results: metrics/results_flux_lorashop/"
echo "================================================================"
