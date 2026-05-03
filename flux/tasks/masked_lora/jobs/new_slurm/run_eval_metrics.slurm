#!/bin/bash
#SBATCH --job-name=flux_eval_metrics
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# run_eval_metrics.slurm  (Flux)
# =============================================================================
# Calcola LPIPS + CLIP localization per tutti i 6 concept Flux.
#
# Uso:
#   sbatch flux/tasks/masked_lora/jobs/new_slurm/run_eval_metrics.slurm
#
# Per un solo concept:
#   sbatch --export=CONCEPT=smile_person \
#     flux/tasks/masked_lora/jobs/new_slurm/run_eval_metrics.slurm
#
# Output: metrics/results/{concept}/eval_results.csv + eval_aggregate.json
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

CONCEPT="${CONCEPT:-all}"
RUNS_ROOT="${RUNS_ROOT:-$FERT_REPO/flux/tasks/masked_lora/runs}"

echo "================================================================"
echo "=== eval_metrics Flux  concept=${CONCEPT}                   ==="
echo "================================================================"
nvidia-smi || echo "[WARNING] nvidia-smi non disponibile"
echo ""

if [ "$CONCEPT" = "all" ]; then
    for C in age_person curlyhair daynight furlength painterly smile_person; do
        echo "--- concept=${C} ---"
        python metrics/eval_masked.py \
            --runs_root "$RUNS_ROOT" \
            --concept   "$C" \
            --device    cuda
    done
else
    python metrics/eval_masked.py \
        --runs_root "$RUNS_ROOT" \
        --concept   "$CONCEPT" \
        --device    cuda
fi

echo "================================================================"
echo "=== eval_metrics Flux COMPLETATO                            ==="
echo "=== Output: metrics/results/{concept}/eval_aggregate.json  ==="
echo "================================================================"

echo "=== Syncing results to submit node ==="
rsync -avz "$FERT_REPO/metrics/results/" "lnode01:${SLURM_SUBMIT_DIR}/metrics/results/" \
    && echo "=== Sync OK ===" \
    || echo "[WARNING] rsync to lnode01 failed — results are on compute node only"
