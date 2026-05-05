#!/bin/bash
#SBATCH --job-name=eval_selectivity
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# run_eval_selectivity.slurm
# =============================================================================
# Calcola le metriche di selettività per tutti e 4 i concept SDXL.
#
# Uso:
#   sbatch sdxl/tasks/selectivity/jobs/new_slurm/run_eval_selectivity.slurm
#
# Per un solo concept:
#   sbatch --export=CONCEPT=age \
#     sdxl/tasks/selectivity/jobs/new_slurm/run_eval_selectivity.slurm
#
# Output: metrics/results_sdxl_selectivity/{concept}/eval_results.csv
#                                                    /eval_aggregate.json
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

CONCEPT="${CONCEPT:-all}"
RUNS_ROOT="${RUNS_ROOT:-$FERT_REPO/sdxl/tasks/selectivity/runs}"
OUTPUT_DIR="$FERT_REPO/metrics/results_sdxl_selectivity"

# Pull latest script from NFS
rsync -avz "lnode01:${SLURM_SUBMIT_DIR}/metrics/eval_selectivity.py" \
    "$FERT_REPO/metrics/eval_selectivity.py" \
    && echo "=== Script sync from NFS OK ===" \
    || echo "[WARNING] Could not sync eval_selectivity.py from NFS — using local copy"

ls -la "$RUNS_ROOT/" > /dev/null 2>&1 || true

echo "================================================================"
echo "=== eval_selectivity  concept=${CONCEPT}                    ==="
echo "================================================================"
nvidia-smi || echo "[WARNING] nvidia-smi non disponibile"
echo ""

if [ "$CONCEPT" = "all" ]; then
    for C in age curlyhair furlength smile; do
        echo "--- concept=${C} ---"
        python metrics/eval_selectivity.py \
            --runs_root  "$RUNS_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --concept    "$C" \
            --device     cuda
    done
else
    python metrics/eval_selectivity.py \
        --runs_root  "$RUNS_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --concept    "$CONCEPT" \
        --device     cuda
fi

echo "================================================================"
echo "=== eval_selectivity COMPLETATO                             ==="
echo "=== Output: ${OUTPUT_DIR}/{concept}/eval_aggregate.json    ==="
echo "================================================================"

echo "=== Syncing results to submit node ==="
rsync -avz "$OUTPUT_DIR/" "lnode01:${SLURM_SUBMIT_DIR}/metrics/results_sdxl_selectivity/" \
    && echo "=== Sync OK ===" \
    || echo "[WARNING] rsync to lnode01 failed — results are on compute node only"
