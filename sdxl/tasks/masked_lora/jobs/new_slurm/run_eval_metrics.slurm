#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# run_eval_metrics.slurm
# =============================================================================
# Compute LPIPS_in/out, CLIP localization, attribute score for all
# 6 concepts. Accepts --concept as an argument (default: all).
#
# Usage:
#   sbatch run_eval_metrics.slurm                      # all concepts
#   sbatch --export=CONCEPT=smile_person run_eval_metrics.slurm  # uno solo
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

CONCEPT="${CONCEPT:-all}"
RUNS_ROOT="${RUNS_ROOT:-$FERT_REPO/sdxl/tasks/masked_lora/runs}"
OUTPUT_DIR="$FERT_REPO/metrics/results_sdxl_masked"

# Pull latest eval_masked.py from NFS (in case BeeGFS has a stale copy)
rsync -avz "lnode01:${SLURM_SUBMIT_DIR}/metrics/eval_masked.py" "$FERT_REPO/metrics/eval_masked.py" \
    && echo "=== Script sync from NFS OK ===" \
    || echo "[WARNING] Could not sync eval_masked.py from NFS — using local copy"

# Force NFS directory cache revalidation before listing
ls -la "$RUNS_ROOT/" > /dev/null 2>&1 || true

echo "================================================================"
echo "=== eval_metrics  concept=${CONCEPT}                        ==="
echo "================================================================"
nvidia-smi || echo "[WARNING] nvidia-smi non disponibile su questo nodo"
echo ""

if [ "$CONCEPT" = "all" ]; then
    for C in age_person curlyhair daynight furlength painterly smile_person; do
        echo "--- concept=${C} ---"
        python metrics/eval_masked.py \
            --runs_root  "$RUNS_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --concept    "$C" \
            --device     cuda
    done
else
    python metrics/eval_masked.py \
        --runs_root  "$RUNS_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --concept    "$CONCEPT" \
        --device     cuda
fi

echo "================================================================"
echo "=== eval_metrics COMPLETED                                 ==="
echo "=== Output: ${OUTPUT_DIR}/{concept}/eval_aggregate.json    ==="
echo "================================================================"

# Sync results back to submit node (NFS) so they are visible via Cyberduck
echo "=== Syncing results to submit node ==="
rsync -avz "$OUTPUT_DIR/" "lnode01:${SLURM_SUBMIT_DIR}/metrics/results_sdxl_masked/" \
    && echo "=== Sync OK ===" \
    || echo "[WARNING] rsync to lnode01 failed — results are on compute node only"
