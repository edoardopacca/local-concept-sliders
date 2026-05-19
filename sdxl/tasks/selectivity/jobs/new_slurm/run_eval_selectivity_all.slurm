#!/bin/bash
#SBATCH --job-name=eval_selectivity_all
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=metrics/logs/%x_%j.out
#SBATCH --error=metrics/logs/%x_%j.err

# =============================================================================
# run_eval_selectivity_all.slurm
# =============================================================================
# Rilancia eval_selectivity.py per tutti e 4 i concept SDXL.
# Motivazione: convenzione di lpips_selectivity invertita a target/nontarget
# (paper convention, higher = better — coerente con generate_paper_tables.py
# e col poster).
# Sovrascrive i JSON/CSV in metrics/results_sdxl_selectivity/<concept>/.
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

RUNS_ROOT="$FERT_REPO/sdxl/tasks/selectivity/runs"
OUTPUT_DIR="$FERT_REPO/metrics/results_sdxl_selectivity"
mkdir -p metrics/logs "$OUTPUT_DIR"

echo "================================================================"
echo "=== eval_selectivity (all 4 concepts) — new ratio convention ==="
echo "================================================================"
nvidia-smi || echo "[WARNING] nvidia-smi non disponibile"
echo ""

for C in age curlyhair furlength smile; do
    echo "--- concept=${C} ---"
    python metrics/eval_selectivity.py \
        --concept    "$C" \
        --runs_root  "$RUNS_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --device     cuda
    echo ""
done

echo "================================================================"
echo "=== DONE — output in: ${OUTPUT_DIR}                          ==="
echo "================================================================"
ls -la "$OUTPUT_DIR"

# Sync results back to submit node (NFS) per Cyberduck/git
echo ""
echo "=== Sync results to submit node (NFS) ==="
rsync -avz "$OUTPUT_DIR/" "lnode01:${SLURM_SUBMIT_DIR}/metrics/results_sdxl_selectivity/" \
    && echo "=== Sync OK ===" \
    || echo "[WARNING] rsync to lnode01 failed — risultati solo su compute node"
