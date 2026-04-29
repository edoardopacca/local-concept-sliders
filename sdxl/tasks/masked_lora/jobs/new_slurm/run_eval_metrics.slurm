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
# Calcola LPIPS_in/out, Localization Score CLIP, Attribute Score per tutti
# e 6 concept. Accetta --concept come argomento (default: tutti).
#
# Usage:
#   sbatch run_eval_metrics.slurm                      # tutti i concept
#   sbatch --export=CONCEPT=smile_person run_eval_metrics.slurm  # uno solo
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

CONCEPT="${CONCEPT:-all}"

echo "================================================================"
echo "=== eval_metrics  concept=${CONCEPT}                        ==="
echo "================================================================"
nvidia-smi
echo ""

if [ "$CONCEPT" = "all" ]; then
    for C in age_person curlyhair daynight furlength painterly smile_person; do
        echo "--- concept=${C} ---"
        python metrics/eval_masked.py \
            --runs_root sdxl/tasks/masked_lora/runs \
            --concept "$C" \
            --device cuda
    done
else
    python metrics/eval_masked.py \
        --runs_root sdxl/tasks/masked_lora/runs \
        --concept "$CONCEPT" \
        --device cuda
fi

echo "================================================================"
echo "=== eval_metrics COMPLETATO                                 ==="
echo "==="
echo "=== Output: metrics/results/{concept}/eval_aggregate.json  ==="
echo "================================================================"
