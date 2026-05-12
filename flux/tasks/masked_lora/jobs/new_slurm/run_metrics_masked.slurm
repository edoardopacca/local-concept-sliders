#!/bin/bash
#SBATCH --job-name=metrics_masked
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

RUNS_ROOT="$FERT_REPO/flux/tasks/masked_lora/runs"

mkdir -p flux/tasks/masked_lora/logs

run_metrics() {
    local CONCEPT="$1"
    local PREFIX="$2"
    local OUT_DIR="$3"
    echo "--- concept=${CONCEPT}  prefix=${PREFIX} ---"
    python metrics/eval_masked.py \
        --concept     "${CONCEPT}" \
        --edit_prefix "${PREFIX}" \
        --runs_root   "${RUNS_ROOT}" \
        --output_dir  "${OUT_DIR}" \
        --device      cuda
    echo ""
}

echo "================================================================"
echo "=== METRICHE MASKED LORA — trial_v1                        ==="
echo "================================================================"
nvidia-smi
echo ""

run_metrics "age_person"   "age_t"        "metrics/results_trial_v1"
run_metrics "curlyhair"    "curly_t"      "metrics/results_trial_v1"
run_metrics "furlength"    "furlength_t"  "metrics/results_trial_v1"
run_metrics "painterly"    "painterly_t"  "metrics/results_trial_v1"
run_metrics "smile_person" "smile_t"      "metrics/results_trial_v1"

echo "================================================================"
echo "=== METRICHE MASKED LORA — federico_v2                     ==="
echo "================================================================"
echo ""

run_metrics "age_person"   "age_f"        "metrics/results_federico_v2"
run_metrics "curlyhair"    "curly_f"      "metrics/results_federico_v2"
run_metrics "furlength"    "furlength_f"  "metrics/results_federico_v2"
run_metrics "painterly"    "painterly_f"  "metrics/results_federico_v2"
run_metrics "smile_person" "smile_f"      "metrics/results_federico_v2"

echo "================================================================"
echo "=== COMPLETATO                                              ==="
echo "  trial_v1  : metrics/results_trial_v1/"
echo "  federico  : metrics/results_federico_v2/"
echo "================================================================"
