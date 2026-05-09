#!/bin/bash
#SBATCH --job-name=eval_phase2
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# run_eval_phase2.slurm — genera edited images + calcola metriche
# Richiede che base.png + mask_target.png esistano già (fase 1 completata).
#
# Uso tipico:
#   sbatch --export=ALL,SLIDERS_ROOT=/path/to/final_sliders run_eval_phase2.slurm
#
# Default SLIDERS_ROOT = trial_v1 (per test rapidi).
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

RUNS_ROOT="$FERT_REPO/flux/tasks/shop_concept/runs"
SLIDERS_ROOT="${SLIDERS_ROOT:-$FERT_REPO/flux/trained_sliders/sliders/trial_v1}"

echo "================================================================"
echo "=== EVAL PHASE 2: edited images + metrics ==="
echo "  SLIDERS_ROOT=${SLIDERS_ROOT}"
echo "================================================================"
nvidia-smi || true
echo ""

# --- Genera immagini editate (edited_only: usa base/mask esistenti) ---
python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept smile_woman \
    --slider_path "${SLIDERS_ROOT}/smile_woman_flux_v1/flux-smile_woman_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode edited_only \
    --device cuda

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept curlyhair_man \
    --slider_path "${SLIDERS_ROOT}/curlyhair_man_flux_v1/flux-curlyhair_man_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode edited_only \
    --device cuda

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept age_woman \
    --slider_path "${SLIDERS_ROOT}/age_woman_flux_v1/flux-age_woman_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode edited_only \
    --device cuda

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept furlength_dog \
    --slider_path "${SLIDERS_ROOT}/furlength_dog_flux_v1/flux-furlength_dog_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode edited_only \
    --device cuda

echo ""
echo "--- Generazione edited completata, calcolo metriche ---"
echo ""

# --- Calcola metriche di localizzazione ---
for CONCEPT in smile_woman curlyhair_man age_woman furlength_dog; do
    echo "--- [metrics] concept=${CONCEPT} ---"
    python metrics/eval_lorashop.py \
        --concept "${CONCEPT}" \
        --runs_root "${RUNS_ROOT}" \
        --device cuda
done

echo ""
echo "================================================================"
echo "=== PHASE 2 COMPLETATA ==="
echo "  Results: metrics/results_flux_lorashop/"
echo "================================================================"
