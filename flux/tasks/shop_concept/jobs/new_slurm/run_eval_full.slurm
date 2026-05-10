#!/bin/bash
#SBATCH --job-name=eval_full
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# run_eval_full.slurm — eval completa: 4 concept × 20 immagini = 80 run
#   per ogni run: base.png + mask_target.png + edited (scale 1/2/3) + metriche
#
# Uso:
#   sbatch run_eval_full.slurm
#
# Per usare slider diversi da trial_v1:
#   sbatch --export=ALL,SLIDERS_ROOT=/path/to/altri_sliders run_eval_full.slurm
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
echo "=== EVAL FULL: 4 concept × 20 immagini ==="
echo "  SLIDERS_ROOT = ${SLIDERS_ROOT}"
echo "  RUNS_ROOT    = ${RUNS_ROOT}"
echo "================================================================"
nvidia-smi || true
echo ""

for CONCEPT in smile_woman curlyhair_man age_woman furlength_dog; do
    SLIDER="${SLIDERS_ROOT}/${CONCEPT}_flux_v1/flux-${CONCEPT}_flux_v1/slider_0.pt"

    if [[ ! -f "${SLIDER}" ]]; then
        echo "[!] Slider non trovato per ${CONCEPT}: ${SLIDER}"
        echo "    Salta concept."
        continue
    fi

    echo "--- [generate] concept=${CONCEPT} ---"
    python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
        --concept   "${CONCEPT}" \
        --slider_path "${SLIDER}" \
        --runs_root "${RUNS_ROOT}" \
        --mode all \
        --device cuda
    echo ""
done

echo "--- [metrics] calcolo metriche di localizzazione ---"
for CONCEPT in smile_woman curlyhair_man age_woman furlength_dog; do
    echo "--- [metrics] concept=${CONCEPT} ---"
    python metrics/eval_lorashop.py \
        --concept   "${CONCEPT}" \
        --runs_root "${RUNS_ROOT}" \
        --device cuda
done

echo ""
echo "================================================================"
echo "=== COMPLETATO ==="
echo "  Immagini : ${RUNS_ROOT}"
echo "  Metriche : metrics/results_flux_lorashop/"
echo "================================================================"
