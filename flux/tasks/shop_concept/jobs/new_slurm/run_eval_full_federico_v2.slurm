#!/bin/bash
#SBATCH --job-name=lorashop_fed_v2
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# run_eval_full_federico_v2.slurm
# =============================================================================
# LoRAShop eval completa con i pesi di Federico (v2_custom):
# 4 concept × 20 immagini = 80 run + metriche.
#
# Le run vengono salvate in una sotto-directory separata rispetto a trial_v1
# per non sovrascrivere i risultati già generati.
#
# CyberDuck: Sliders-flux/ → $FERT_REPO/flux/trained_sliders/sliders/federico_v2/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S_ROOT="$FERT_REPO/flux/trained_sliders/sliders/federico_v2"
RUNS_ROOT="$FERT_REPO/flux/tasks/shop_concept/runs_federico_v2"

mkdir -p "$RUNS_ROOT" flux/tasks/shop_concept/logs

echo "================================================================"
echo "=== LORASHOP EVAL — federico_v2 — 4 concept × 20 immagini ==="
echo "  RUNS_ROOT = ${RUNS_ROOT}"
echo "================================================================"
nvidia-smi || true
echo ""

for CONCEPT in smile_woman curlyhair_man age_woman furlength_dog; do
    # Mappa concept → slider di Federico
    case "$CONCEPT" in
        smile_woman)   SLIDER="${S_ROOT}/flux-smile_flux_v2_custom/slider_0.pt" ;;
        curlyhair_man) SLIDER="${S_ROOT}/flux-curly_hair_flux_v2_custom/slider_0.pt" ;;
        age_woman)     SLIDER="${S_ROOT}/flux-age_flux_v2_custom/slider_0.pt" ;;
        furlength_dog) SLIDER="${S_ROOT}/flux-fur_flux_v2_custom/slider_0.pt" ;;
    esac

    if [[ ! -f "${SLIDER}" ]]; then
        echo "[!] Slider non trovato per ${CONCEPT}: ${SLIDER}"
        continue
    fi

    echo "--- [generate] concept=${CONCEPT} ---"
    python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
        --concept     "${CONCEPT}" \
        --slider_path "${SLIDER}" \
        --runs_root   "${RUNS_ROOT}" \
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
