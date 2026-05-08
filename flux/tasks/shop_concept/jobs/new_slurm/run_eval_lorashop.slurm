#!/bin/bash
#SBATCH --job-name=eval_lorashop
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# run_eval_lorashop.slurm — LoRAShop localization eval (4 concepts × 20 images)
# =============================================================================
# Per ogni concept:
#   1. genera 20 run (base + mask_target + edited s1/s2/s3) via RealGenerationPipeline
#   2. calcola metriche di localizzazione (lpips_localization, clip_localization)
#
# Output:
#   flux/tasks/shop_concept/runs/eval_{concept}_{seed:04d}/
#   metrics/results_flux_lorashop/{concept}/eval_results.csv
#   metrics/results_flux_lorashop/{concept}/eval_aggregate.json
#
# Prerequisito: slider trial_v1 trainati e disponibili in:
#   flux/trained_sliders/sliders/trial_v1/{concept}_flux_v1/
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

RUNS_ROOT="$FERT_REPO/flux/tasks/shop_concept/runs"
SLIDERS_ROOT="$FERT_REPO/flux/trained_sliders/sliders/trial_v1"

echo "================================================================"
echo "=== EVAL LORASHOP: generate runs ==="
echo "================================================================"
nvidia-smi || true
echo ""

for CONCEPT in smile_woman curlyhair_man age_woman furlength_dog; do
    SLIDER_NAME="${CONCEPT}_flux_v1"
    SLIDER_PATH="${SLIDERS_ROOT}/${SLIDER_NAME}/flux-${SLIDER_NAME}/slider_0.pt"

    echo ""
    echo "--- [generate] concept=${CONCEPT} ---"
    echo "    slider: ${SLIDER_PATH}"

    python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
        --concept "${CONCEPT}" \
        --slider_path "${SLIDER_PATH}" \
        --runs_root "${RUNS_ROOT}" \
        --height 1024 \
        --width 1024 \
        --steps 30 \
        --guidance_scale 3.5 \
        --device cuda
done

echo ""
echo "================================================================"
echo "=== EVAL LORASHOP: compute metrics ==="
echo "================================================================"
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
