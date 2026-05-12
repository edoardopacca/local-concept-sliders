#!/bin/bash
#SBATCH --job-name=lorashop_curly
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# LoRAShop eval — curlyhair_man — 20 immagini con multitoken attention.
# Prompts: flux/tasks/shop_concept/eval/prompts.yaml  (seed 200-219)
# Slider:  federico_v2 / flux-curly_hair_flux_v2_custom
# Scale:   2, 3.5, 5  → edited_lorashop_s2.0.png, s3.5.png, s5.0.png
# Output:  flux/tasks/shop_concept/runs/eval_curlyhair_man_{0200..0219}/
# Metriche: metrics/results_flux_lorashop/curlyhair_man/

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SLIDER="$FERT_REPO/flux/trained_sliders/sliders/federico_v2/flux-curly_hair_flux_v2_custom/slider_0.pt"
RUNS_ROOT="$FERT_REPO/flux/tasks/shop_concept/runs"

mkdir -p "$RUNS_ROOT" flux/tasks/shop_concept/logs

echo "================================================================"
echo "=== LORASHOP EVAL — curlyhair_man — scale 5,6,8            ==="
echo "  SLIDER    = ${SLIDER}"
echo "  RUNS_ROOT = ${RUNS_ROOT}"
echo "================================================================"
nvidia-smi || true
echo ""

if [[ ! -f "${SLIDER}" ]]; then
    echo "[!] Slider non trovato: ${SLIDER}"
    exit 1
fi

echo "--- [generate] curlyhair_man ---"
python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept     curlyhair_man \
    --slider_path "${SLIDER}" \
    --runs_root   "${RUNS_ROOT}" \
    --eval_scales 2 3.5 5 \
    --mode        all \
    --device      cuda
echo ""

echo "--- [metrics] curlyhair_man ---"
python metrics/eval_lorashop.py \
    --concept   curlyhair_man \
    --runs_root "${RUNS_ROOT}" \
    --device    cuda

echo ""
echo "================================================================"
echo "=== COMPLETATO ==="
echo "  Immagini : ${RUNS_ROOT}/eval_curlyhair_man_*/"
echo "  Metriche : metrics/results_flux_lorashop/curlyhair_man/"
echo "================================================================"
