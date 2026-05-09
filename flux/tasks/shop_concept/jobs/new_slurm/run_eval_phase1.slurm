#!/bin/bash
#SBATCH --job-name=eval_phase1
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# run_eval_phase1.slurm — genera SOLO base.png + mask_target.png (20 run × 4 concept)
# Non dipende dai pesi definitivi: usa qualsiasi slider con scale=0.
# Output: flux/tasks/shop_concept/runs/eval_{concept}_{seed:04d}/
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
echo "=== EVAL PHASE 1: base + mask only ==="
echo "  SLIDERS_ROOT=${SLIDERS_ROOT}"
echo "================================================================"
nvidia-smi || true
echo ""

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept smile_woman \
    --slider_path "${SLIDERS_ROOT}/smile_woman_flux_v1/flux-smile_woman_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode base_only \
    --device cuda

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept curlyhair_man \
    --slider_path "${SLIDERS_ROOT}/curlyhair_man_flux_v1/flux-curlyhair_man_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode base_only \
    --device cuda

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept age_woman \
    --slider_path "${SLIDERS_ROOT}/age_woman_flux_v1/flux-age_woman_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode base_only \
    --device cuda

python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \
    --concept furlength_dog \
    --slider_path "${SLIDERS_ROOT}/furlength_dog_flux_v1/flux-furlength_dog_flux_v1/slider_0.pt" \
    --runs_root "${RUNS_ROOT}" \
    --mode base_only \
    --device cuda

echo ""
echo "================================================================"
echo "=== PHASE 1 COMPLETATA ==="
echo "  base.png + mask_target.png in: ${RUNS_ROOT}"
echo "================================================================"
