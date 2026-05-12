#!/bin/bash
#SBATCH --job-name=phase3_rebe_fur
#SBATCH --account=3352899
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# phase3_rebe_furlength.slurm
# =============================================================================
# Phase 3 masked LoRA — furlength × 20 run — per Rebe.
# Prerequisiti in RUNS_ROOT:
#   eval_furlength_{01..20}/base.png + mask_target.png + metadata.json
#
# Pesi richiesti (in $FERT_REPO):
#   trial_v1 : flux/trained_sliders/sliders/trial_v1/furlength_dog_flux_v1/flux-furlength_dog_flux_v1/slider_0.pt
#
# Output per ogni run:
#   edited_furlength_t_s{1,2,3}.png   (trial_v1)
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS_ROOT="$FERT_REPO/flux/tasks/masked_lora/runs"
CACHE_DIR="$FERT_REPO/flux/tasks/masked_lora/_peft_cache"
T_ROOT="$FERT_REPO/flux/trained_sliders/sliders/trial_v1"

mkdir -p flux/tasks/masked_lora/logs "$CACHE_DIR"

echo "================================================================"
echo "=== PHASE 3 — furlength × 20 run — Rebe                  ==="
echo "================================================================"
nvidia-smi
echo ""

run_concept() {
    local CONCEPT="$1"
    local SLIDER="$2"
    local PREFIX="$3"
    local SCALES="$4"

    echo "======== ${CONCEPT} — ${PREFIX} ========"
    for i in $(seq 1 20); do
        RUN_NUM=$(printf "%02d" "$i")
        RUN_DIR="${RUNS_ROOT}/eval_${CONCEPT}_${RUN_NUM}"
        [ -d "$RUN_DIR" ] || { echo "[SKIP] dir mancante: $RUN_DIR"; continue; }
        [ -f "${RUN_DIR}/mask_target.png" ] || { echo "[SKIP] mask mancante: $RUN_DIR"; continue; }

        python flux/tasks/masked_lora/scripts/03_masked_edit.py \
            --run_dir      "$RUN_DIR" \
            --slider_path  "$SLIDER" \
            --slider_scales $SCALES \
            --mask_name    mask_target.png \
            --lora_fill_rank 16 \
            --cache_dir    "$CACHE_DIR" \
            --device       cuda

        for SCALE in $SCALES; do
            SRC="${RUN_DIR}/edited_scale${SCALE}.png"
            [ -f "$SRC" ] || SRC="${RUN_DIR}/edited_scale${SCALE}.0.png"
            DST="${RUN_DIR}/edited_${PREFIX}_s${SCALE}.png"
            [ -f "$SRC" ] && mv "$SRC" "$DST"
        done
        echo "  [ok] eval_${CONCEPT}_${RUN_NUM}"
    done
    echo ""
}

run_concept "furlength" \
    "${T_ROOT}/furlength_dog_flux_v1/flux-furlength_dog_flux_v1/slider_0.pt" \
    "furlength_t" "1 2 3"

echo "================================================================"
echo "=== PHASE 3 furlength COMPLETATA                          ==="
echo "  furlength_t (trial_v1) : s1, s2, s3"
echo "================================================================"
