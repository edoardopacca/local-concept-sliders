#!/bin/bash
#SBATCH --job-name=p3_edo_furlength
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=gnode02,gnode04
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# Phase 3 masked LoRA — furlength × 20 run — Edoardo
# furlength_t (trial_v1)   : s1, s2, s3
# furlength_f (federico_v2): s5, s7, s8

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
F_ROOT="$FERT_REPO/flux/trained_sliders/sliders/federico_v2"

mkdir -p flux/tasks/masked_lora/logs "$CACHE_DIR"

echo "================================================================"
echo "=== PHASE 3 — furlength × 20 run — Edoardo                ==="
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

run_concept "furlength" \
    "${F_ROOT}/flux-fur_flux_v2_custom/slider_0.pt" \
    "furlength_f" "5 7 8"

echo "================================================================"
echo "=== PHASE 3 furlength COMPLETATA                           ==="
echo "  furlength_t (trial_v1)   : s1, s2, s3"
echo "  furlength_f (federico_v2): s5, s7, s8"
echo "================================================================"
