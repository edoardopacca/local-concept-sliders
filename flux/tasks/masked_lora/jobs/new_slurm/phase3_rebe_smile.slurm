#!/bin/bash
#SBATCH --job-name=phase3_rebe_smile
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
# phase3_rebe_smile.slurm
# =============================================================================
# Phase 3 masked LoRA — smile_person × 20 run — per Rebe.
# Prerequisiti in RUNS_ROOT:
#   eval_smile_person_{01..20}/base.png + mask_target.png + metadata.json
#
# Pesi richiesti (in $FERT_REPO):
#   trial_v1 : flux/trained_sliders/sliders/trial_v1/smile_person_flux_v1/flux-smile_person_flux_v1/slider_0.pt
#   federico : flux/trained_sliders/sliders/federico_v2/flux-smile_flux_v2_custom/slider_0.pt
#
# Output per ogni run:
#   edited_smile_t_s{1,1.5,2}.png   (trial_v1)
#   edited_smile_f_s{1,2,3}.png     (federico_v2)
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
F_ROOT="$FERT_REPO/flux/trained_sliders/sliders/federico_v2"

mkdir -p flux/tasks/masked_lora/logs "$CACHE_DIR"

echo "================================================================"
echo "=== PHASE 3 — smile_person × 20 run — Rebe               ==="
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

run_concept "smile_person" \
    "${T_ROOT}/smile_person_flux_v1/flux-smile_person_flux_v1/slider_0.pt" \
    "smile_t" "1 1.5 2"

run_concept "smile_person" \
    "${F_ROOT}/flux-smile_flux_v2_custom/slider_0.pt" \
    "smile_f" "1 2 3"

echo "================================================================"
echo "=== PHASE 3 smile COMPLETATA                               ==="
echo "  smile_t (trial_v1)  : s1, s1.5, s2"
echo "  smile_f (federico)  : s1, s2, s3"
echo "================================================================"
