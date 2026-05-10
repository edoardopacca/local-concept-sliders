#!/bin/bash
#SBATCH --job-name=phase3_fed_v2
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_phase3_federico_v2.slurm
# =============================================================================
# Phase 3 masked LoRA — 5 concept × 20 run con i pesi di Federico (v2_custom).
# Night_sky escluso (nessuna maschera disponibile).
#
# CyberDuck: Sliders-flux/ → $FERT_REPO/flux/trained_sliders/sliders/federico_v2/
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
S_ROOT="$FERT_REPO/flux/trained_sliders/sliders/federico_v2"

mkdir -p flux/tasks/masked_lora/logs "$CACHE_DIR"

echo "================================================================"
echo "=== PHASE 3 — federico_v2 — 5 concept × 20 run            ==="
echo "================================================================"
nvidia-smi
echo ""

run_concept() {
    local CONCEPT="$1"
    local SLIDER="$2"
    local PREFIX="$3"
    local SCALES="$4"

    echo "======== ${CONCEPT} ========"
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
            DST="${RUN_DIR}/edited_${PREFIX}_s${SCALE}.png"
            [ -f "$SRC" ] && mv "$SRC" "$DST"
        done
        echo "  [ok] eval_${CONCEPT}_${RUN_NUM}"
    done
    echo ""
}

run_concept "age_person"   "${S_ROOT}/flux-age_flux_v2_custom/slider_0.pt"            "age"        "0.5 1.0 1.5"
run_concept "curlyhair"    "${S_ROOT}/flux-curly_hair_flux_v2_custom/slider_0.pt"     "curly"      "0.5 1.0 1.5"
run_concept "furlength"    "${S_ROOT}/flux-fur_flux_v2_custom/slider_0.pt"            "furlength"  "0.5 1.0 1.5"
run_concept "painterly"    "${S_ROOT}/flux-pittoric_style_flux_v2_custom/slider_0.pt" "painterly"  "2 3 4"
run_concept "smile_person" "${S_ROOT}/flux-smile_flux_v2_custom/slider_0.pt"          "smile"      "0.5 1.0 1.5"
run_concept "daynight"     "${S_ROOT}/flux-night_sky_flux_v2_custom/slider_0.pt"      "daynight"   "3 4 5"

echo "================================================================"
echo "=== PHASE 3 federico_v2 COMPLETATA                         ==="
echo "Per le metriche:"
for C in age_person curlyhair furlength painterly smile_person daynight; do
    echo "  python metrics/eval_masked.py --concept $C --runs_root ${RUNS_ROOT}"
done
echo "================================================================"
