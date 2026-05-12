#!/bin/bash
#SBATCH --job-name=test_scale_rebe
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# test_scale_rebe.slurm
# =============================================================================
# Test scale sweep — 1 run per concept — per Rebe (furlength, smile).
#
# Pesi richiesti (metti in $FERT_REPO):
#   Edoardo (trial_v1):
#     flux/trained_sliders/sliders/trial_v1/furlength_dog_flux_v1/flux-furlength_dog_flux_v1/slider_0.pt
#     flux/trained_sliders/sliders/trial_v1/smile_person_flux_v1/flux-smile_person_flux_v1/slider_0.pt
#   Fede (federico_v2):
#     flux/trained_sliders/sliders/federico_v2/flux-fur_flux_v2_custom/slider_0.pt
#     flux/trained_sliders/sliders/federico_v2/flux-smile_flux_v2_custom/slider_0.pt
#
# Output per ogni concetto in: runs/eval_CONCEPT_01/edited_{prefix}_s{scale}.png
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
echo "=== TEST SCALE — REBE (furlength, smile)                    ==="
echo "================================================================"
nvidia-smi
echo ""

test_one() {
    local CONCEPT="$1"   # nome cartella: furlength, smile_person
    local SLIDER="$2"    # path assoluto slider_0.pt
    local PREFIX="$3"    # prefisso output: furlength_t, furlength_f, smile_t, smile_f
    local SCALES="$4"
    local RUN_DIR="${RUNS_ROOT}/eval_${CONCEPT}_01"

    echo "--- ${PREFIX} (${CONCEPT}_01) scales=[${SCALES}] ---"

    if [ ! -d "$RUN_DIR" ]; then
        echo "[SKIP] dir mancante: $RUN_DIR"; echo ""; return
    fi
    if [ ! -f "${RUN_DIR}/mask_target.png" ]; then
        echo "[SKIP] mask_target.png mancante in $RUN_DIR"; echo ""; return
    fi
    if [ ! -f "$SLIDER" ]; then
        echo "[SKIP] slider mancante: $SLIDER"; echo ""; return
    fi

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
        [ -f "$SRC" ] && mv "$SRC" "$DST" && echo "  -> $DST"
    done
    echo ""
}

# ---------- furlength ----------
test_one "furlength" "${T_ROOT}/furlength_dog_flux_v1/flux-furlength_dog_flux_v1/slider_0.pt" \
    "furlength_t" "1 2 3"

test_one "furlength" "${F_ROOT}/flux-fur_flux_v2_custom/slider_0.pt" \
    "furlength_f" "2 3 5 7"

# ---------- smile ----------
test_one "smile_person" "${T_ROOT}/smile_person_flux_v1/flux-smile_person_flux_v1/slider_0.pt" \
    "smile_t" "1 1.5 2"

test_one "smile_person" "${F_ROOT}/flux-smile_flux_v2_custom/slider_0.pt" \
    "smile_f" "1 2 3"

echo "================================================================"
echo "=== TEST SCALE REBE COMPLETATO                              ==="
echo "  runs/eval_furlength_01/    : furlength_t s1,2,3 | furlength_f s2,3,5,7"
echo "  runs/eval_smile_person_01/ : smile_t s1,1.5,2   | smile_f s1,2,3"
echo "================================================================"
