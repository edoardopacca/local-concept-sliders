#!/bin/bash
#SBATCH --job-name=test_scale_tommy
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# test_scale_tommy.slurm
# =============================================================================
# Test scale sweep — 1 run per concept — (custom config) (age, curlyhair, painterly).
#
# Pesi richiesti (metti in $FERT_REPO):
#   (trial_v1 config):
#     flux/trained_sliders/sliders/trial_v1/age_person_flux_v1/flux-age_person_flux_v1/slider_0.pt
#     flux/trained_sliders/sliders/trial_v1/curlyhair_person_flux_v1/flux-curlyhair_person_flux_v1/slider_0.pt
#     flux/trained_sliders/sliders/trial_v1/painterly_flux_v1/flux-painterly_flux_v1/slider_0.pt
#   Fede (federico_v2):
#     flux/trained_sliders/sliders/federico_v2/flux-age_flux_v2_custom/slider_0.pt
#     flux/trained_sliders/sliders/federico_v2/flux-pittoric_style_flux_v2_custom/slider_0.pt
#
# Output per ogni concetto in: runs/eval_CONCEPT_01/edited_{prefix}_s{scale}.png
# curlyhair_f saltato — esiste gia'.
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
echo "=== TEST SCALE — TOMMY (age, curlyhair_t, painterly)        ==="
echo "================================================================"
nvidia-smi
echo ""

test_one() {
    local CONCEPT="$1"   # nome cartella: age_person, curlyhair, painterly
    local SLIDER="$2"    # path assoluto slider_0.pt
    local PREFIX="$3"    # prefisso output: age_t, age_f, curly_t, ...
    local SCALES="$4"    # es. "0.5 0.7 1"
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

# ---------- age ----------
test_one "age_person" "${T_ROOT}/age_person_flux_v1/flux-age_person_flux_v1/slider_0.pt" \
    "age_t" "0.5 0.7 1"

test_one "age_person" "${F_ROOT}/flux-age_flux_v2_custom/slider_0.pt" \
    "age_f" "0.5 1 1.5"

# ---------- curlyhair (solo _t, _f gia' esistente) ----------
test_one "curlyhair" "${T_ROOT}/curlyhair_person_flux_v1/flux-curlyhair_person_flux_v1/slider_0.pt" \
    "curly_t" "0.5 0.8 1"

# ---------- painterly ----------
test_one "painterly" "${T_ROOT}/painterly_flux_v1/flux-painterly_flux_v1/slider_0.pt" \
    "painterly_t" "1 1.5 2"

test_one "painterly" "${F_ROOT}/flux-pittoric_style_flux_v2_custom/slider_0.pt" \
    "painterly_f" "6 8 10"

echo "================================================================"
echo "=== TEST SCALE TOMMY COMPLETED                             ==="
echo "  runs/eval_age_person_01/   : age_t s0.5,0.7,1 | age_f s0.5,1,1.5"
echo "  runs/eval_curlyhair_01/    : curly_t s0.5,0.8,1"
echo "  runs/eval_painterly_01/    : painterly_t s1,1.5,2 | painterly_f s6,8,10"
echo "================================================================"
