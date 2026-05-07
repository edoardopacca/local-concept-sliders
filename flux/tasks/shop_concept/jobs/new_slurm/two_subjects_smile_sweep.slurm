#!/bin/bash
#SBATCH --job-name=two_subjects_smile
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# two_subjects_smile_sweep.slurm — prova semplice con 2 soggetti
# =============================================================================
# Stesso setup del job sweep_smile_couple (che funzionava bene): coppia
# man + woman, prompt minimal, 512x512, 30 steps. Slider smile applicato
# a entrambi con scale opposte per testare l'asimmetria.
#
# 3 immagini:
#   1. baseline_zero    → man=0   woman=0
#   2. plus1_minus1     → man=+1  woman=-1
#   3. plus2_minus2     → man=+2  woman=-2
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p flux/tasks/shop_concept/logs

# -----------------------------------------------------------------------------
# Parametri (stessi del vecchio sweep_smile_couple che girava bene)
# -----------------------------------------------------------------------------
SLIDER="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"

PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"

TARGETS=(man woman)

SEED=42
RUN_NAME="two_subjects_smile_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 2 subjects smile sweep   ==="
echo "=============================================="
echo "prompt    : $PROMPT"
echo "targets   : ${TARGETS[*]}"
echo "seed      : $SEED"
echo "out_dir   : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local S1="$2" S2="$3"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : scales = ${S1} / ${S2}"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "$SLIDER" "$SLIDER" \
        --target_prompt "${TARGETS[0]}" "${TARGETS[1]}" \
        --lora_scales "$S1" "$S2" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

run_one "01_baseline_zero"   0   0
run_one "02_plus1_minus1"    1  -1
run_one "03_plus2_minus2"    2  -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
