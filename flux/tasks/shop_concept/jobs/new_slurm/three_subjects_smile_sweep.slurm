#!/bin/bash
#SBATCH --job-name=three_subjects_smile
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# three_subjects_smile_sweep.slurm
# =============================================================================
# Genera 3 immagini con shop_concept (Flux + multi-LoRA per-soggetto) usando
# lo stesso slider "smile" con scale diverse per ognuno dei 3 soggetti.
#
# Prompt: scena con 3 soggetti che guardano di fronte (uomo / donna / bambino).
# Slider: smile_flux_v5_symmetric (rank=16, xattn, alpha=1).
#
# 3 immagini:
#   1. baseline_zero      → scales: man=0   woman=0   child=0   (slider OFF ovunque)
#   2. minus1_zero_plus1  → scales: man=-1  woman=0   child=+1  (uno serio, uno neutro, uno sorride)
#   3. minus2_zero_minus2 → scales: man=-2  woman=0   child=-2  (primo+terzo molto seri)
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
# Parametri condivisi
# -----------------------------------------------------------------------------
# Path al checkpoint finale dello slider (training completato).
# Per testare i checkpoint intermedi sostituisci con uno tra:
#   .../flux-smile_flux_v5_symmetric_step100/slider_0.pt   (early)
#   .../flux-smile_flux_v5_symmetric_step200/slider_0.pt
#   .../flux-smile_flux_v5_symmetric_step300/slider_0.pt
#   .../flux-smile_flux_v5_symmetric_step400/slider_0.pt   (late)
SLIDER="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"

# Prompt principale (i 3 target devono esserne sottostringhe letterali)
PROMPT="a photograph of a man, a woman, and a child standing side by side, looking at the camera, in a sunny park with trees and grass in the background, natural light, sharp focus, high detail"

# Sottostringhe del prompt che identificano i 3 soggetti
TARGETS=(man woman child)

SEED=42
RUN_NAME="three_subjects_smile_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
EDIT_START=8
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 3 subjects smile sweep ==="
echo "=============================================="
echo "prompt  : $PROMPT"
echo "targets : ${TARGETS[*]}"
echo "seed    : $SEED"
echo "out_dir : $OUT_DIR"
echo ""

# -----------------------------------------------------------------------------
# Funzione helper: lancia 1 generazione con 3 scale (una per soggetto)
# -----------------------------------------------------------------------------
run_one() {
    local NAME="$1"
    local S1="$2" S2="$3" S3="$4"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : scales = ${S1} / ${S2} / ${S3}"
    echo "  out: $OUT_PATH"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "$SLIDER" "$SLIDER" "$SLIDER" \
        --target_prompt "${TARGETS[0]}" "${TARGETS[1]}" "${TARGETS[2]}" \
        --lora_scales "$S1" "$S2" "$S3" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --edit_start_step "$EDIT_START" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

# -----------------------------------------------------------------------------
# Le 3 generazioni
# -----------------------------------------------------------------------------
run_one "01_baseline_zero"        0   0   0
run_one "02_minus1_zero_plus1"   -1   0   1
run_one "03_minus2_zero_minus2"  -2   0  -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
