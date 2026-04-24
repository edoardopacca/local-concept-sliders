#!/bin/bash
#SBATCH --job-name=two_subj_smile_age_on_man
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# two_subjects_smile_age_on_man.slurm — 2 slider DIVERSI sullo STESSO target
# =============================================================================
# Entrambi gli slider (smile + age) vengono applicati al soggetto "man".
# La donna non ha slider e resta baseline.
#
# Test di composizione multi-slider sullo stesso target — funziona
# ripetendo lo stesso target_prompt con slider_paths diversi.
#
# 3 immagini, scale (smile_man, age_man):
#   01_scale_0_0.png    → smile=0  age=0   (baseline, tutto OFF)
#   02_scale_1_-1.png   → smile=1  age=-1  (sorride e ringiovanisce lieve)
#   03_scale_2_-2.png   → smile=2  age=-2  (estremi: sorriso ampio + molto giovane)
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
# Parametri
# -----------------------------------------------------------------------------
SLIDER_SMILE="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"
SLIDER_AGE="flux/trained_sliders/sliders/age_flux_v1_symmetric_rank16_xattn_alpha1/flux-age_flux_v1_symmetric/slider_0.pt"

PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"

# Entrambi gli slider vanno sul SAME target = "man"
# (la donna non compare nei target → resta baseline)
SLIDERS=("$SLIDER_SMILE" "$SLIDER_AGE")
TARGETS=(man man)

SEED=42
RUN_NAME="two_subjects_smile_age_on_man_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
EDIT_START=8
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 2 slider su MAN          ==="
echo "=============================================="
echo "prompt      : $PROMPT"
echo "man slider1 : smile"
echo "man slider2 : age"
echo "woman       : (nessuno slider, baseline)"
echo "seed        : $SEED"
echo "out_dir     : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local S_SMILE="$2" S_AGE="$3"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : smile=${S_SMILE}  age=${S_AGE}"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "${SLIDERS[@]}" \
        --target_prompt "${TARGETS[@]}" \
        --lora_scales "$S_SMILE" "$S_AGE" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --edit_start_step "$EDIT_START" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

run_one "01_scale_0_0"    0    0
run_one "02_scale_1_-1"   1   -1
run_one "03_scale_2_-2"   2   -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
