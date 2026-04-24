#!/bin/bash
#SBATCH --job-name=two_subjects_smile_age
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# two_subjects_smile_age_sweep.slurm — 2 slider diversi, 2 soggetti
# =============================================================================
# Stessa scena man + woman del sweep smile, ma con 2 slider differenti:
#   - slider SMILE  applicato all'UOMO   (positivo = sorride)
#   - slider AGE    applicato alla DONNA (negativo = ringiovanisce)
#
# 3 immagini:
#   01_scale_0_0.png   → man_smile=0  woman_age=0   (entrambi OFF, baseline)
#   02_scale_1_-1.png  → man_smile=1  woman_age=-1  (man sorride, woman più giovane)
#   03_scale_2_-2.png  → man_smile=2  woman_age=-2  (estremi)
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

# Target: (soggetto, slider applicato a quel soggetto)
#   man   → slider SMILE
#   woman → slider AGE
TARGETS=(man woman)

SEED=42
RUN_NAME="two_subjects_smile_age_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
EDIT_START=8
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 2 subjects smile+age     ==="
echo "=============================================="
echo "prompt      : $PROMPT"
echo "man slider  : smile (positivo = sorride)"
echo "woman slider: age   (negativo = ringiovanisce)"
echo "seed        : $SEED"
echo "out_dir     : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local S_SMILE_MAN="$2" S_AGE_WOMAN="$3"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME : man_smile=${S_SMILE_MAN}  woman_age=${S_AGE_WOMAN}"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "$SLIDER_SMILE" "$SLIDER_AGE" \
        --target_prompt "${TARGETS[0]}" "${TARGETS[1]}" \
        --lora_scales "$S_SMILE_MAN" "$S_AGE_WOMAN" \
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
