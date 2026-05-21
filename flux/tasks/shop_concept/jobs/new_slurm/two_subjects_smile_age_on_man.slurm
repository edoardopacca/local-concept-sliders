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
# 2 composed sliders on a single target. Compositional aggregation realised
# via PEFT multi-adapter: two sliders active at the same time on the same
# LoRA layer, with their deltas summed additively inside the region matched
# by the target "man".
#
#   --target_prompt man                    # one masked region
#   --slider_paths smile.pt age.pt         # 2 sliders
#   --slider_to_target 0 0                 # both -> target "man"
#   --lora_scales s_smile s_age            # one scale per slider
#
# The woman is not a target and therefore stays on the baseline (no LoRA).
#
# 3 runs, scales (smile, age):
#   01_scale_0_0.png    -> smile=0  age=0   (baseline, everything OFF)
#   02_scale_1_-1.png   -> smile=1  age=-1  (light smile + light rejuvenation)
#   03_scale_2_-2.png   -> smile=2  age=-2  (extremes: broad smile + young)
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
# Parameters
# -----------------------------------------------------------------------------
SLIDER_SMILE="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"
SLIDER_AGE="flux/trained_sliders/sliders/age_flux_v1_symmetric_rank16_xattn_alpha1/flux-age_flux_v1_symmetric/slider_0.pt"

PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"

# Single region (man); both sliders mapped to that region.
TARGET="man"
SLIDERS=("$SLIDER_SMILE" "$SLIDER_AGE")
SLIDER_TO_TARGET=(0 0)

SEED=42
RUN_NAME="two_subjects_smile_age_on_man_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "=============================================="
echo "=== shop_concept: 2 composed sliders on MAN ==="
echo "=============================================="
echo "prompt          : $PROMPT"
echo "target          : $TARGET"
echo "slider 0 (smile): $SLIDER_SMILE"
echo "slider 1 (age)  : $SLIDER_AGE"
echo "slider->target  : ${SLIDER_TO_TARGET[*]} (both -> man)"
echo "woman           : (no slider, baseline)"
echo "seed            : $SEED"
echo "out_dir         : $OUT_DIR"
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
        --target_prompt "$TARGET" \
        --slider_to_target "${SLIDER_TO_TARGET[@]}" \
        --lora_scales "$S_SMILE" "$S_AGE" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

run_one "01_scale_0_0"    0    0
run_one "02_scale_1_-1"   1   -1
run_one "03_scale_2_-2"   2   -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
