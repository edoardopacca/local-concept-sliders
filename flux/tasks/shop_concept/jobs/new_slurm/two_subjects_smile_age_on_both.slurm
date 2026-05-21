#!/bin/bash
#SBATCH --job-name=two_subj_smile_age_on_both
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# 2 sliders (smile, age) applied to 2 subjects (man, woman). Compositional
# aggregation realised by loading every slider twice as a distinct PEFT
# adapter (once per region), so the same concept (e.g. smile) can take a
# different sign or intensity on each subject (e.g. +2 on the man, -2 on
# the woman).
#
#   PEFT adapters (load order):                       slider_to_target:
#     0 = smile (on man)                                0
#     1 = age   (on man)                                0
#     2 = smile (on woman)                              1
#     3 = age   (on woman)                              1
#
#   target_prompt:  man  woman
#                    0     1
#
# Inside each region the 2 sliders are SUMMED additively:
#   region_man   <- baseline + s_smile_man  * Delta_smile + s_age_man  * Delta_age
#   region_woman <- baseline + s_smile_wom  * Delta_smile + s_age_wom  * Delta_age
#
# 3 runs (s_smile_man, s_age_man, s_smile_woman, s_age_woman):
#   01_all_zero.png         -> 0  0  0  0    (clean baseline, no LoRA)
#   02_pos_man_neg_woman.png-> 1  1 -1 -1    (man +smile +age, woman -smile -age)
#   03_extreme.png          -> 2  2 -2 -2    (extremes)
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

# 2 regions.
TARGETS=(man woman)

# 4 PEFT adapters: smile and age loaded twice each (once per region).
# Order matches slider_to_target: [smile@man, age@man, smile@woman, age@woman].
SLIDERS=("$SLIDER_SMILE" "$SLIDER_AGE" "$SLIDER_SMILE" "$SLIDER_AGE")
SLIDER_TO_TARGET=(0 0 1 1)

SEED=42
RUN_NAME="two_subjects_smile_age_on_both_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

echo "================================================="
echo "=== shop_concept: 2 composed sliders x 2 targets ==="
echo "================================================="
echo "prompt          : $PROMPT"
echo "targets         : ${TARGETS[*]}"
echo "slider PEFT     : smile@man age@man smile@woman age@woman"
echo "slider->target  : ${SLIDER_TO_TARGET[*]}"
echo "seed            : $SEED"
echo "out_dir         : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local SM_M="$2" AG_M="$3" SM_W="$4" AG_W="$5"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME"
    echo "  man  : smile=${SM_M}  age=${AG_M}"
    echo "  woman: smile=${SM_W}  age=${AG_W}"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "${SLIDERS[@]}" \
        --target_prompt "${TARGETS[@]}" \
        --slider_to_target "${SLIDER_TO_TARGET[@]}" \
        --lora_scales "$SM_M" "$AG_M" "$SM_W" "$AG_W" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

run_one "01_all_zero"           0    0    0    0
run_one "02_pos_man_neg_woman"  1    1   -1   -1
run_one "03_extreme"            2    2   -2   -2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
