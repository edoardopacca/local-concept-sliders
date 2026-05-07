#!/bin/bash
#SBATCH --job-name=three_subj_smile_compose_1024
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# three_subjects_smile_compose_1024.slurm
# =============================================================================
# 3 soggetti (man, woman, child) in scena complessa con sfondo, risoluzione
# 1024x1024. Slider smile applicato indipendentemente a ognuno con scale
# diverse per ogni soggetto.
#
# Setup:
#   * 1 file slider (smile_v5_symmetric) caricato 3 volte come adapter
#     PEFT distinti -> permette scale diverse per i 3 target.
#   * 3 target_prompt: man, woman, child (in quest'ordine, perche' "man"
#     e' sottostringa di "woman" -> ordine conta per prompt.index()).
#   * slider_to_target: 0 1 2 (1 slider per target, identity mapping).
#
# 3 run, scale (s_man, s_woman, s_child):
#   01_all_zero.png        -> 0   0   0     (baseline, tutti neutri)
#   02_mid_split.png       -> -1  0   1     (man frown, woman neutra, child smile)
#   03_extreme_split.png   -> -2  0   2     (estremi opposti su man e child)
#
# Costo: 1024^2 ~4x token rispetto 512 -> ~25-30 min totali su A100 stud.
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

PROMPT="a man, a woman, and a child sitting at a wooden table in a cozy cafe with warm lights and bookshelves in the background, looking at the camera, neutral expression, professional photo"

# Importante: ordine man, woman, child -> "man" matcha "a man," (non "woMan")
TARGETS=(man woman child)

# Stesso slider caricato 3 volte come adapter distinti
SLIDERS=("$SLIDER_SMILE" "$SLIDER_SMILE" "$SLIDER_SMILE")
SLIDER_TO_TARGET=(0 1 2)

SEED=42
RUN_NAME="three_subjects_smile_compose_1024_$(date +%Y%m%d_%H%M)"
OUT_DIR="flux/tasks/shop_concept/outputs/${RUN_NAME}"
mkdir -p "$OUT_DIR"

STEPS=30
HEIGHT=1024
WIDTH=1024
GUIDANCE=3.5

echo "=========================================================="
echo "=== shop_concept: 3 soggetti, smile per-target, 1024^2 ==="
echo "=========================================================="
echo "prompt          : $PROMPT"
echo "targets         : ${TARGETS[*]}"
echo "slider          : smile_v5 (loaded 3x as default_0, default_1, default_2)"
echo "slider->target  : ${SLIDER_TO_TARGET[*]}"
echo "seed            : $SEED"
echo "resolution      : ${HEIGHT}x${WIDTH}"
echo "out_dir         : $OUT_DIR"
echo ""

run_one() {
    local NAME="$1"
    local S_MAN="$2" S_WOMAN="$3" S_CHILD="$4"
    local OUT_PATH="$OUT_DIR/${NAME}.png"

    echo "----------------------------------------------"
    echo "[run] $NAME"
    echo "  man  : smile=${S_MAN}"
    echo "  woman: smile=${S_WOMAN}"
    echo "  child: smile=${S_CHILD}"
    echo "----------------------------------------------"

    python -u -m flux.tasks.shop_concept.scripts.generate \
        --slider_paths "${SLIDERS[@]}" \
        --target_prompt "${TARGETS[@]}" \
        --slider_to_target "${SLIDER_TO_TARGET[@]}" \
        --lora_scales "$S_MAN" "$S_WOMAN" "$S_CHILD" \
        --prompt "$PROMPT" \
        --output_path "$OUT_PATH" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --cache_dir flux/tasks/shop_concept/_peft_cache
}

run_one "01_all_zero"       0    0    0
run_one "02_mid_split"     -1    0    1
run_one "03_extreme_split" -2    0    2

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
