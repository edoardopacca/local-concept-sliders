#!/bin/bash
#SBATCH --job-name=shop_vangogh_sweep
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# sweep_vangogh_sky.slurm
# =============================================================================
# Sweep 4 scale x 2 seed dello slider "vangogh_flux_v1" sul paesaggio
# (mask solo su "sky"). 8 PNG totali in
# shop_concept/outputs/sweep_vangogh_sky/seed<S>_scale<C>.png
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

mkdir -p flux/tasks/shop_concept/logs shop_concept/outputs shop_concept/_peft_cache

echo "=== Pre-flight ==="
nvidia-smi
echo ""

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SLIDER="flux/trained_sliders/training/outputs/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1/slider_0.pt"
TARGET="sky"
PROMPT="a natural landscape with mountains, trees, a meadow and a blue sky"
OUT_DIR="shop_concept/outputs/sweep_vangogh_sky"

SEEDS=(42 123)
SCALES=(0.0 0.3 0.7 1.0)

STEPS=30
EDIT_START=8
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "=== shop_concept: SWEEP vangogh on sky ==="
echo "=============================================="
echo "prompt : $PROMPT"
echo "target : $TARGET"
echo "seeds  : ${SEEDS[*]}"
echo "scales : ${SCALES[*]}"
echo "steps  : $STEPS   edit_start: $EDIT_START"
echo "total  : $((${#SEEDS[@]} * ${#SCALES[@]})) immagini"
echo ""

# -----------------------------------------------------------------------------
# Sweep loop
# -----------------------------------------------------------------------------
COUNT=0
for SEED in "${SEEDS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
        COUNT=$((COUNT + 1))
        OUT_PATH="$OUT_DIR/seed${SEED}_scale${SCALE}.png"
        echo "----------------------------------------------"
        echo "[$COUNT/$((${#SEEDS[@]} * ${#SCALES[@]}))] seed=$SEED scale=$SCALE"
        echo "  out: $OUT_PATH"
        echo "----------------------------------------------"

        python -u -m flux.tasks.shop_concept.scripts.generate \
            --slider_paths "$SLIDER" \
            --target_prompt "$TARGET" \
            --lora_scales "$SCALE" \
            --prompt "$PROMPT" \
            --output_path "$OUT_PATH" \
            --height "$HEIGHT" --width "$WIDTH" \
            --num_inference_steps "$STEPS" \
            --guidance_scale "$GUIDANCE" \
            --edit_start_step "$EDIT_START" \
            --seed "$SEED" \
            --cache_dir flux/tasks/shop_concept/_peft_cache
    done
done

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
