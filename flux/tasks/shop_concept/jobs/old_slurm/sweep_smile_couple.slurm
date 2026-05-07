#!/bin/bash
#SBATCH --job-name=shop_smile_sweep
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# sweep_smile_couple.slurm
# =============================================================================
# Sweep 4 scale x 2 seed dello slider "smile symmetric" sulla coppia
# (mask solo su "man"). 8 PNG totali in
# shop_concept/outputs/sweep_smile_couple/seed<S>_scale<C>.png
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
SLIDER="flux/trained_sliders/training/outputs/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"
TARGET="man"
PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"
OUT_DIR="shop_concept/outputs/sweep_smile_couple"

SEEDS=(42 123)
SCALES=(0.0 0.3 0.7 1.0)

STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "=== shop_concept: SWEEP smile on man ==="
echo "=============================================="
echo "prompt : $PROMPT"
echo "target : $TARGET"
echo "seeds  : ${SEEDS[*]}"
echo "scales : ${SCALES[*]}"
echo "steps  : $STEPS"
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
            --seed "$SEED" \
            --cache_dir flux/tasks/shop_concept/_peft_cache
    done
done

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
