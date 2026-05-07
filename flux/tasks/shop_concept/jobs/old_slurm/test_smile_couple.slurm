#!/bin/bash
#SBATCH --job-name=shop_smile_couple
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# test_smile_couple.slurm
# =============================================================================
# Test 1 di LoRAShop su Concept Sliders:
#   - scena: un uomo e una donna rivolti verso la camera
#   - slider: smile_flux_v5_symmetric (sorriso simmetrico)
#   - mask: solo sulla regione "man" (la donna non viene toccata)
# Verifica che il masking per-target faccia sorridere SOLO l'uomo,
# lasciando la donna inalterata dal Flux base.
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
SLIDERS=(
    "flux/trained_sliders/training/outputs/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"
)
TARGETS=(
    "man"
)
SCALES=(
    "1.0"
)

# Il prompt contiene "man" letteralmente ("a man" al primo posto);
# "woman" contiene "man" come substring: .index("man") ritorna la prima
# occorrenza (dentro "a man"), quindi la mask copre la regione uomo.
PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"
OUT_DIR="shop_concept/outputs/test_smile_couple"
SEED=42
STEPS=30
HEIGHT=512
WIDTH=512
GUIDANCE=3.5

mkdir -p "$OUT_DIR"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
echo "=============================================="
echo "=== shop_concept: smile on man (couple) ==="
echo "=============================================="
echo "prompt : $PROMPT"
echo "targets: ${TARGETS[*]}"
echo "scales : ${SCALES[*]}"
echo "steps  : $STEPS"
echo ""

python -u -m flux.tasks.shop_concept.scripts.generate \
    --slider_paths "${SLIDERS[@]}" \
    --target_prompt "${TARGETS[@]}" \
    --lora_scales "${SCALES[@]}" \
    --prompt "$PROMPT" \
    --output_path "$OUT_DIR/seed${SEED}.png" \
    --height "$HEIGHT" --width "$WIDTH" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$GUIDANCE" \
    --seed "$SEED" \
    --cache_dir flux/tasks/shop_concept/_peft_cache

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
