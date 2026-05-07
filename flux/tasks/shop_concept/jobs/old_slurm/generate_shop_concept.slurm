#!/bin/bash
#SBATCH --job-name=shop_concept_gen
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# generate_shop_concept.slurm
# =============================================================================
# Template: genera una immagine con Flux applicando piu' Concept Sliders
# alla LoRAShop (mask dall'attenzione block 19, blending per-token).
#
# Modificare la sezione CONFIG in basso (SLIDERS / TARGETS / SCALES / PROMPT)
# per ogni esperimento. Gli slider sono accettati in formato .pt
# (LoRANetwork del training flux_slider) e convertiti on-the-fly.
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
# CONFIG (modificare per esperimento)
# -----------------------------------------------------------------------------
# Una voce per ogni slider da applicare. Stesso ordine negli array.
SLIDERS=(
    "flux/trained_sliders/training/outputs/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1/slider_0.pt"
    "flux/trained_sliders/training/outputs/age_man_flux_v1_yaml_trick_rank16_xattn_alpha1/flux-age_man_flux_v1_yaml_trick/slider_0.pt"
)
TARGETS=(
    "landscape"
    "man"
)
SCALES=(
    "1.0"
    "1.0"
)

PROMPT="a man standing in front of a landscape"
OUT_DIR="shop_concept/outputs/demo_vangogh_age"
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
echo "=== shop_concept generate ==="
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
