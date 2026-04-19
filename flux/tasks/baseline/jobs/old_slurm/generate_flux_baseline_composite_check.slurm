#!/bin/bash
#SBATCH --job-name=flux_baseline_composite
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_baseline_composite_check.slurm
# =============================================================================
# Pre-check Flux baseline (no LoRA, scale=0.0) sui prompt che andrebbero
# usati per v5 composite training.
#
# Domanda diagnostica: Flux *gia'* distingue "smiling man and a woman"
# (sorriso solo su uomo) da "a man and a smiling woman" (sorriso solo
# su donna)? Se si', T5 + cross-attn gia' routing-aware sui token di
# genere -> v5 training e' amplificazione di pattern esistente. Se no
# (sorridono entrambi o confusione) -> problema e' a monte, training
# v5 difficile e local sliders serviranno davvero.
#
# Panel MINIMO (no LoRA effect, solo baseline):
#   - 5 prompt:
#       P0: "a smiling man and a woman"             <- test principale
#       P1: "a smiling man and a woman standing"    <- variante test
#       P2: "a smiling man and a woman together"    <- variante test
#       P3: "a man and a woman"                     <- neutral reference
#       P4: "a man and a smiling woman"             <- test simmetria
#   - 1 ckpt dummy (v4 con scale=0.0 -> LoRA disabled)
#   - 1 scale: 0.0
#   - 1 seed: 42
#
# Totale: 5 immagini
# Tempo atteso: ~5 min load + ~2 min gen = ~7 min
#
# Il checkpoint v4 e' usato solo come "contenitore" per caricare la
# pipeline, ma scale=0.0 significa LoRA spenta -> output = pure Flux.
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

echo "=== Pre-flight ==="
nvidia-smi
echo ""

CKPT_BASE="flux/trained_sliders/training/outputs/smile_man_flux_v4_preserve_boost_rank16_xattn_alpha1"
SAVE_DIR="flux/trained_sliders/training/outputs/generations/baseline_composite_check"

mkdir -p "$SAVE_DIR"

python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs \
        "$CKPT_BASE/flux-smile_man_flux_v4_preserve_boost" \
    --lora_scales 0.0 \
    --prompt "a smiling man and a woman" \
    --prompt "a smiling man and a woman standing" \
    --prompt "a smiling man and a woman together" \
    --prompt "a man and a woman" \
    --prompt "a man and a smiling woman" \
    --seeds 42 \
    --save_dir "$SAVE_DIR" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

echo ""
echo "=== GENERAZIONI COMPLETATE ==="
ls -la "$SAVE_DIR" | head -30
echo ""
echo "Totale PNG generati:"
ls "$SAVE_DIR"/*.png 2>/dev/null | wc -l
