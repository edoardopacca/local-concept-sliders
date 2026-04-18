#!/bin/bash
#SBATCH --job-name=flux_smile_man_v2_asym
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_smile_man_flux_v2_asym.slurm
# =============================================================================
# Training slider "smile_man" su Flux.1-dev con prompt ASIMMETRICI:
# il locus (man) cambia espressione tra positive/negative, mentre la
# descrizione della woman e' IDENTICA in tutti e tre i prompt. La loss
# Concept Sliders Eq.7 rimane invariata.
#
# Idea (analogo testuale del SDXL "preserve_woman"):
#
#   target    = "a photo of a man and a woman"
#   positive  = "a photo of a smiling man, big smile, happy, and a woman"
#   negative  = "a photo of a frowning man, sad, grumpy,    and a woman"
#
# Il delta (positive - negative) contiene SOLO l'informazione "smile vs
# frown sul man" — la frase "and a woman" e' byte-identica tra positive
# e negative, quindi per la T5 i token della donna cancellano nel delta.
# La LoRA viene guidata a spostare SOLO la direzione smile/frown del man.
#
# Non richiede modifiche allo script train_flux_slider.py (i 3 arg CLI
# target/positive/negative esistono gia').
#
# Config:
#   - 500 step, rank 16, xattn, lr 0.002, warmup 200, eta 2
#   - altri hyperparams identici a v1 (che non avevamo ancora lanciato)
#   - save_every 100 per poter confrontare ckpt intermedi
#
# Tempo atteso (da smoke test 7.3 s/step): 500 * 7.3s + ~5 min load
#   = ~65-70 min. Walltime 12h per sicurezza.
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

# Output dir con tag "asym" per distinguere dalla v1 (simmetrica, non lanciata)
OUT_DIR="flux/trained_sliders/training/outputs/smile_man_flux_v2_asym_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name smile_man_flux_v2_asym \
    --target_prompt   "a photo of a man and a woman" \
    --positive_prompt "a photo of a smiling man, big smile, happy face, and a woman" \
    --negative_prompt "a photo of a frowning man, sad, grumpy face, and a woman" \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --eta 2 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
