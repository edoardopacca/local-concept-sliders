#!/bin/bash
#SBATCH --job-name=flux_smile_man_v4_preserve_boost
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_smile_man_flux_v4_preserve_boost.slurm
# =============================================================================
# Iterazione su v3_yaml_trick dopo la diagnosi del leakage nella validation
# (woman-only passa da sorriso normale a bocca aperta a scale=1).
#
# Single variable change vs v3:
#   WOMAN entries: 5 -> 15 (coprono phrasing, età, etnia, attributi,
#                  espressione). Man entries invariate (5).
#
# Iperparametri IDENTICI a v3 per confronto pulito:
#   - 500 max_train_steps
#   - rank 16, xattn, alpha 1
#   - lr 0.002, warmup 200
#   - eta per-entry dal YAML (2 man / 1 woman)
#   - save_every 100 -> checkpoint a 100/200/300/400/final
#
# Stima tempo:
#   - v3 (10 entries, dedup medio ~2 fwd/step) = 1h01m a 500 step
#   - v4 (20 entries): stessa dinamica di dedup (man-like ancora 3-4
#     unique fwd, woman-like sempre 1). Cambia solo la frequenza di
#     hit dedup per entries woman (più varianti = cache hit identico
#     ma lookup stringa più ampio, trascurabile).
#   - Expected: ~1h-1h10m. Walltime 4h abbondante.
#
# Ratio enhance:preservation passa da 50:50 (v3) a 25:75 (v4):
#   - 5 man × ~25 step = 125 step enhance (vs 250 in v3)
#   - 15 woman × ~25 step = 375 step preserve (vs 250 in v3)
# La direzione smile resta appresa (5 entries x 25 step sono
# abbondantemente sufficienti per far scendere la man-loss; su v3
# l'intera dinamica di learning era già completa a step 200-300).
# Il surplus di step preservation dovrebbe ridurre il leakage
# globale, in particolare sull'amplificazione di espressioni già
# positive grazie alle 4 entries "espressione preservata".
#
# Output layout identico a v3:
#   outputs/smile_man_flux_v4_preserve_boost_rank16_xattn_alpha1/
#     slider_0.pt (final)
#     flux-smile_man_flux_v4_preserve_boost_step{100,200,300,400}/
#       slider_0.pt
#     smile_man_flux_v4_preserve_boost_losses.npy
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

# safety pyyaml
python -c "import yaml" 2>/dev/null || pip install pyyaml --quiet

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pre-flight ==="
nvidia-smi
echo ""

OUT_DIR="flux/trained_sliders/training/outputs/smile_man_flux_v4_preserve_boost_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name smile_man_flux_v4_preserve_boost \
    --prompts_yaml flux/trained_sliders/training/prompts/old_prompt/smile_man_flux_v4_preserve_boost.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
