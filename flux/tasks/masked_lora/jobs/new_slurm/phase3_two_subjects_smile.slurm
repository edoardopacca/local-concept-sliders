#!/bin/bash
#SBATCH --job-name=mlf_p3_two_subjects_smile
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# phase3_two_subjects_smile.slurm — Phase 3: masked edit con slider SMILE
# =============================================================================
# Prerequisiti (nella cartella RUN_ID):
#   - base.png      (da phase 1)
#   - metadata.json (da phase 1)
#   - mask.png      (da phase 2 SAM sul Mac)
#
# IMPORTANTE: modifica RUN_ID sotto con il valore che ha generato phase 1
# (è nella forma "two_subjects_masked_YYYYMMDD_HHMM").
#
# Output: edited_scale{S}.png per ogni scale, applicato SOLO dentro maschera.
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

mkdir -p flux/tasks/masked_lora/logs flux/tasks/masked_lora/_peft_cache

# -----------------------------------------------------------------------------
# CONFIG - modifica RUN_ID con quello di phase 1
# -----------------------------------------------------------------------------
RUN_ID="two_subjects_masked_20260424_1910"
SLIDER_PATH="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"
SCALES=(-2 -1 0 1 2)

echo "=============================================="
echo "=== masked_lora Flux: Phase 3 (smile)      ==="
echo "=============================================="
echo "run_id : $RUN_ID"
echo "slider : smile_flux_v5_symmetric (final)"
echo "scales : ${SCALES[*]}"
echo ""

python -u flux/tasks/masked_lora/scripts/03_masked_edit.py \
    --run_dir "flux/tasks/masked_lora/outputs/$RUN_ID" \
    --slider_path "$SLIDER_PATH" \
    --slider_scales "${SCALES[@]}" \
    --lora_fill_rank 16 \
    --cache_dir flux/tasks/masked_lora/_peft_cache \
    --device cuda \
    --mask_name mask.png

echo ""
echo "=== DONE phase3 ==="
ls -la "flux/tasks/masked_lora/outputs/$RUN_ID"
