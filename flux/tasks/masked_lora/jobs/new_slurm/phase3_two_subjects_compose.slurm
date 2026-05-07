#!/bin/bash
#SBATCH --job-name=mlf_p3_two_subjects_compose
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# phase3_two_subjects_compose.slurm
# =============================================================================
# Phase 3 multi-mask multi-LoRA: 2 maschere (man, woman), per ogni
# regione si compongono additivamente piu' slider (paper-style).
#
# Config esempio:
#   * Maschere SAM (gia' nella run_dir):
#       mask_man.png    -> regione del soggetto uomo
#       mask_woman.png  -> regione della soggetta donna
#   * Sliders + mapping:
#       slider 0 (smile)  -> mask 0 (man)
#       slider 1 (age)    -> mask 0 (man)
#       slider 2 (smile)  -> mask 1 (woman)   [stesso file, scale negativa]
#       slider 3 (age)    -> mask 1 (woman)   [stesso file, scale negativa]
#   * Scale (smile_man, age_man, smile_woman, age_woman):
#       1.5  1.5  -1.5  -1.5
#     -> uomo sorride/ringiovanisce, donna corrucciata/invecchia.
#
# Costo: 1 forward base + 2 forward per-mask = 3x compute per step.
# 30 step ~3 min su A100. Load Flux ~5-7 min. Totale ~10 min.
#
# PRIMA DI LANCIARE:
#   1. Aggiorna RUN_ID con quello della tua phase 1 (cartella outputs/<...>).
#   2. Sul Mac, prepara le 2 maschere SAM:
#        python mask_SAM/segment_with_sam.py --run_dir <run_dir> \
#          --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
#          --image_name base.png --mode interactive --output_name mask_man.png
#        python mask_SAM/segment_with_sam.py --run_dir <run_dir> \
#          --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
#          --image_name base.png --mode interactive --output_name mask_woman.png
#   3. Push: ./tools/push_to_hpc.sh   (oppure aggiungi i PNG a git e push)
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
# CONFIG
# -----------------------------------------------------------------------------
RUN_ID="two_subjects_masked_20260424_1910"   # <-- aggiorna se cambi base

SLIDER_SMILE="flux/trained_sliders/sliders/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt"
SLIDER_AGE="flux/trained_sliders/sliders/age_flux_v1_symmetric_rank16_xattn_alpha1/flux-age_flux_v1_symmetric/slider_0.pt"

# 2 maschere disgiunte
MASK_NAMES=(mask_man.png mask_woman.png)

# 4 slider PEFT (smile e age caricati 2 volte, una per regione)
SLIDER_PATHS=("$SLIDER_SMILE" "$SLIDER_AGE" "$SLIDER_SMILE" "$SLIDER_AGE")
SLIDER_TO_MASK=(0 0 1 1)

# Scale (smile_man, age_man, smile_woman, age_woman)
SLIDER_SCALES=(1.5 1.5 -1.5 -1.5)

echo "================================================="
echo "=== masked_lora Flux: Phase 3 multi-compose   ==="
echo "================================================="
echo "run_id            : $RUN_ID"
echo "masks             : ${MASK_NAMES[*]}"
echo "slider paths      : ${SLIDER_PATHS[*]}"
echo "slider->mask      : ${SLIDER_TO_MASK[*]}"
echo "slider scales     : ${SLIDER_SCALES[*]}"
echo ""

python -u flux/tasks/masked_lora/scripts/03_masked_edit.py \
    --run_dir "flux/tasks/masked_lora/outputs/$RUN_ID" \
    --slider_paths "${SLIDER_PATHS[@]}" \
    --mask_names "${MASK_NAMES[@]}" \
    --slider_to_mask "${SLIDER_TO_MASK[@]}" \
    --slider_scales "${SLIDER_SCALES[@]}" \
    --lora_fill_rank 16 \
    --cache_dir flux/tasks/masked_lora/_peft_cache \
    --device cuda \
    --output_name edited_compose_smile_age.png

echo ""
echo "=== DONE phase3 multi-compose ==="
ls -la "flux/tasks/masked_lora/outputs/$RUN_ID"
