#!/bin/bash
#SBATCH --job-name=flux_gen_quickcheck
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_quickcheck_smile_vangogh.slurm
# =============================================================================
# Quick sanity-check dei due sliders appena addestrati (smile_v5_symmetric,
# vangogh_v1) + confronto con l'ultimo smile asimmetrico (v4_preserve_boost).
#
# Panel minimale (solo 1 seed per velocita', focus su direction-working
# piu' che robustness):
#   [1/3] smile v5 symmetric : 3 scale x 3 prompt man/woman/couple camera
#   [2/3] smile v4 asymmetric : 3 scale x 3 prompt man/woman/couple camera
#   [3/3] vangogh v1 style    : 3 scale x 2 prompt natural/city landscape
#
# Totale: 9 + 9 + 6 = 24 immagini. ~8 min gen + 3*7 min load = ~30 min.
#
# Tre call Python separate -> tre save_dir separate. Serve perche' sia
# smile_v5 che smile_v4 producono ckpt_short="final" (no "_step" nel nome)
# e andrebbero in collisione se salvati insieme.
#
# Confronti chiave:
#   - v5 sym vs v4 asym @ scale 1.0, woman isolated:
#       v5 -> donna sorride (expected, entrambi enhance in training)
#       v4 -> donna NON dovrebbe sorridere (trained solo su man enhance)
#     Se v4 la fa sorridere anyway -> leakage confermato.
#   - vangogh @ scale 1.0: applicazione stile pulita su entrambi i prompt?
#     Style slider come baseline facile per lo spectrum.
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

SMILE_V5_CKPT="flux/trained_sliders/training/outputs/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric"
SMILE_V4_CKPT="flux/trained_sliders/training/outputs/smile_man_flux_v4_preserve_boost_rank16_xattn_alpha1/flux-smile_man_flux_v4_preserve_boost"
VANGOGH_CKPT="flux/trained_sliders/training/outputs/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1"

SAVE_BASE="flux/trained_sliders/training/outputs/generations/quickcheck_smile_vangogh"
SAVE_V5="$SAVE_BASE/smile_v5_symmetric"
SAVE_V4="$SAVE_BASE/smile_v4_asymmetric"
SAVE_VG="$SAVE_BASE/vangogh_v1"

mkdir -p "$SAVE_V5" "$SAVE_V4" "$SAVE_VG"

# -----------------------------------------------------------------------------
# [1/3] smile v5 symmetric
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [1/3] smile v5 SYMMETRIC ==="
echo "=============================================="
python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs "$SMILE_V5_CKPT" \
    --lora_scales 0.0 0.5 1.0 \
    --prompt "a photo of a man looking at camera" \
    --prompt "a photo of a woman looking at camera" \
    --prompt "a photo of a man and a woman looking at camera" \
    --seeds 42 \
    --save_dir "$SAVE_V5" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

# -----------------------------------------------------------------------------
# [2/3] smile v4 asymmetric (last non-symmetric smile)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [2/3] smile v4 ASYMMETRIC (preserve_boost) ==="
echo "=============================================="
python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs "$SMILE_V4_CKPT" \
    --lora_scales 0.0 0.5 1.0 \
    --prompt "a photo of a man looking at camera" \
    --prompt "a photo of a woman looking at camera" \
    --prompt "a photo of a man and a woman looking at camera" \
    --seeds 42 \
    --save_dir "$SAVE_V4" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

# -----------------------------------------------------------------------------
# [3/3] vangogh v1 style
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [3/3] vangogh v1 STYLE ==="
echo "=============================================="
python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs "$VANGOGH_CKPT" \
    --lora_scales 0.0 0.5 1.0 \
    --prompt "a photo of a natural landscape" \
    --prompt "a photo of a city landscape" \
    --seeds 42 \
    --save_dir "$SAVE_VG" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

echo ""
echo "=== TUTTE LE GENERAZIONI COMPLETATE ==="
echo "smile v5 symmetric  : $(ls $SAVE_V5/*.png 2>/dev/null | wc -l) png -> $SAVE_V5"
echo "smile v4 asymmetric : $(ls $SAVE_V4/*.png 2>/dev/null | wc -l) png -> $SAVE_V4"
echo "vangogh v1          : $(ls $SAVE_VG/*.png 2>/dev/null | wc -l) png -> $SAVE_VG"
