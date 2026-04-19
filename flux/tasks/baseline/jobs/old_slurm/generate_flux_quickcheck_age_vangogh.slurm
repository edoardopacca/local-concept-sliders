#!/bin/bash
#SBATCH --job-name=flux_gen_age_vangogh
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_quickcheck_age_vangogh.slurm
# =============================================================================
# Quick test dei due sliders attualmente disponibili su HPC:
#   - age asimmetrico (age_man_flux_v1_yaml_trick)
#   - vangogh v1 (style slider)
#
# Panel:
#   [1/2] age asimmetrico : 3 scale x 3 prompt man/woman/couple camera x seed 42 = 9 img
#   [2/2] vangogh v1      : 3 scale x 2 prompt natural/city landscape x seed 42 = 6 img
# Totale: 15 immagini, ~6 min gen + 2*7 min load = ~20 min.
#
# Prompt age: "looking at camera" (stessi del vecchio test smile asimmetrico
# e del panel age_vs_eyes, per confronto diretto).
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

AGE_CKPT="flux/trained_sliders/training/outputs/age_man_flux_v1_yaml_trick_rank16_xattn_alpha1/flux-age_man_flux_v1_yaml_trick"
VANGOGH_CKPT="flux/trained_sliders/training/outputs/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1"

SAVE_BASE="flux/trained_sliders/training/outputs/generations/quickcheck_age_vangogh"
SAVE_AGE="$SAVE_BASE/age_asymmetric"
SAVE_VG="$SAVE_BASE/vangogh_v1"

mkdir -p "$SAVE_AGE" "$SAVE_VG"

# -----------------------------------------------------------------------------
# [1/2] age asimmetrico (yaml_trick)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [1/2] AGE asimmetrico (yaml_trick) ==="
echo "=============================================="
python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs "$AGE_CKPT" \
    --lora_scales 0.0 0.5 1.0 \
    --prompt "a photo of a man looking at camera" \
    --prompt "a photo of a woman looking at camera" \
    --prompt "a photo of a man and a woman looking at camera" \
    --seeds 42 \
    --save_dir "$SAVE_AGE" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

# -----------------------------------------------------------------------------
# [2/2] vangogh v1 style
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [2/2] vangogh v1 STYLE ==="
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
echo "=== GENERAZIONI COMPLETATE ==="
echo "age asimmetrico : $(ls $SAVE_AGE/*.png 2>/dev/null | wc -l) png -> $SAVE_AGE"
echo "vangogh v1      : $(ls $SAVE_VG/*.png 2>/dev/null | wc -l) png -> $SAVE_VG"
