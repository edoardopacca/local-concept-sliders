#!/bin/bash
#SBATCH --job-name=flux_gen_age_vs_eyes_3seeds
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_age_vs_eyes_3seeds_camera.slurm
# =============================================================================
# Panel di confronto diretto age_man vs eyes_man al checkpoint finale.
# Solo ckpt final (step 500) per entrambi gli slider, multi-seed per
# controllare che i pattern osservati siano robusti a variazione seed e
# non artefatti del seed 42.
#
# Prompt "looking at camera" forzano frontal-face + gaze diretto ->
# massima visibilita' sia di feature d'eta' (rughe, capelli, pelle) sia
# di occhi (centrali, ben definiti). Piu' diagnostici di "a photo of a
# man" generico dove la posa puo' oscurare il volto.
#
# Panel per OGNI slider (age e eyes):
#   - 1 ckpt: final (step500)
#   - 4 scale: 0.0, 0.5, 1.0, 1.5
#   - 3 prompt: man / woman / couple (tutti "looking at camera")
#   - 3 seed: 42, 123, 7
#
# Totale per slider: 4 * 3 * 3 = 36 immagini
# Totale job: 72 immagini = ~48 min gen + 2*7 min load = ~1h10
#
# Domande diagnostiche:
#   Q1 (consistency): il pattern osservato con seed 42 su age (donna
#      giovane in isolamento, leakage solo in couple @ scale 1.5) e'
#      robusto su seed 123 e 7? Se solo seed 42, artefatto.
#   Q2 (attribute spectrum): eyes leakka MENO di age in couple @ scale
#      1.5? Se si', conferma lo spectrum smile>age>eyes. Se no, problema
#      e' structural.
#   Q3 (baseline bias): Flux prior "man alone" -> anziano e' confermato
#      su multi-seed o seed-specific?
#
# NB: i due python call vanno in sequenza (stessa pipeline Flux caricata
# due volte: ~14 min overhead totale di load ma inevitabile perche' il
# ckpt_short cablato nel filename sarebbe "final" per entrambi -> se li
# mettiamo nello stesso save_dir si sovrascrivono. Save_dir separati.
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
EYES_CKPT="flux/trained_sliders/training/outputs/eyes_man_flux_v1_yaml_trick_rank16_xattn_alpha1/flux-eyes_man_flux_v1_yaml_trick"

SAVE_AGE="flux/trained_sliders/training/outputs/generations/age_vs_eyes_3seeds_camera/age"
SAVE_EYES="flux/trained_sliders/training/outputs/generations/age_vs_eyes_3seeds_camera/eyes"

mkdir -p "$SAVE_AGE" "$SAVE_EYES"

# -----------------------------------------------------------------------------
# 1/2: AGE slider
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [1/2] AGE slider @ final, 3 seeds ==="
echo "=============================================="
python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs "$AGE_CKPT" \
    --lora_scales 0.0 0.5 1.0 1.5 \
    --prompt "a photo of a man looking at camera" \
    --prompt "a photo of a woman looking at camera" \
    --prompt "a photo of a man and a woman looking at camera" \
    --seeds 42 123 7 \
    --save_dir "$SAVE_AGE" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

echo ""
echo "=== [AGE] generate complete ==="
ls "$SAVE_AGE"/*.png 2>/dev/null | wc -l
echo ""

# -----------------------------------------------------------------------------
# 2/2: EYES slider
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "=== [2/2] EYES slider @ final, 3 seeds ==="
echo "=============================================="
python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs "$EYES_CKPT" \
    --lora_scales 0.0 0.5 1.0 1.5 \
    --prompt "a photo of a man looking at camera" \
    --prompt "a photo of a woman looking at camera" \
    --prompt "a photo of a man and a woman looking at camera" \
    --seeds 42 123 7 \
    --save_dir "$SAVE_EYES" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

echo ""
echo "=== [EYES] generate complete ==="
ls "$SAVE_EYES"/*.png 2>/dev/null | wc -l

echo ""
echo "=== GENERAZIONI COMPLETATE ==="
echo "AGE  -> $SAVE_AGE  ($(ls $SAVE_AGE/*.png 2>/dev/null | wc -l) png)"
echo "EYES -> $SAVE_EYES ($(ls $SAVE_EYES/*.png 2>/dev/null | wc -l) png)"
