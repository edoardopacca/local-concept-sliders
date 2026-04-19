#!/bin/bash
#SBATCH --job-name=flux_gen_age_man_v1_ckpt_sweep
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_age_man_v1_ckpt_sweep.slurm
# =============================================================================
# Sweep checkpoint intermedi age_man_flux_v1 per localizzare a quale step
# emerge il leakage a scale alta e quando il binding uomo-only e' piu' pulito.
#
# Validation @ final ha mostrato:
#   - donna isolata: preservation perfetta a tutte le scale (0.0-1.5)
#   - couple @ scale 1.0: aging asimmetrico OK (uomo vecchio, donna giovane)
#   - couple @ scale 1.5: leakage (entrambi vecchi)
#   - uomo isolato @ scale 0.5: NON-MONOTONO (sembra ringiovanire)
#
# Questo sweep risponde a:
#   Q1 (training dynamics): a quale step (100/200/300/400/500) emerge il
#      leakage sulla donna nel couple @ scale 1.5? E' graduale o a soglia?
#   Q2 (best ckpt): esiste un early-stop ottimale (es. step 300) che ha
#      direzione sufficiente su uomo ma leakage minore su donna composite?
#   Q3 (non-monotonia): il dip a scale 0.5 e' presente gia' a step 100 o
#      si sviluppa solo tardi (over-training)?
#
# Panel:
#   - 5 ckpt: step100, step200, step300, step400, final (step500)
#   - 2 scale: 1.0 (best case validation), 1.5 (leakage case)
#   - 3 prompt: couple, man-only, woman-only
#   - 1 seed: 42
# Totale: 5 * 2 * 3 * 1 = 30 immagini, ~20-30 min.
#
# NB: aggiungiamo scale 0.0 solo una volta (step finale) sarebbe ridondante -
# il baseline NON dipende dal ckpt (LoRA disabilitata). Lo abbiamo gia' dalla
# validation precedente, quindi non lo ripetiamo qui.
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

CKPT_BASE="flux/trained_sliders/training/outputs/age_man_flux_v1_yaml_trick_rank16_xattn_alpha1"
SAVE_DIR="flux/trained_sliders/training/outputs/generations/age_man_flux_v1_ckpt_sweep"

mkdir -p "$SAVE_DIR"

echo "=== Ckpt disponibili ==="
ls -la "$CKPT_BASE" | grep -E "flux-age_man" || true
echo ""

python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs \
        "$CKPT_BASE/flux-age_man_flux_v1_yaml_trick_step100" \
        "$CKPT_BASE/flux-age_man_flux_v1_yaml_trick_step200" \
        "$CKPT_BASE/flux-age_man_flux_v1_yaml_trick_step300" \
        "$CKPT_BASE/flux-age_man_flux_v1_yaml_trick_step400" \
        "$CKPT_BASE/flux-age_man_flux_v1_yaml_trick" \
    --lora_scales 1.0 1.5 \
    --prompt "a photo of a man and a woman" \
    --prompt "a photo of a man" \
    --prompt "a photo of a woman" \
    --seeds 42 \
    --save_dir "$SAVE_DIR" \
    --rank 16 --alpha 1.0 --train_method xattn \
    --height 512 --width 512 \
    --steps 30 --guidance_scale 3.5

echo ""
echo "=== GENERAZIONI COMPLETATE ==="
ls -la "$SAVE_DIR" | head -40
echo ""
echo "Totale PNG generati:"
ls "$SAVE_DIR"/*.png 2>/dev/null | wc -l
