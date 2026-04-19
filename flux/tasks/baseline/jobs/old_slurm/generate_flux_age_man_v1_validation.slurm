#!/bin/bash
#SBATCH --job-name=flux_gen_age_man_v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_age_man_v1_validation.slurm
# =============================================================================
# Validazione age_man_flux_v1_yaml_trick.
#
# Test diagnostico: stesso protocollo di v4 validation (3 prompt
# critici, 1 seed, 4 scale) per comparazione diretta con smile.
#
# Panel MINIMAL:
#   - 1 ckpt: final (step500)
#   - 4 scale: 0.0, 0.5, 1.0, 1.5
#   - 3 prompt: couple, man-only, woman-only (critico per leakage)
#   - 1 seed: 42
#
# Totale: 12 immagini, ~8 min.
#
# Ipotesi da verificare:
#   H1 (attribute-specific): age NON leakka su donna isolata -> smile
#     era special case (diffuso semanticamente). Age funziona perche'
#     ha feature facciali stabili (rughe, capelli grigi) piu' localizzate.
#   H2 (structural): age leakka come smile -> problema e' Flux che non
#     fa binding strutturale per nessun adjective.
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
SAVE_DIR="flux/trained_sliders/training/outputs/generations/age_man_flux_v1_validation"

mkdir -p "$SAVE_DIR"

python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs \
        "$CKPT_BASE/flux-age_man_flux_v1_yaml_trick" \
    --lora_scales 0.0 0.5 1.0 1.5 \
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
ls -la "$SAVE_DIR" | head -30
echo ""
echo "Totale PNG generati:"
ls "$SAVE_DIR"/*.png 2>/dev/null | wc -l
