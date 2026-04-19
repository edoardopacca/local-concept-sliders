#!/bin/bash
#SBATCH --job-name=flux_gen_sky_day_night_validation
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_slider_sky_day_night_validation.slurm
# =============================================================================
# Validazione visiva del concept slider sky_day_night_flux_v1.
#
# Caratteristica chiave: slider BIDIREZIONALE.
#   scale > 0 -> direzione giorno (cielo luminoso, blu)
#   scale < 0 -> direzione notte (cielo scuro, stelle)
#   scale = 0 -> baseline Flux (LoRA disabled)
#
# Quindi valutiamo il range completo negativi + positivi.
#
# Panel MINIMAL (solo final ckpt, 1 seed):
#   - 1 checkpoint: final (step500)
#   - 5 scale:      -1.0, -0.5, 0.0, 0.5, 1.0
#   - 2 prompt:     landscape, city
#   - 1 seed:       42
#
# Totale: 1 x 5 x 2 x 1 = 10 immagini
# Tempo stimato: ~5 min load + ~3 min generazioni = ~8 min
#
# Primo check veloce: se a scale -1.0 si vede chiaro cielo notturno
# e a scale +1.0 cielo giorno luminoso su entrambi i contesti,
# lo slider funziona. Per range esteso (-1.5, +1.5) e multi-seed
# si fa un panel piu' profondo in un secondo momento.
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

CKPT_BASE="flux/trained_sliders/training/outputs/sky_day_night_flux_v1_rank16_xattn_alpha1"
SAVE_DIR="flux/trained_sliders/training/outputs/generations/sky_day_night_flux_v1_validation"

mkdir -p "$SAVE_DIR"

python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs \
        "$CKPT_BASE/flux-sky_day_night_flux_v1" \
    --lora_scales -1.0 -0.5 0.0 0.5 1.0 \
    --prompt "a photo of a landscape" \
    --prompt "a photo of a city" \
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
