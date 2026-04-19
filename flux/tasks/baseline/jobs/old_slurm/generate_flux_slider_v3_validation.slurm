#!/bin/bash
#SBATCH --job-name=flux_gen_v3_validation
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_slider_v3_validation.slurm
# =============================================================================
# Validazione visiva del concept slider smile_man_flux_v3_yaml_trick.
#
# Panel di confronto:
#   - 4 checkpoint: step200, step300, step400, final (step500)
#   - 4 scale:      0.0 (baseline), 0.5, 1.0, 1.5
#   - 3 prompt:     couple (test principale), man-only (sanity), woman-only (sanity)
#   - 2 seed:       42, 7 (per robustezza)
#
# Totale: 4 ckpt x 4 scale x 3 prompt x 2 seed = 96 immagini
# Tempo stimato: ~5 min Flux load + ~25 min generazioni (15s/img) = ~30 min
#
# Output organizzato con naming ordinabile:
#   ckpt{step200/step300/step400/final}__scale{0.00/0.50/1.00/1.50}__seed{42/7}__p{0/1/2}_{slug}.png
# Un `ls | sort` produce l'ordine naturale per ispezione a griglia.
#
# Skippiamo step100 perche' troppo early (warmup) per essere utile.
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

CKPT_BASE="flux/trained_sliders/training/outputs/smile_man_flux_v3_yaml_trick_rank16_xattn_alpha1"
SAVE_DIR="flux/trained_sliders/training/outputs/generations/smile_man_flux_v3_yaml_trick_validation"

mkdir -p "$SAVE_DIR"

python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs \
        "$CKPT_BASE/flux-smile_man_flux_v3_yaml_trick_step200" \
        "$CKPT_BASE/flux-smile_man_flux_v3_yaml_trick_step300" \
        "$CKPT_BASE/flux-smile_man_flux_v3_yaml_trick_step400" \
        "$CKPT_BASE/flux-smile_man_flux_v3_yaml_trick" \
    --lora_scales 0.0 0.5 1.0 1.5 \
    --prompt "a photo of a man and a woman" \
    --prompt "a photo of a man" \
    --prompt "a photo of a woman" \
    --seeds 42 7 \
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
