#!/bin/bash
#SBATCH --job-name=flux_sky_day_night_v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_sky_day_night_flux_v1.slurm
# =============================================================================
# Global concept slider: direzione giorno <-> notte del cielo.
#
# Setup semplificato rispetto a smile_man (NO preservation / NO YAML-trick):
#   - 10 entries tutte enhance con positive=giorno, negative=notte
#   - varianti di contesto (landscape, sky, city, beach, forest,
#     mountains, field, house, street, desert)
#   - eta=2 uniforme (direzione bidirezionale day/night)
#
# Iperparametri identici a smile_man v3/v4 (proven-stable):
#   - 500 max_train_steps
#   - rank 16, xattn, alpha 1
#   - lr 0.002, warmup 200
#   - save_every 100 -> checkpoint a 100/200/300/400/final
#
# Costo atteso:
#   - 3 forward LoRA-off unique per step (target==neutral -> 2 unique
#     fwd per quelli; positive e negative sempre distinti)
#   - ~5-6 s/step -> 500 step ~45-55 min + load ~7 min = ~1h total
#   - Walltime 4h abbondante.
#
# Output layout:
#   outputs/sky_day_night_flux_v1_rank16_xattn_alpha1/
#     slider_0.pt (final)
#     flux-sky_day_night_flux_v1_step{100,200,300,400}/slider_0.pt
#     sky_day_night_flux_v1_losses.npy
#
# Uso scale a inferenza:
#   scale > 0 -> direzione giorno (cielo luminoso)
#   scale < 0 -> direzione notte (cielo scuro, stelle)
#   scale = 0 -> baseline Flux (LoRA disabled)
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

OUT_DIR="flux/trained_sliders/training/outputs/sky_day_night_flux_v1_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name sky_day_night_flux_v1 \
    --prompts_yaml flux/trained_sliders/training/prompts/old_prompt/sky_day_night_flux_v1.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
