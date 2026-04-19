#!/bin/bash
#SBATCH --job-name=flux_gen_v4_validation
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_slider_v4_validation.slurm
# =============================================================================
# Validazione visiva del concept slider smile_man_flux_v4_preserve_boost.
#
# Scopo diagnostico: confrontare leakage su donna isolata tra v3 e v4.
# v3 (5+5 entries, ratio 50:50) mostrava donna-sola sorridere di piu'
# a scale=1.0 (amplificazione + exposure bias). v4 (5+15, ratio 25:75,
# con 4 entries anti-amplificazione) dovrebbe ridurre o eliminare
# l'effetto. Test principale: donna isolata @ scale 0.0 vs @ scale 1.0.
#
# Panel MINIMAL (solo final ckpt, 1 seed):
#   - 1 checkpoint: final (step500)
#   - 4 scale:      0.0 (baseline), 0.5, 1.0, 1.5
#   - 3 prompt:     couple, man-only, woman-only (critico)
#   - 1 seed:       42
#
# Totale: 1 x 4 x 3 x 1 = 12 immagini
# Tempo stimato: ~5 min load + ~3 min generazioni = ~8 min
#
# Questo e' il primo check rapido: se la donna isolata a scale=1.0
# resta uguale a scale=0.0 -> v4 ha funzionato, facciamo panel piu'
# profondo per confermare. Se invece sorride piu' come in v3 -> YAML
# trick confermato strutturalmente insufficiente, si passa a local.
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

CKPT_BASE="flux/trained_sliders/training/outputs/smile_man_flux_v4_preserve_boost_rank16_xattn_alpha1"
SAVE_DIR="flux/trained_sliders/training/outputs/generations/smile_man_flux_v4_preserve_boost_validation"

mkdir -p "$SAVE_DIR"

python -u flux/tasks/baseline/generate_flux_slider.py \
    --lora_dirs \
        "$CKPT_BASE/flux-smile_man_flux_v4_preserve_boost" \
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
