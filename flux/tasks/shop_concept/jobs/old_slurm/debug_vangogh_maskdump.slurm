#!/bin/bash
#SBATCH --job-name=shop_vg_maskdump
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# debug_vangogh_maskdump.slurm
# =============================================================================
# Diagnostica mask: a scale=5.0 lo stile van Gogh si vede ma e' diffuso su
# tutta l'immagine (solo leggermente piu' marcato sulle montagne). La mask
# non sta localizzando.
#
# Questa run ripete la generazione con --dump_masks, che fa salvare:
#   seed<S>_scale<C>_mask_target0_seg.png   (binary mask usata per il blend)
#   seed<S>_scale<C>_mask_target0_soft.png  (sigmoid soft pre-binarize)
#
# Abbastanza per rispondere a:
#   1) la mask e' localizzata sulle montagne o e' "quasi tutto bianco"?
#   2) la soft (pre-quantile) ha un picco sensato o e' uniforme?
#
# 1 seed x 1 scale (solo quella dove l'effetto era visibile) = 1 immagine,
# ~2 min.
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

mkdir -p flux/tasks/shop_concept/logs shop_concept/outputs shop_concept/_peft_cache

echo "=== Pre-flight ==="
nvidia-smi
echo ""

python -u -m flux.tasks.shop_concept.scripts.sweep \
    --slider_path "flux/trained_sliders/training/outputs/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1/slider_0.pt" \
    --target_prompt "mountains" \
    --prompt "a natural landscape with mountains, trees, a meadow and a blue sky" \
    --seeds 42 \
    --scales 5.0 \
    --output_dir "shop_concept/outputs/debug_vangogh_maskdump" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --cache_dir "shop_concept/_peft_cache" \
    --dump_masks

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/debug_vangogh_maskdump
