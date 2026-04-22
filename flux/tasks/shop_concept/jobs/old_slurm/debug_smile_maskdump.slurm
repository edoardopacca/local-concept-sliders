#!/bin/bash
#SBATCH --job-name=shop_smile_maskdump
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# debug_smile_maskdump.slurm
# =============================================================================
# Sanity check: lo smile slider su "man" (subject LoRA, non style) - verifica
# che la mask sia localizzata sulla faccia dell'uomo e che l'effetto NON si
# propaghi alla donna.
#
# Se smile funziona localizzato -> confermiamo che il problema van Gogh e'
# specifico dello stile (attention leak intrinseco per style LoRAs).
# Se smile invece si propaga anche sulla donna -> il problema e' piu' generale.
#
# 1 seed x 2 scale = 2 immagini, ~3 min.
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
    --slider_path "flux/trained_sliders/training/outputs/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric/slider_0.pt" \
    --target_prompt "man" \
    --prompt "a man and a woman facing the camera, studio portrait, neutral expression" \
    --seeds 42 \
    --scales 0.0 1.0 \
    --output_dir "shop_concept/outputs/debug_smile_maskdump" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache" \
    --dump_masks

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/debug_smile_maskdump
