#!/bin/bash
#SBATCH --job-name=shop_eyes_couple
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# test_big_eyes_couple.slurm
# =============================================================================
# Doppio test sullo slider big_eyes v1:
#
#   1) MAGNITUDE: a scale 1 era debole, a 2/3/5 poco visibile. Proviamo
#      0 / 3 / 5 / 7 per capire se serve solo scale piu' aggressiva.
#
#   2) LOCALIZZAZIONE LORASHOP: scena con una coppia (uomo + donna),
#      target_prompt="woman" -> la mask di LoRAShop dovrebbe ancorarsi
#      solo sulla donna. Se funziona: occhi grandi sulla donna, uomo
#      invariato. Se fallisce: occhi grandi anche sull'uomo o ovunque.
#
# Abilito --dump_masks per ispezionare visualmente la mask e confermare
# la localizzazione (stesso pattern del test smile precedente).
#
# Costo: 4 scale x 1 seed + mask dump = circa 10-15 min.
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
    --slider_path "flux/trained_sliders/training/outputs/big_eyes_flux_v1_symmetric_rank16_xattn_alpha1/flux-big_eyes_flux_v1_symmetric/slider_0.pt" \
    --target_prompt "woman" \
    --prompt "a portrait of a couple, a man and a woman standing side by side, studio photography, neutral expression" \
    --seeds 42 \
    --scales 0.0 3.0 5.0 7.0 \
    --output_dir "shop_concept/outputs/test_big_eyes_couple" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache" \
    --dump_masks

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/test_big_eyes_couple
