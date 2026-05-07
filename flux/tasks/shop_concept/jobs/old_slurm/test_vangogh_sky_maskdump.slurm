#!/bin/bash
#SBATCH --job-name=shop_vg_sky_maskdump
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# test_vangogh_sky_maskdump.slurm
# =============================================================================
# Obiettivo: visualizzare la mask che LoRAShop estrae per "sky" su un
# paesaggio naturale con cielo ben visibile, e confrontarla con il comporta-
# mento dello slider van Gogh applicato a quella mask.
#
# Serve a:
#   1) Verificare la qualita' della mask "sky" (ben localizzata? ragionevole?)
#   2) Vedere su piu' scale se lo stile resta nel cielo o leakka sulle
#      montagne/alberi (aspettativa: leakka per self-attention, come con
#      "mountains").
#   3) Avere un asset utile per la tesi: "questa e' la mask corretta,
#      eppure lo stile cola fuori".
#
# Usa sweep.py (Flux caricato una volta sola, ~3 min di load + 4 x 40s img).
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
    --target_prompt "sky" \
    --prompt "a natural landscape, a mountain range under a vast sky, photorealistic, golden hour" \
    --seeds 42 \
    --scales 0.0 2.0 5.0 \
    --output_dir "shop_concept/outputs/test_vangogh_sky_maskdump" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --cache_dir "shop_concept/_peft_cache" \
    --dump_masks

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/test_vangogh_sky_maskdump
