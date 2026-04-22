#!/bin/bash
#SBATCH --job-name=shop_vangogh_mtn
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# debug_vangogh_mountains.slurm
# =============================================================================
# Test diagnostico: stesso slider van Gogh, stesso prompt, ma target
# spostato da "sky" a "mountains". Serve a capire se il bug del sweep
# precedente (van Gogh non applicato) e' specifico del token "sky" o
# piu' generale.
#
# Tre scenari possibili dopo questo job:
#
# (A) mountains diventano van Gogh stilizzate, resto rimane foto -> bug
#     e' specifico del token "sky": mask extraction in LoRAShop non
#     localizza bene l'attention di "sky". Soluzione: prompt reword o
#     tweak mask thresholds in flux_real_pipeline.py.
#
# (B) mountains NON diventano van Gogh, immagini uguali tra loro -> il
#     problema non e' sky-specifico, lo slider van Gogh non si applica
#     in LoRAShop per qualche altro motivo (propagazione scale, alpha,
#     ecc.). Andiamo in debug piu' profondo (dump delle mask, ecc.).
#
# (C) mountains leggermente stilizzate ma non pienamente van Gogh ->
#     il blending funziona, ma la mask e' troppo ristretta o troppo
#     diluita. Serve tuning dei threshold.
#
# Stesso parametro set del sweep_vangogh_sky_v2 per confronto 1:1:
#   - seeds (42, 123)
#   - scales (0.0, 0.3, 0.7, 1.0)
#   - 512x512, 30 step, edit_start 8, guidance 3.5
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
    --seeds 42 123 \
    --scales 0.0 0.3 0.7 1.0 \
    --output_dir "shop_concept/outputs/debug_vangogh_mountains" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --edit_start_step 8 \
    --cache_dir "shop_concept/_peft_cache"

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/debug_vangogh_mountains
