#!/bin/bash
#SBATCH --job-name=shop_vg_highscl
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=flux/tasks/shop_concept/logs/%x_%j.out
#SBATCH --error=flux/tasks/shop_concept/logs/%x_%j.err

# =============================================================================
# debug_vangogh_highscale_earlystart.slurm
# =============================================================================
# Diagnostica: scenario (B) confermato (mountains non cambiano mai). Ora
# testiamo due ipotesi insieme prima di buttarci sul mask-dump:
#
#   1) edit_start_step=8 troppo tardi per uno stile (van Gogh imposta
#      colori/texture nei primi step; se partiamo dallo step 8/30 il
#      latente e' gia' su traiettoria "foto" e non redirige).
#   2) scale=1.0 attenuata dal blend con la mask -> pushiamo a 2.0 e 5.0
#      per vedere se satura.
#
# Parametri minimi per velocita': 1 seed, 3 scale -> 3 immagini.
#
# Risultati attesi:
#   (B1) Anche a scale=5.0 con edit_start_step=5: NULLA sulle montagne
#        -> lo slider non passa proprio. Mask vuota o LoRA non applicata
#        in quel path. Next: dump mask come PNG.
#   (B2) scale=2.0 leggero effetto, scale=5.0 van Gogh evidente
#        -> era solo attenuazione/timing. Tuning scale+start fine.
#   (B3) effetto indipendente dalla scale, ma non "van Gogh" (tipo
#        artefatti random) -> loading bug della LoRA. Ricontrolliamo
#        convert_slider_to_peft.
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
    --scales 0.0 2.0 5.0 \
    --output_dir "shop_concept/outputs/debug_vangogh_highscale_earlystart" \
    --height 512 --width 512 \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --cache_dir "shop_concept/_peft_cache"

echo ""
echo "=== DONE ==="
ls -la shop_concept/outputs/debug_vangogh_highscale_earlystart
