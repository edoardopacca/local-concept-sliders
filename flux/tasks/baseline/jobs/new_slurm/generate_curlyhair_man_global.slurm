#!/bin/bash
#SBATCH --job-name=gen_curl_man_v1
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_curlyhair_man_global.slurm  —  OBSOLETE (cannot run)
# =============================================================================
# This job generated a baseline grid for a subject-specific Flux
# curlyhair slider whose weights and training prompt YAML have been
# removed from the repository. It is retained only because deleting it
# would require also editing surrounding code paths. The current paper
# proposes a single method — external mask-guided localization of a
# LoRA adapter at inference time (§4) — and does not include a
# subject-specific training variant.
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SLIDER_DIR="flux/trained_sliders/sliders/trial_v1/curlyhair_man_flux_v1/flux-curlyhair_man_flux_v1"
SAVE_DIR="flux/tasks/baseline/outputs/curlyhair_man_global"
PROMPT="a portrait of a man, bust shot, looking at camera, natural lighting, photorealistic, 4k, sharp focus"

mkdir -p flux/tasks/baseline/logs "$SAVE_DIR"

echo "================================================================"
echo "=== FASE 1+3: curlyhair_man_flux_v1  (10 seed × 6 scale)    ==="
echo "================================================================"
nvidia-smi || true
echo "Slider: $SLIDER_DIR/slider_0.pt"
echo "Prompt: $PROMPT"
echo ""

python -u flux/tasks/baseline/scripts/generate_flux_slider.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --lora_dirs    "$SLIDER_DIR" \
    --lora_scales  0 0.5 1.0 1.5 2.0 3.0 \
    --prompt       "$PROMPT" \
    --seeds        42 7 123 456 789 1024 2048 31415 1337 65535 \
    --save_dir     "$SAVE_DIR" \
    --height 1024 --width 1024 \
    --steps 50 --guidance_scale 3.5 \
    --max_sequence_length 512 \
    --rank 16 --alpha 1.0 --train_method xattn \
    --skip_slider_timestep_till 8 \
    --cuda_device 0

echo ""
echo "================================================================"
echo "=== COMPLETATO                                              ==="
echo "================================================================"
find "$SAVE_DIR" -type f | sort | head
echo "..."
echo "Totale file: $(find "$SAVE_DIR" -type f | wc -l)"
