#!/bin/bash
#SBATCH --job-name=mlf_p3_sweep
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# masked_Lora_FLUX - PHASE 3 (dual-path velocity-blend edit)
# =============================================================================
# Legge flux/tasks/masked_lora/runs/<RUN_ID>/ (base.png, metadata.json, mask.png)
# ed esegue il denoising dual-path:
#
#   v_base   = transformer(latents, LoRA_off)
#   v_styled = transformer(latents, LoRA_on, scale=$SLIDER_SCALE)
#   v_final  = mask * v_styled + (1 - mask) * v_base
#   latents  = scheduler.step(v_final, t, latents)
#
# NO leak via self-attention: i due forward sono fisicamente indipendenti,
# il blend avviene FUORI dal transformer.
#
# Costo: ~2x compute/step. Su A100 512x512 30 step: ~70s/immagine.
# =============================================================================

set -euxo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p flux/tasks/masked_lora/logs flux/tasks/masked_lora/_peft_cache

echo "=== Pre-flight ==="
nvidia-smi
echo ""

# -----------------------------------------------------------------------------
# CONFIG - DEVE combaciare con RUN_ID dei phase 1 e 2
# -----------------------------------------------------------------------------
RUN_ID="vangogh_sky_seed42"
SLIDER_PATH="flux/trained_sliders/training/outputs/vangogh_flux_v1_rank16_xattn_alpha1/flux-vangogh_flux_v1/slider_0.pt"
# Sweep di scale: 1 load Flux + 6 denoising cycles
SCALES=(0 1 2 3 4 5)
EDIT_START_STEP=8

python -u flux/tasks/masked_lora/03_masked_edit.py \
    --run_dir "flux/tasks/masked_lora/runs/$RUN_ID" \
    --slider_path "$SLIDER_PATH" \
    --slider_scales "${SCALES[@]}" \
    --edit_start_step "$EDIT_START_STEP" \
    --lora_fill_rank 16 \
    --cache_dir flux/tasks/masked_lora/_peft_cache \
    --device cuda \
    --mask_name mask.png

echo ""
echo "=== DONE phase3 ==="
ls -la "flux/tasks/masked_lora/runs/$RUN_ID"
