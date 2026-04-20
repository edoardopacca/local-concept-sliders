#!/bin/bash
#SBATCH --job-name=mlf_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# masked_Lora_FLUX - PHASE 1 (Flux base generation)
# =============================================================================
# Genera la base image Flux in flux/tasks/masked_lora/runs/<RUN_ID>/
#   - base.png
#   - metadata.json   (seed, prompt, steps, scheduler_config, ecc.)
#
# metadata.json viene riletto da phase 3 per riprodurre esattamente la
# stessa traiettoria di denoising (stesso seed, stesso scheduler config).
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

mkdir -p flux/tasks/masked_lora/logs flux/tasks/masked_lora/runs

echo "=== Pre-flight ==="
nvidia-smi
echo ""

# -----------------------------------------------------------------------------
# CONFIG - paesaggio con cielo ben visibile (stesso prompt di shop sweep
# cosi' i risultati sono confrontabili 1:1 con LoRAShop)
# -----------------------------------------------------------------------------
PROMPT="a natural landscape, a mountain range under a vast sky, photorealistic, golden hour"
SEED=42
RUN_ID="vangogh_sky_seed42"

python -u flux/tasks/masked_lora/01_generate_base.py \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --steps 30 \
    --guidance_scale 3.5 \
    --max_sequence_length 256 \
    --height 512 \
    --width 512 \
    --model_id black-forest-labs/FLUX.1-dev \
    --run_id "$RUN_ID" \
    --output_root flux/tasks/masked_lora/runs \
    --device cuda

echo ""
echo "=== DONE phase1 ==="
ls -la "flux/tasks/masked_lora/runs/$RUN_ID"
