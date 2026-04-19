#!/bin/bash
#SBATCH --job-name=flux_gen_baseline
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_baseline.slurm
# =============================================================================
# Genera baseline di Flux.1-dev (senza slider): uomo sorridente, donna
# seria, coppie. Varianti di prompt comparabili a quelli del setup SDXL
# per avere un riferimento visivo del base model Flux.
#
# 2 prompt x 3 seed = 6 immagini 1024x1024
# Tempo atteso: ~8-12 min su MIG 3g.40gb
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

OUT_DIR="flux/trained_sliders/training/outputs/generations/baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

python -u flux/tasks/baseline/generate_flux.py \
    --prompt "a smiling man next to a serious woman, portrait, photograph" \
    --prompt "two people: a smiling man on the left, a serious woman on the right, portrait, shoulders up" \
    --seeds 42 7 2026 \
    --save_dir "$OUT_DIR" \
    --height 1024 --width 1024 \
    --steps 30

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
