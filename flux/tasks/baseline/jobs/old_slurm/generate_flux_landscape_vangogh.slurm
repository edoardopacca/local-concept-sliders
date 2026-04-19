#!/bin/bash
#SBATCH --job-name=flux_gen_landscape_vangogh
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_landscape_vangogh.slurm
# =============================================================================
# Test rapido di attribute binding locale su Flux.1-dev:
#
#   "natural landscape hyperrealistic, only the sky in Van Gogh style"
#
# Verifica se Flux riesce a localizzare lo stile Van Gogh al solo cielo
# mantenendo iperreale il resto della scena.
#
# 1 prompt x 2 seed = 2 immagini 1024x1024, ~5-6 min (cache warm).
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

OUT_DIR="flux/trained_sliders/training/outputs/generations/landscape_vangogh_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

python -u flux/tasks/baseline/generate_flux.py \
    --prompt "natural landscape hyperrealistic, only the sky in Van Gogh style" \
    --seeds 42 7 \
    --save_dir "$OUT_DIR" \
    --height 1024 --width 1024 \
    --steps 30

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
