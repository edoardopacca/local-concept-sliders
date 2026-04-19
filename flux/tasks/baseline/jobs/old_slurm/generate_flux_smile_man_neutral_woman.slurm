#!/bin/bash
#SBATCH --job-name=flux_gen_smile_neutral
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_flux_smile_man_neutral_woman.slurm
# =============================================================================
# Test attribute binding locale su Flux.1-dev, follow-up del baseline:
#
#   "a smiling man next to a woman with a neutral expression"
#
# Nel baseline abbiamo visto che con "serious woman" Flux a volte fa
# sorridere anche la donna. Qui rilassiamo il target a "neutral expression"
# (piu' vicino a "normale") per vedere se il leakage dello smile peggiora
# o si stabilizza. Utile per capire quanto l'attribute binding di Flux
# regge quando l'attributo del secondo soggetto e' debole/ambiguo.
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

OUT_DIR="flux/trained_sliders/training/outputs/generations/smile_neutral_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

python -u flux/tasks/baseline/generate_flux.py \
    --prompt "a smiling man next to a woman with a neutral expression" \
    --seeds 42 7 \
    --save_dir "$OUT_DIR" \
    --height 1024 --width 1024 \
    --steps 30

echo ""
echo "=== DONE ==="
ls -la "$OUT_DIR"
