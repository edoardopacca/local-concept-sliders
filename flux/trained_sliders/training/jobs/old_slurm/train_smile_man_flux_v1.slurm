#!/bin/bash
#SBATCH --job-name=flux_smile_man_v1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_smile_man_flux_v1.slurm
# =============================================================================
# Primo training REALE dello slider "smile_man" su Flux.1-dev (MIG 3g.40gb).
#
# Config:
#   - 500 step, rank 16, xattn, lr 0.002, warmup 200
#   - Eq.7 Concept Sliders con eta=2
#   - T5 offload attivo (vedi script train_flux_slider.py)
#   - gradient checkpointing attivo
#
# Tempo atteso: ~4-8h a seconda della velocita' effettiva su MIG
#   (stimare dal smoke test: s/step * 500 + ~5 min load)
# Walltime budget: 12h (safety margin su stima 4-8h)
#
# LANCIA PRIMA smoke_test_flux.slurm e aspetta che finisca OK prima
# di mandare questo job.
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

echo "=== Pre-flight ==="
nvidia-smi
echo ""

# Output dir con config nel nome per tracciabilita'
OUT_DIR="flux/trained_sliders/training/outputs/smile_man_flux_v1_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name smile_man_flux_v1 \
    --target_prompt "picture of a person" \
    --positive_prompt "photo of a person, smiling, happy" \
    --negative_prompt "photo of a person, frowning" \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --eta 2 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
