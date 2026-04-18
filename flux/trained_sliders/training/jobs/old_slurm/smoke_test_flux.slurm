#!/bin/bash
#SBATCH --job-name=flux_smoke_test
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# smoke_test_flux.slurm
# =============================================================================
# SMOKE TEST: verifica che il setup Flux + T5 offload + training loop
# funzioni su MIG A100 3g.40gb (42 GB VRAM).
#
# Scopo:
#   (1) Confermare che carichiamo Flux.1-dev (transformer+VAE) senza OOM
#   (2) Confermare che T5 offload libera ~9.7 GB (mem_report ti dirà)
#   (3) Confermare che 50 training step girano senza OOM o errori runtime
#   (4) Stimare il tempo per step effettivo -> estrapolare tempo per 500 step
#
# NON produce uno slider utile (solo 50 step con prompt di default). Il file
# .pt generato e' da cancellare dopo.
#
# Tempo atteso: ~15-20 min (~3-5 min load + ~10-12 min training su MIG)
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
# Model gia' scaricato -> offline OK, piu' veloce all'avvio
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Debug CUDA OOM piu' informativo se servisse
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pre-flight checks ==="
nvidia-smi
echo ""
python -c "import torch; print(f'torch {torch.__version__} cuda_avail={torch.cuda.is_available()}'); print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""

# --- Smoke test: 50 step, output temporaneo ---
OUT_DIR="flux/trained_sliders/training/outputs/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "=== Lancio training (50 step, smoke test) ==="
echo "Output in: $OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 50 \
    --output_dir "$OUT_DIR" \
    --slider_name smoke_test \
    --target_prompt "picture of a person" \
    --positive_prompt "photo of a person, smiling, happy" \
    --negative_prompt "photo of a person, frowning" \
    --rank 16 --alpha 1 --train_method xattn \
    --lr_warmup_steps 50

echo ""
echo "=== SMOKE TEST COMPLETATO ==="
echo "Controlla log sopra per:"
echo "  - mem_report dopo T5 offload (deve essere ~25-28 GB)"
echo "  - tempo per step (cerca 's/step' nella riga finale)"
echo "  - assenza di OOM errors"
ls -la "$OUT_DIR"
