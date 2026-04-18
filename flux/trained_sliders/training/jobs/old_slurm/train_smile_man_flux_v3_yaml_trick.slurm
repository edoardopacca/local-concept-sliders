#!/bin/bash
#SBATCH --job-name=flux_smile_man_v3_yaml_trick
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=flux/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=flux/trained_sliders/training/logs/%x_%j.err

# =============================================================================
# train_smile_man_flux_v3_yaml_trick.slurm
# =============================================================================
# Port fedele su Flux.1-dev di SDXL_train/jobs/train_smile_man_strong_v3_yaml_trick.slurm
#
# Usa il NUOVO supporto --prompts_yaml in train_flux_slider.py per
# samplare uniformemente da un set di 20 prompt entries (10 man enhance
# + 10 woman preservation). La preservation e' implementata via
# degeneration della Eq.7 (target==positive==unconditional==neutral
# collapsa loss in MSE(LoRA_on, LoRA_off) sulle woman entries).
#
# Ratio enhance:preservation = 50:50 (lambda_pres ≈ 1.0 analogico).
# Costo per step variabile: 3 forward LoRA-off su man entries
# (target==neutral quindi 3 unique), 1 forward LoRA-off su woman
# entries (tutti uguali). Media ~2 forward/step -> lievemente piu'
# veloce del single-triple training.
#
# Config:
#   - 500 step (stesso budget di smile_man_flux_v1)
#   - rank 16, xattn, lr 0.002, warmup 200 (come v1)
#   - eta per-entry dal YAML: 2 per man, 1 per woman (irrelevante)
#   - T5 offload + gradient checkpointing attivi (auto da script)
#
# Tempo atteso (estrapolando da smoke 7.3s/step):
#   - Average ~2 off-forwards per step (woman entries piu' veloci)
#   - Stima: ~5.5-6.5 s/step -> 500 step ~45-55 min + ~7 min load
#   - Total ~55 min ± 10min. Walltime 4h per sicurezza.
#
# Setup analogo di v3_yaml_trick SDXL (che gira in ~45min):
#   - Stessi 10 man entries (5 etnie x 2 varianti phrasing)
#   - Stessi 10 woman entries (5 etnie x 2 varianti: bare / portrait)
#   - Differenza chiave: eta=2 invece di 4 (stabilita' Flux untested oltre)
#
# Lancia DOPO aver verificato che train_flux_slider.py nuovo parsa lo
# YAML correttamente (smoke test consigliato).
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

# Safety: assicuriamoci che pyyaml sia installato nel venv flux-sliders.
# Se gia' presente, pip skip. Altrimenti install veloce (<5s).
python -c "import yaml" 2>/dev/null || pip install pyyaml --quiet

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Pre-flight ==="
nvidia-smi
echo ""

OUT_DIR="flux/trained_sliders/training/outputs/smile_man_flux_v3_yaml_trick_rank16_xattn_alpha1"
mkdir -p "$OUT_DIR"

python -u flux/trained_sliders/training/scripts/train_flux_slider.py \
    --max_train_steps 500 \
    --output_dir "$OUT_DIR" \
    --slider_name smile_man_flux_v3_yaml_trick \
    --prompts_yaml flux/trained_sliders/training/prompts/old_prompt/smile_man_flux_v3_yaml_trick.yaml \
    --rank 16 --alpha 1 --train_method xattn \
    --lr 0.002 --lr_warmup_steps 200 \
    --save_every 100

echo ""
echo "=== TRAINING COMPLETATO ==="
ls -la "$OUT_DIR"
