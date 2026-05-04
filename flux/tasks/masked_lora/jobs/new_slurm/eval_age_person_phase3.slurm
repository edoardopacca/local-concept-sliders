#!/bin/bash
#SBATCH --job-name=flux_eval_age_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_age_person_phase3.slurm  (Flux)
# =============================================================================
# FASE 3 — Applica il masked LoRA age_person su tutte e 20 le run.
# Prerequisiti: base.png + metadata.json (fase 1) e mask_target.png (fase 2)
#
# Scale testate: 0.5  1.0  1.5
#
# ⚠️  IMPOSTA IL PATH DEL TUO SLIDER QUI:
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

# ⚠️  Imposta il path del tuo slider Flux per age_person:
SLIDER_PATH="flux/trained_sliders/sliders/<AGE_SLIDER_DIR>/<AGE_SLIDER_FILE>_last.pt"

RUNS_ROOT="flux/tasks/masked_lora/runs"
CACHE_DIR="flux/tasks/masked_lora/_peft_cache"

echo "================================================================"
echo "=== FASE 3: eval_age_person Flux  (20 runs × 3 scale)       ==="
echo "================================================================"
nvidia-smi
echo ""

for i in $(seq 1 20); do
    RUN_NUM=$(printf "%02d" "$i")
    RUN_ID="eval_age_person_${RUN_NUM}"
    RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

    if [ ! -d "$RUN_DIR" ]; then
        echo "[SKIP] Run dir non trovata: ${RUN_DIR}"
        continue
    fi
    if [ ! -f "${RUN_DIR}/mask_target.png" ]; then
        echo "[SKIP] Maschera assente: ${RUN_DIR}/mask_target.png"
        continue
    fi

    echo "==== ${RUN_ID} ===="

    # Sweep: carica il modello una volta sola per run dir
    python flux/tasks/masked_lora/scripts/03_masked_edit.py \
        --run_dir      "$RUN_DIR" \
        --slider_path  "$SLIDER_PATH" \
        --slider_scales 0.5 1.0 1.5 \
        --mask_name    mask_target.png \
        --lora_fill_rank 16 \
        --cache_dir    "$CACHE_DIR" \
        --device       cuda

    # Rinomina output sweep → naming convention eval_masked.py
    for SCALE in 0.5 1.0 1.5; do
        SRC="${RUN_DIR}/edited_scale${SCALE}.png"
        DST="${RUN_DIR}/edited_age_s${SCALE}.png"
        [ -f "$SRC" ] && mv "$SRC" "$DST" && echo "  renamed: $(basename $DST)"
    done
    echo ""
done

echo "================================================================"
echo "=== FASE 3 COMPLETATA: eval_age_person Flux                 ==="
echo "================================================================"
echo "Eseguire dopo:"
echo "  python metrics/eval_masked.py --concept age_person \\"
echo "    --runs_root flux/tasks/masked_lora/runs --device cuda"
