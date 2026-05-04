#!/bin/bash
#SBATCH --job-name=flux_eval_fur_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_furlength_phase3.slurm  (Flux)
# =============================================================================
# FASE 3 — Applica il masked LoRA furlength su tutte e 20 le run.
# Scale testate: 1.5  2.0  3.0
# ⚠️  IMPOSTA IL PATH DEL TUO SLIDER QUI.
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

# ⚠️  Imposta il path del tuo slider Flux per furlength:
SLIDER_PATH="flux/trained_sliders/sliders/<FUR_SLIDER_DIR>/<FUR_SLIDER_FILE>_last.pt"

RUNS_ROOT="flux/tasks/masked_lora/runs"
CACHE_DIR="flux/tasks/masked_lora/_peft_cache"

echo "================================================================"
echo "=== FASE 3: eval_furlength Flux  (20 runs × 3 scale)        ==="
echo "================================================================"
nvidia-smi
echo ""

for i in $(seq 1 20); do
    RUN_NUM=$(printf "%02d" "$i")
    RUN_ID="eval_furlength_${RUN_NUM}"
    RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

    if [ ! -d "$RUN_DIR" ]; then echo "[SKIP] ${RUN_DIR}"; continue; fi
    if [ ! -f "${RUN_DIR}/mask_target.png" ]; then echo "[SKIP] mask assente: ${RUN_DIR}"; continue; fi

    echo "==== ${RUN_ID} ===="
    python flux/tasks/masked_lora/scripts/03_masked_edit.py \
        --run_dir      "$RUN_DIR" \
        --slider_path  "$SLIDER_PATH" \
        --slider_scales 1.5 2.0 3.0 \
        --mask_name    mask_target.png \
        --lora_fill_rank 16 \
        --cache_dir    "$CACHE_DIR" \
        --device       cuda

    for SCALE in 1.5 2.0 3.0; do
        SRC="${RUN_DIR}/edited_scale${SCALE}.png"
        DST="${RUN_DIR}/edited_furlength_s${SCALE}.png"
        [ -f "$SRC" ] && mv "$SRC" "$DST" && echo "  renamed: $(basename $DST)"
    done
    echo ""
done

echo "================================================================"
echo "=== FASE 3 COMPLETATA: eval_furlength Flux                  ==="
echo "================================================================"
echo "Eseguire dopo:"
echo "  python metrics/eval_masked.py --concept furlength \\"
echo "    --runs_root flux/tasks/masked_lora/runs --device cuda"
