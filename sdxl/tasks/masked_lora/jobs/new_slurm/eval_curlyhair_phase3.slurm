#!/bin/bash
#SBATCH --job-name=eval_curly_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_curlyhair_phase3.slurm
# =============================================================================
# FASE 3 — Applica il masked LoRA curlyhair su tutte e 20 le run.
#
# Requirements: base.png + init_latents.pt (fase 1)  e  mask.png (fase 2)
#
# Slider: curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn  (generale)
# Scale testate: 1.5  2.0  3.0
#
# Phase 2 — produce the masks locally:
#   Per ogni eval_curlyhair_{NN}:
#     python mask_SAM/segment_with_sam.py \
#       --image_path sdxl/tasks/masked_lora/runs/eval_curlyhair_NN/base.png \
#       --output_dir sdxl/tasks/masked_lora/runs/eval_curlyhair_NN \
#       --mode interactive
#   Cliccare sui CAPELLI del soggetto da editare (qualunque persona).
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
SLIDERS_ROOT="sdxl/trained_sliders/sliders"
SLIDER_PATH="${SLIDERS_ROOT}/curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn/curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"
RUNS_ROOT="sdxl/tasks/masked_lora/runs"

echo "================================================================"
echo "=== FASE 3: eval_curlyhair  (20 runs × 3 scale)             ==="
echo "================================================================"
nvidia-smi
echo ""

for i in $(seq 1 20); do
    RUN_NUM=$(printf "%02d" "$i")
    RUN_ID="eval_curlyhair_${RUN_NUM}"
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
    for SCALE in 1.5 2.0 3.0; do
        echo "--- scale=${SCALE} ---"
        python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
            --run_dir      "$RUN_DIR" \
            --slider_path  "$SLIDER_PATH" \
            --slider_scale "$SCALE" \
            --rank         4 \
            --start_noise  750 \
            --model_id     "$MODEL_ID" \
            --device       cuda \
            --dtype        float16 \
            --mask_name    mask_target.png \
            --output_name  "edited_curly_s${SCALE}.png" \
            --skip_metrics
    done
    echo ""
done

echo "================================================================"
echo "=== FASE 3 COMPLETED: eval_curlyhair                       ==="
echo "================================================================"
echo "Run afterwards:"
echo "  python metrics/eval_masked.py --concept curlyhair --device cuda"
