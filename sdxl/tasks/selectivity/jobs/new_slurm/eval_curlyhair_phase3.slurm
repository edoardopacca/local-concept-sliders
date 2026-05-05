#!/bin/bash
#SBATCH --job-name=sel_curly_p3
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=sdxl/tasks/selectivity/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/selectivity/logs/%x_%j.err

# =============================================================================
# eval_curlyhair_phase3.slurm  (selectivity task)
# =============================================================================
# FASE 3 — Applica GLOBALMENTE (senza maschera spaziale) due slider curlyhair:
#   - SPECIFICO: curlyhair_man    → deve modificare solo l'uomo
#   - GENERALE:  curlyhair_person → baseline, modifica tutti
#
# La selettività si misura dopo con eval_selectivity.py usando
# mask_target.png (uomo) e mask_nontarget.png (donna) create in fase 2.
#
# ⚠️  IMPOSTA I PATH DEI TUOI SLIDER QUI:
# =============================================================================

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_sdxl_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
SLIDERS_ROOT="sdxl/trained_sliders/sliders"

# ⚠️  IMPOSTA IL PATH DEL TUO SLIDER SPECIFICO QUI (curlyhair_man):
SLIDER_SPECIFIC="${SLIDERS_ROOT}/curlyhair_man_sdxl_v1_alpha1.0_rank4_noxattn/curlyhair_man_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"

# Slider generale (curlyhair_person — già usato in masked_lora):
SLIDER_GENERAL="${SLIDERS_ROOT}/curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn/curlyhair_person_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"

RUNS_ROOT="sdxl/tasks/selectivity/runs"

echo "================================================================"
echo "=== FASE 3: selectivity/curlyhair  (20 runs × 2 slider × 3 scale) ==="
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
    if [ ! -f "${RUN_DIR}/base.png" ]; then
        echo "[SKIP] base.png assente (fase 1 non eseguita): ${RUN_DIR}"
        continue
    fi

    echo "==== ${RUN_ID} ===="

    echo "  [specifico] curlyhair_man"
    python sdxl/tasks/selectivity/scripts/03_apply_slider.py \
        --run_dir        "$RUN_DIR" \
        --slider_path    "$SLIDER_SPECIFIC" \
        --slider_scales  1.0 2.0 3.0 \
        --output_prefix  "edited_curlyhair_specific" \
        --rank           4 \
        --start_noise    750 \
        --model_id       "$MODEL_ID" \
        --device         cuda \
        --dtype          float16

    echo "  [generale] curlyhair_person"
    python sdxl/tasks/selectivity/scripts/03_apply_slider.py \
        --run_dir        "$RUN_DIR" \
        --slider_path    "$SLIDER_GENERAL" \
        --slider_scales  1.0 2.0 3.0 \
        --output_prefix  "edited_curlyhair_general" \
        --rank           4 \
        --start_noise    750 \
        --model_id       "$MODEL_ID" \
        --device         cuda \
        --dtype          float16

    echo ""
done

echo "================================================================"
echo "=== FASE 3 COMPLETATA: selectivity/curlyhair                ==="
echo "================================================================"
echo "Per ogni run trovi:"
echo "  edited_curlyhair_specific_s1.0.png / s2.0.png / s3.0.png"
echo "  edited_curlyhair_general_s1.0.png  / s2.0.png / s3.0.png"
echo ""
echo "Esegui poi (dopo fase 2 maschere):"
echo "  python metrics/eval_selectivity.py --concept curlyhair --device cuda"
