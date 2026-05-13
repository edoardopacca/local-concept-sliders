#!/bin/bash
#SBATCH --job-name=mlf_p3_smile_age_on_both
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# phase3_two_subjects_smile_age_on_both.slurm
# =============================================================================
# Replica dell'esperimento shop_concept "two_subjects_smile_age_on_both"
# ma con maschere SAM esterne (masked_lora) invece di mask attention-based.
#
# Setup identico a quello shop_concept:
#   * 2 soggetti (man, woman) con 2 slider distinti per ognuno
#   * smile e age caricati 2 volte come adapter PEFT distinti (4 totali)
#   * 4 run con scale (smile_man, age_man, smile_woman, age_woman):
#       0,0,0,0           (baseline)
#       0.5,0.5,-0.5,-0.5
#       1,1,-1,-1
#       1.5,1.5,-1.5,-1.5
#
# Setup masked_lora:
#   * Maschere SAM (mask_man.png, mask_woman.png)
#   * Forward (1 base + 2 per-mask) per step, blend velocity FUORI dal modello
#   * edit_start_step=8 (default Python, equivalente a SDXL start_noise=750)
#
# -----------------------------------------------------------------------------
# WORKFLOW COMPLETO (3 fasi)
# -----------------------------------------------------------------------------
# 1. PHASE 1 (HPC): genera base.png + metadata.json
#      sbatch flux/tasks/masked_lora/jobs/new_slurm/phase1_two_subjects.slurm
#    → crea runs/two_subjects_masked_<TIMESTAMP>/{base.png, metadata.json}
#    → annota il TIMESTAMP del RUN_ID generato (vedi log dello SLURM)
#
# 2. PHASE 2 (Mac, interattivo): segmenta i 2 soggetti con SAM
#    Pull i risultati di phase1 sul Mac, poi:
#      RUN=flux/tasks/masked_lora/outputs/two_subjects_masked_<TIMESTAMP>
#      for name in man woman; do
#        python mask_SAM/segment_with_sam.py \
#          --run_dir "$RUN" \
#          --image_name base.png --mode interactive \
#          --output_name "mask_${name}.png" \
#          --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth
#      done
#    → click sul viso dell'uomo, poi sul viso della donna
#    → push delle maschere su HPC (git add + commit + push)
#
# 3. PHASE 3 (HPC): questo SLURM
#    Aggiorna RUN_ID sotto col valore di phase1, poi:
#      sbatch flux/tasks/masked_lora/jobs/new_slurm/phase3_two_subjects_smile_age_on_both.slurm
#    → 3 PNG dentro la run dir
#
# Output: 3 PNG dentro la run dir
#   edited_smile_age_on_both_01_all_zero.png
#   edited_smile_age_on_both_02_pos_man_neg_woman.png
#   edited_smile_age_on_both_03_extreme.png
#
# Compute phase3: 3 invocazioni × (~7 min carico Flux + ~5 min compute) = ~35 min totali
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

mkdir -p flux/tasks/masked_lora/logs flux/tasks/masked_lora/_peft_cache

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# ⚠️  AGGIORNA RUN_ID con il timestamp generato da phase1_two_subjects.slurm.
#     Dopo aver lanciato phase1, controlla:
#       ls flux/tasks/masked_lora/outputs/ | grep two_subjects_masked
#     e copia qui sotto il nome esatto della cartella.
RUN_ID="two_subjects_masked_20260424_1910"
RUN_DIR="flux/tasks/masked_lora/outputs/${RUN_ID}"

SLIDER_SMILE="flux/trained_sliders/sliders/smile.pt"
SLIDER_AGE="flux/trained_sliders/sliders/age.pt"

# 2 maschere SAM disgiunte
MASK_NAMES=(mask_man.png mask_woman.png)

# 4 adapter PEFT (smile e age caricati 2 volte, una per regione)
SLIDER_PATHS=("$SLIDER_SMILE" "$SLIDER_AGE" "$SLIDER_SMILE" "$SLIDER_AGE")
SLIDER_TO_MASK=(0 0 1 1)

CACHE_DIR="flux/tasks/masked_lora/_peft_cache"

echo "================================================================"
echo "=== masked_lora multi-mask multi-LoRA: 2 soggetti × 2 slider ==="
echo "================================================================"
echo "run_dir         : $RUN_DIR"
echo "masks           : ${MASK_NAMES[*]}"
echo "slider PEFT     : smile@man age@man smile@woman age@woman"
echo "slider->mask    : ${SLIDER_TO_MASK[*]}"
echo "edit_start_step : 8 (default Python)"
echo ""
nvidia-smi
echo ""

run_one() {
    local NAME="$1"
    local SM_M="$2" AG_M="$3" SM_W="$4" AG_W="$5"
    local OUT_NAME="edited_smile_age_on_both_${NAME}.png"

    echo "----------------------------------------"
    echo "[run] $NAME"
    echo "  man  : smile=${SM_M}  age=${AG_M}"
    echo "  woman: smile=${SM_W}  age=${AG_W}"
    echo "  out  : $OUT_NAME"
    echo "----------------------------------------"

    python flux/tasks/masked_lora/scripts/03_masked_edit.py \
        --run_dir         "$RUN_DIR" \
        --slider_paths    "${SLIDER_PATHS[@]}" \
        --mask_names      "${MASK_NAMES[@]}" \
        --slider_to_mask  "${SLIDER_TO_MASK[@]}" \
        --slider_scales   "$SM_M" "$AG_M" "$SM_W" "$AG_W" \
        --lora_fill_rank  16 \
        --cache_dir       "$CACHE_DIR" \
        --device          cuda \
        --output_name     "$OUT_NAME"
}

run_one "01_all_zero"           0    0    0    0
run_one "02_pos_man_neg_woman"  1    1   -1   -1
run_one "03_extreme"            2    2   -2   -2

echo ""
echo "================================================================"
echo "=== DONE — outputs in: $RUN_DIR                              ==="
echo "================================================================"
ls -la "$RUN_DIR" | grep "edited_smile_age_on_both"
