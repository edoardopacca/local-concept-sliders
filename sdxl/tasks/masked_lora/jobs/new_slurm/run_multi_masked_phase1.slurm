#!/bin/bash
#SBATCH --job-name=multi_masked_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=sdxl/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# run_multi_masked_phase1.slurm
# =============================================================================
# FASE 1 — Genera le 6 immagini base per i masked LoRA experiments.
#
# Scena e regione da segmentare con SAM (fase 2 locale):
#   masked_age_person_01    → gruppo di 3 persone → clicca su UNA persona
#   masked_curlyhair_01     → uomo e donna       → clicca sui capelli dell'uomo
#   masked_daynight_01      → paesaggio           → clicca sul cielo
#   masked_furlength_01     → cane e gatto        → clicca sul cane
#   masked_smile_person_01  → uomo e donna        → clicca sul viso di uno dei due
#   masked_painterly_01     → villaggio europeo   → clicca sul cielo o su un edificio
#
# Workflow completo (3 fasi):
#   1. sbatch run_multi_masked_phase1.slurm       <- questo file (HPC)
#   2. SAM localmente (vedi istruzioni sotto)
#   3. sbatch run_masked_<slider>_phase3.slurm    <- uno per slider (HPC)
#
# Istruzioni fase 2 (locale, dopo pull delle run dir):
#   for RUN in masked_age_person_01 masked_curlyhair_01 masked_daynight_01 \
#              masked_furlength_01 masked_smile_person_01 masked_painterly_01; do
#     python mask_SAM/segment_with_sam.py \
#       --run_dir sdxl/tasks/masked_lora/runs/${RUN} \
#       --sam_checkpoint ~/sam_vit_h_4b8939.pth \
#       --mode interactive \
#       --image_name base.png --output_name mask.png
#   done
#   # poi push le mask e lancia i phase3.
#
# Output:
#   sdxl/tasks/masked_lora/runs/masked_age_person_01/
#   sdxl/tasks/masked_lora/runs/masked_curlyhair_01/
#   sdxl/tasks/masked_lora/runs/masked_daynight_01/
#   sdxl/tasks/masked_lora/runs/masked_furlength_01/
#   sdxl/tasks/masked_lora/runs/masked_smile_person_01/
#   sdxl/tasks/masked_lora/runs/masked_painterly_01/
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p sdxl/tasks/masked_lora/logs sdxl/tasks/masked_lora/runs

MODEL_ID="$FERT_HF_CACHE/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
NEG_PROMPT="painting, illustration, cartoon, blurry, low quality, watermark, text, signature"

echo "================================================================"
echo "=== FASE 1: GENERATE BASE IMAGES (6 scenes)                  ==="
echo "================================================================"
nvidia-smi
echo ""

gen() {
    local RUN_ID="$1"
    local PROMPT="$2"
    local SEED="$3"
    echo "--- Generating: ${RUN_ID} ---"
    echo "    Prompt: ${PROMPT}"
    python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
      --prompt "$PROMPT" \
      --negative_prompt "$NEG_PROMPT" \
      --seed "$SEED" \
      --steps 50 \
      --guidance_scale 7.5 \
      --height 1024 \
      --width 1024 \
      --run_id "$RUN_ID" \
      --output_root sdxl/tasks/masked_lora/runs \
      --model_id "$MODEL_ID" \
      --device cuda \
      --dtype float16
    echo ""
}

# 1. Age person — group of 3 people, segment one person to age only them
gen "masked_age_person_01" \
    "a photorealistic group portrait of three people, two men and one woman, different ages, sharp focus, studio lighting, 4k" \
    42

# 2. Curly hair — man and woman, segment the man's hair region
gen "masked_curlyhair_01" \
    "a photorealistic portrait of a man and a woman standing together, sharp focus, studio lighting, 4k" \
    42

# 3. Day-night — landscape with sky, segment the sky
gen "masked_daynight_01" \
    "a photorealistic landscape with mountains and a winding river, vivid blue sky with clouds, sharp focus, 4k" \
    42

# 4. Fur length — dog and cat together, segment only the dog
gen "masked_furlength_01" \
    "a photorealistic portrait of a dog and a cat sitting side by side, sharp focus, studio lighting, 4k" \
    42

# 5. Smile person — man and woman, segment one face
gen "masked_smile_person_01" \
    "a photorealistic portrait of a man and a woman, neutral expressions, sharp focus, studio lighting, 4k" \
    42

# 6. Painterly — European village, segment sky or a building
gen "masked_painterly_01" \
    "a photorealistic European village with a church and red rooftops, sharp focus, 4k" \
    42

echo "================================================================"
echo "=== FASE 1 COMPLETATA                                         ==="
echo "=== PROSSIMO PASSO: scarica le run dir, applica SAM          ==="
echo "=== localmente (vedi commento in cima a questo file),        ==="
echo "=== poi push le mask e lancia i phase3.                      ==="
echo "================================================================"
for RUN in masked_age_person_01 masked_curlyhair_01 masked_daynight_01 \
           masked_furlength_01 masked_smile_person_01 masked_painterly_01; do
    echo "--- ${RUN} ---"
    ls -lh "sdxl/tasks/masked_lora/runs/${RUN}/"
done
