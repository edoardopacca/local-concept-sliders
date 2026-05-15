#!/bin/bash
#SBATCH --job-name=flux_eval_fur_p1
#SBATCH --account=3352759
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_furlength_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione furlength.
# Prompt LoRAShop-style: scene specifiche, ≥2 animali distinti, photorealistic.
# Fase 2 (SAM) in locale: cliccare sul corpo del PRIMO cane menzionato.
#
# Output: flux/tasks/masked_lora/runs/eval_furlength_{01..20}/
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

OUTPUT_ROOT="flux/tasks/masked_lora/runs"
mkdir -p "$OUTPUT_ROOT" flux/tasks/masked_lora/logs

echo "================================================================"
echo "=== FASE 1: eval_furlength Flux  (20 immagini base)         ==="
echo "================================================================"
nvidia-smi
echo ""

declare -a SEEDS=(
    1001 1002 1003 1004 1005
    1006 1007 1008 1009 1010
    1011 1012 1013 1014 1015
    1016 1017 1018 1019 1020
)

declare -a PROMPTS=(
    "a golden retriever and a labrador resting on a sunlit hardwood floor next to a large window, pet portrait, photorealistic, 4k"
    "a husky and a border collie playing in a snowy backyard at dawn, outdoor pet photo, photorealistic, 4k"
    "a poodle and a corgi waiting at the door of a country cottage, candid pet photo, photorealistic, 4k"
    "a beagle and a dachshund napping together on a tartan blanket beside a fireplace, indoor pet portrait, photorealistic, 4k"
    "a German shepherd and a labrador walking along a pebble beach at sunset, outdoor photo, photorealistic, 4k"
    "a samoyed and a malamute sitting in a snowy alpine meadow, outdoor pet portrait, photorealistic, 4k"
    "a cocker spaniel and a basset hound resting on a wooden porch in summer, candid pet photo, photorealistic, 4k"
    "a chihuahua and a pug curled up together on a velvet armchair, indoor pet portrait, photorealistic, 4k"
    "a shiba inu and a jack russell terrier exploring a mossy forest path, outdoor photo, photorealistic, 4k"
    "an Australian shepherd and a border collie running through a wheat field at golden hour, outdoor pet photo, photorealistic, 4k"
    "a dalmatian and a boxer sitting at attention in a leafy city park, candid pet photo, photorealistic, 4k"
    "a maltese and a bichon frise on a flowered garden bench, outdoor pet portrait, photorealistic, 4k"
    "a saint bernard and a bernese mountain dog resting in a mountain pasture, outdoor photo, photorealistic, 4k"
    "a doberman and a rottweiler sitting on a stone patio at dusk, outdoor pet portrait, photorealistic, 4k"
    "a yorkshire terrier and a maltese on a polished marble kitchen floor, indoor pet photo, photorealistic, 4k"
    "a great dane and a golden retriever standing together on a country road, outdoor portrait, photorealistic, 4k"
    "an Irish setter and a springer spaniel beside a lakeside dock at sunrise, outdoor pet photo, photorealistic, 4k"
    "a french bulldog and a pug napping on a sunny windowsill, indoor pet portrait, photorealistic, 4k"
    "a cavalier king charles spaniel and a beagle playing in an English garden, outdoor photo, photorealistic, 4k"
    "a vizsla and a weimaraner standing alert in a misty field at dawn, outdoor pet portrait, photorealistic, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_furlength_${RUN_NUM}"
    SEED="${SEEDS[$i]}"
    PROMPT="${PROMPTS[$i]}"

    echo "--- ${RUN_ID}  seed=${SEED} ---"
    echo "    ${PROMPT}"

    python flux/tasks/masked_lora/scripts/01_generate_base.py \
        --prompt        "$PROMPT" \
        --seed          "$SEED" \
        --steps         30 \
        --guidance_scale 3.5 \
        --max_sequence_length 256 \
        --height        1024 \
        --width         1024 \
        --run_id        "$RUN_ID" \
        --output_root   "$OUTPUT_ROOT" \
        --model_id      "black-forest-labs/FLUX.1-dev" \
        --device        cuda
    echo ""
done

echo "================================================================"
echo "=== FASE 1 COMPLETATA: eval_furlength Flux                  ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_furlength_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_furlength_* | wc -l)"
