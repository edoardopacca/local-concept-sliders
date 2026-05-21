#!/bin/bash
#SBATCH --job-name=flux_eval_smile_p1
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# eval_smile_person_phase1.slurm  (Flux)
# =============================================================================
# FASE 1 — Genera 20 immagini base per il set di valutazione smile_person.
# Prompt LoRAShop-style: scene specifiche, ≥2 persone, photorealistic, una
# donna identificabile come target.
# Fase 2 (SAM) in locale: cliccare sul viso della donna (target_prompt).
#
# Output: flux/tasks/masked_lora/runs/eval_smile_person_{01..20}/
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
echo "=== FASE 1: eval_smile_person Flux  (20 immagini base)      ==="
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
    "a woman in a white silk blouse and a man in a navy blazer share espresso at a Parisian café, neutral expression, portrait photo, photorealistic, 4k"
    "two women in floral summer dresses walk through a sunlit botanical garden, facing the camera, neutral expression, candid portrait, photorealistic, 4k"
    "a woman in a gray cashmere sweater and a man in a tweed coat read newspapers by a fireplace, neutral expression, indoor portrait, photorealistic, 4k"
    "a woman in a beige trench coat and a man in a wool overcoat wait at a London bus stop, neutral expression, candid photo, photorealistic, 4k"
    "a young woman in a red wool sweater and an elderly man in a corduroy jacket sit on a park bench in autumn, neutral expression, outdoor portrait, photorealistic, 4k"
    "a woman in a black silk blouse and a man in a charcoal suit toast champagne at a rooftop dinner, neutral expression, portrait photo, photorealistic, 4k"
    "two women in matching denim jackets browse vinyl records at a vintage shop, neutral expression, candid photo, photorealistic, 4k"
    "a woman in a teal wrap dress and a man in a linen shirt dance at a beach wedding reception, neutral expression, outdoor portrait, photorealistic, 4k"
    "a woman in a yellow sundress and a man in a striped polo enjoy gelato on a Mediterranean piazza, neutral expression, candid portrait, photorealistic, 4k"
    "a woman in a navy peacoat and a man in a thick wool scarf admire the view from a snowy mountain overlook, facing the camera, neutral expression, outdoor photo, photorealistic, 4k"
    "a mother in a soft pink cardigan and her teenage daughter in a graphic t-shirt bake cookies in a sunlit kitchen, neutral expression, candid photo, photorealistic, 4k"
    "a woman in a green plaid blazer and a man in a brown leather jacket sit at a vintage bookstore reading area, neutral expression, indoor portrait, photorealistic, 4k"
    "two women in evening gowns share a moment under a chandelier at an art gallery opening, neutral expression, candid portrait, photorealistic, 4k"
    "a woman in an emerald velvet dress and a man in a dark suit sit by candlelight at a fine dining restaurant, facing the camera, neutral expression, indoor portrait, photorealistic, 4k"
    "a woman in a chunky knit sweater and a man in a parka warm their hands by a campfire at a winter cabin, neutral expression, outdoor photo, photorealistic, 4k"
    "a young woman in a denim jacket and her grandmother in a lavender shawl tend to a cottage garden, neutral expression, outdoor portrait, photorealistic, 4k"
    "two women in athletic wear stretch on yoga mats in a bright urban studio, facing the camera, neutral expression, candid photo, photorealistic, 4k"
    "a woman in a cream silk blouse and a man in an oxford shirt collaborate over a laptop at a coworking space, neutral expression, indoor portrait, photorealistic, 4k"
    "a woman in a coral linen shirt and a man in a panama hat browse fresh produce at a Sunday farmers market, neutral expression, outdoor photo, photorealistic, 4k"
    "a woman in a tailored black blazer and a man in a slate gray shirt review architectural plans on a construction site, neutral expression, candid portrait, photorealistic, 4k"
)

for i in "${!SEEDS[@]}"; do
    RUN_NUM=$(printf "%02d" $((i + 1)))
    RUN_ID="eval_smile_person_${RUN_NUM}"
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
echo "=== FASE 1 COMPLETED: eval_smile_person Flux               ==="
echo "================================================================"
ls -lh "${OUTPUT_ROOT}/eval_smile_person_01/"
echo "Totale run dir: $(ls -d ${OUTPUT_ROOT}/eval_smile_person_* | wc -l)"
