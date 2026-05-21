#!/bin/bash
#SBATCH --job-name=gen_global_paint_age
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# generate_global_painterly_age.slurm
# =============================================================================
# Genera "fase 1 + fase 3" globali (no mask) in un unico job:
#
#   PARTE 1 — SDXL  : paesaggio (lago di montagna + bosco) × 2 seed × slider
#                     painterly_sdxl_v1, scale {0..6}.
#   PARTE 2 — FLUX  : ritratto di 3 amici (bust shot, candid) × 2 seed × slider
#                     age, scale {0..4}, applicato sia _f (federico_v2, noxattn)
#                     sia _t (trial_v1, xattn).
#
# Output:
#   sdxl/tasks/baseline/outputs/painterly_landscape_global/seed{N}/scale_*.png
#   flux/tasks/baseline/outputs/age_3friends_global/age_t/<files>.png
#   flux/tasks/baseline/outputs/age_3friends_global/age_f/<files>.png
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARTE 1 — SDXL painterly landscape
# -----------------------------------------------------------------------------
(
    source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
    cd "$FERT_REPO"
    activate_sdxl_env

    export HF_HOME="$FERT_HF_CACHE"
    export HF_HUB_CACHE="$FERT_HF_CACHE/hub"

    SLIDER="sdxl/trained_sliders/sliders/painterly_sdxl_v1_alpha1.0_rank4_noxattn/painterly_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors"
    PROMPT="a mountain lake surrounded by pine forest, snow-capped peaks in the background, photorealistic, 4k"
    OUT_BASE="sdxl/tasks/baseline/outputs/painterly_landscape_global"
    mkdir -p sdxl/tasks/baseline/logs "$OUT_BASE"

    echo "================================================================"
    echo "=== PARTE 1: SDXL painterly landscape                       ==="
    echo "================================================================"
    nvidia-smi || true
    echo "Slider: $SLIDER"
    echo "Prompt: $PROMPT"
    echo ""

    for SEED in 42 7; do
        SAVE="${OUT_BASE}/seed${SEED}"
        mkdir -p "$SAVE"
        echo "--- SDXL seed=${SEED} ---"
        python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
            --prompt      "$PROMPT" \
            --slider      "$SLIDER" \
            --save_path   "$SAVE" \
            --scales      0 1 2 3 4 5 6 \
            --start_noise 700 \
            --seed        "$SEED" \
            --rank        4 \
            --alpha       1.0 \
            --steps       50
        echo ""
    done

    echo "=== PARTE 1 COMPLETATA ==="
    find "$OUT_BASE" -type f | sort
    echo ""
)

# -----------------------------------------------------------------------------
# PARTE 2 — FLUX age (3 amici, bust shot)
# -----------------------------------------------------------------------------
(
    source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
    cd "$FERT_REPO"
    activate_flux_env

    export HF_HOME="$FERT_HF_CACHE"
    export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    PROMPT="three friends posing together, bust shot, looking at camera, natural lighting, photorealistic, 4k, sharp focus"
    OUT_BASE="flux/tasks/baseline/outputs/age_3friends_global"
    SCALES="0 0.5 1.0 1.5 2.0 2.5 3.0 4.0"
    SEEDS_LIST="42 7"

    SLIDER_T="flux/trained_sliders/sliders/trial_v1/age_person_flux_v1/flux-age_person_flux_v1"
    SLIDER_F="flux/trained_sliders/sliders/federico_v2/flux-age_flux_v2_custom"

    mkdir -p flux/tasks/baseline/logs "$OUT_BASE/age_t" "$OUT_BASE/age_f"

    echo "================================================================"
    echo "=== PARTE 2: FLUX age 3 amici (bust shot)                   ==="
    echo "================================================================"
    nvidia-smi || true
    echo "Prompt: $PROMPT"
    echo ""

    # ---- slider _t (trial_v1, xattn) ----
    echo "--- FLUX age _t (trial_v1, xattn) ---"
    python -u flux/tasks/baseline/scripts/generate_flux_slider.py \
        --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
        --lora_dirs    "$SLIDER_T" \
        --lora_scales  $SCALES \
        --prompt       "$PROMPT" \
        --seeds        $SEEDS_LIST \
        --save_dir     "$OUT_BASE/age_t" \
        --height 1024 --width 1024 \
        --steps 50 --guidance_scale 3.5 \
        --max_sequence_length 512 \
        --rank 16 --alpha 1.0 --train_method xattn \
        --skip_slider_timestep_till 8 \
        --cuda_device 0
    echo ""

    # ---- slider _f (federico_v2, noxattn) ----
    echo "--- FLUX age _f (federico_v2, noxattn) ---"
    python -u flux/tasks/baseline/scripts/generate_flux_slider.py \
        --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
        --lora_dirs    "$SLIDER_F" \
        --lora_scales  $SCALES \
        --prompt       "$PROMPT" \
        --seeds        $SEEDS_LIST \
        --save_dir     "$OUT_BASE/age_f" \
        --height 1024 --width 1024 \
        --steps 50 --guidance_scale 3.5 \
        --max_sequence_length 512 \
        --rank 16 --alpha 1.0 --train_method noxattn \
        --skip_slider_timestep_till 8 \
        --cuda_device 0
    echo ""

    echo "=== PARTE 2 COMPLETATA ==="
    find "$OUT_BASE" -type f | sort
)

echo ""
echo "================================================================"
echo "=== TUTTO FATTO                                              ==="
echo "================================================================"
