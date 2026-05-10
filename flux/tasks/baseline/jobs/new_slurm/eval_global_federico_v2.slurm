#!/bin/bash
#SBATCH --job-name=eval_global_fed_v2
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# eval_global_federico_v2.slurm
# =============================================================================
# Generazione GLOBALE (no mask) con i 6 slider di Federico (v2_custom).
# 6 slider × 8 scale × 2 prompt = 96 immagini totali.
#
# Tutti noxattn, rank 16, alpha 1.
#
# CyberDuck: mettere Sliders-flux/ su HPC come:
#   $FERT_REPO/flux/trained_sliders/sliders/federico_v2/
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

mkdir -p flux/tasks/baseline/logs

SLIDERS_ROOT="$FERT_REPO/flux/trained_sliders/sliders/federico_v2"
AGE_DIR="${SLIDERS_ROOT}/flux-age_flux_v2_custom"
CURLY_DIR="${SLIDERS_ROOT}/flux-curly_hair_flux_v2_custom"
FUR_DIR="${SLIDERS_ROOT}/flux-fur_flux_v2_custom"
NIGHT_DIR="${SLIDERS_ROOT}/flux-night_sky_flux_v2_custom"
PAINT_DIR="${SLIDERS_ROOT}/flux-pittoric_style_flux_v2_custom"
SMILE_DIR="${SLIDERS_ROOT}/flux-smile_flux_v2_custom"

SCALES="0 0.5 1 1.5 2 3 4 5"
SEED=777
HEIGHT=1024
WIDTH=1024
STEPS=30
GUIDANCE=3.5
TS=$(date +%Y%m%d_%H%M)

PROMPT_2P="two people facing the camera, photorealistic portrait, 4k"
PROMPT_3P="three people facing the camera, photorealistic portrait, 4k"
PROMPT_2P_NEUTRAL="two people facing the camera, neutral expression, photorealistic portrait, 4k"
PROMPT_3P_NEUTRAL="three people facing the camera, neutral expression, photorealistic portrait, 4k"
PROMPT_CITY="a city street with buildings, photorealistic, 4k"
PROMPT_NATURE="a panoramic landscape with mountains and valleys, photorealistic, 4k"
PROMPT_PETS="two pets in a garden, photorealistic, 4k"
PROMPT_SKY="a panoramic landscape at dusk, photorealistic, 4k"

echo "=================================================================="
echo "=== eval_global_federico_v2 : 6 slider × 8 scale × 2 prompt  ==="
echo "=================================================================="
nvidia-smi
echo ""

# 1/6 — AGE
SAVE="$FERT_REPO/flux/tasks/baseline/outputs/fed_v2_age_global_${TS}"
mkdir -p "$SAVE"
echo "[1/6] AGE  →  $SAVE"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$AGE_DIR" --lora_scales $SCALES \
    --prompt "$PROMPT_2P" --prompt "$PROMPT_3P" \
    --seeds $SEED --save_dir "$SAVE" \
    --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method noxattn --skip_slider_timestep_till 8

# 2/6 — CURLY HAIR
SAVE="$FERT_REPO/flux/tasks/baseline/outputs/fed_v2_curly_global_${TS}"
mkdir -p "$SAVE"
echo ""
echo "[2/6] CURLY HAIR  →  $SAVE"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$CURLY_DIR" --lora_scales $SCALES \
    --prompt "$PROMPT_2P" --prompt "$PROMPT_3P" \
    --seeds $SEED --save_dir "$SAVE" \
    --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method noxattn --skip_slider_timestep_till 8

# 3/6 — FUR LENGTH
SAVE="$FERT_REPO/flux/tasks/baseline/outputs/fed_v2_fur_global_${TS}"
mkdir -p "$SAVE"
echo ""
echo "[3/6] FUR LENGTH  →  $SAVE"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$FUR_DIR" --lora_scales $SCALES \
    --prompt "$PROMPT_PETS" --prompt "$PROMPT_NATURE" \
    --seeds $SEED --save_dir "$SAVE" \
    --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method noxattn --skip_slider_timestep_till 8

# 4/6 — NIGHT SKY
SAVE="$FERT_REPO/flux/tasks/baseline/outputs/fed_v2_nightsky_global_${TS}"
mkdir -p "$SAVE"
echo ""
echo "[4/6] NIGHT SKY  →  $SAVE"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$NIGHT_DIR" --lora_scales $SCALES \
    --prompt "$PROMPT_CITY" --prompt "$PROMPT_SKY" \
    --seeds $SEED --save_dir "$SAVE" \
    --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method noxattn --skip_slider_timestep_till 8

# 5/6 — PAINTERLY
SAVE="$FERT_REPO/flux/tasks/baseline/outputs/fed_v2_painterly_global_${TS}"
mkdir -p "$SAVE"
echo ""
echo "[5/6] PAINTERLY  →  $SAVE"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$PAINT_DIR" --lora_scales $SCALES \
    --prompt "$PROMPT_CITY" --prompt "$PROMPT_NATURE" \
    --seeds $SEED --save_dir "$SAVE" \
    --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method noxattn --skip_slider_timestep_till 8

# 6/6 — SMILE
SAVE="$FERT_REPO/flux/tasks/baseline/outputs/fed_v2_smile_global_${TS}"
mkdir -p "$SAVE"
echo ""
echo "[6/6] SMILE  →  $SAVE"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$SMILE_DIR" --lora_scales $SCALES \
    --prompt "$PROMPT_2P_NEUTRAL" --prompt "$PROMPT_3P_NEUTRAL" \
    --seeds $SEED --save_dir "$SAVE" \
    --height $HEIGHT --width $WIDTH --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method noxattn --skip_slider_timestep_till 8

echo ""
echo "=================================================================="
echo "=== DONE — output in flux/tasks/baseline/outputs/fed_v2_*     ==="
echo "=================================================================="
