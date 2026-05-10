#!/bin/bash
#SBATCH --job-name=flux_global_trial_v1
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# eval_global_trial_v1.slurm
# =============================================================================
# Stessa struttura di eval_global_4sliders.slurm (stessi prompt, stesse scale)
# ma con i pesi trial_v1 (xattn, rank 16).
# 4 slider × 8 scale × 2 prompt = 64 immagini totali.
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

SLIDERS_ROOT="$FERT_REPO/flux/trained_sliders/sliders/trial_v1"
AGE_DIR="${SLIDERS_ROOT}/age_person_flux_v1/flux-age_person_flux_v1"
SMILE_DIR="${SLIDERS_ROOT}/smile_person_flux_v1/flux-smile_person_flux_v1"
CURLY_DIR="${SLIDERS_ROOT}/curlyhair_person_flux_v1/flux-curlyhair_person_flux_v1"
PAINTERLY_DIR="${SLIDERS_ROOT}/painterly_flux_v1/flux-painterly_flux_v1"

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

echo "=================================================================="
echo "=== eval_global_trial_v1 : 4 slider × 8 scale × 2 prompt     ==="
echo "=================================================================="
nvidia-smi
echo ""

SAVE_AGE="$FERT_REPO/flux/tasks/baseline/outputs/trial_v1_age_global_${TS}"
mkdir -p "$SAVE_AGE"
echo "----------------------------------------"
echo "[1/4] AGE (xattn, rank 16)"
echo "      out: $SAVE_AGE"
echo "----------------------------------------"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$AGE_DIR" \
    --lora_scales $SCALES \
    --prompt "$PROMPT_2P" --prompt "$PROMPT_3P" \
    --seeds $SEED \
    --save_dir "$SAVE_AGE" \
    --height $HEIGHT --width $WIDTH \
    --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method xattn

SAVE_SMILE="$FERT_REPO/flux/tasks/baseline/outputs/trial_v1_smile_global_${TS}"
mkdir -p "$SAVE_SMILE"
echo ""
echo "----------------------------------------"
echo "[2/4] SMILE (xattn, rank 16)"
echo "      out: $SAVE_SMILE"
echo "----------------------------------------"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$SMILE_DIR" \
    --lora_scales $SCALES \
    --prompt "$PROMPT_2P_NEUTRAL" --prompt "$PROMPT_3P_NEUTRAL" \
    --seeds $SEED \
    --save_dir "$SAVE_SMILE" \
    --height $HEIGHT --width $WIDTH \
    --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method xattn

SAVE_CURLY="$FERT_REPO/flux/tasks/baseline/outputs/trial_v1_curly_global_${TS}"
mkdir -p "$SAVE_CURLY"
echo ""
echo "----------------------------------------"
echo "[3/4] CURLY (xattn, rank 16)"
echo "      out: $SAVE_CURLY"
echo "----------------------------------------"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$CURLY_DIR" \
    --lora_scales $SCALES \
    --prompt "$PROMPT_2P" --prompt "$PROMPT_3P" \
    --seeds $SEED \
    --save_dir "$SAVE_CURLY" \
    --height $HEIGHT --width $WIDTH \
    --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method xattn

SAVE_PAINTERLY="$FERT_REPO/flux/tasks/baseline/outputs/trial_v1_painterly_global_${TS}"
mkdir -p "$SAVE_PAINTERLY"
echo ""
echo "----------------------------------------"
echo "[4/4] PAINTERLY (xattn, rank 16)"
echo "      out: $SAVE_PAINTERLY"
echo "----------------------------------------"
python flux/tasks/baseline/scripts/generate_flux_slider.py \
    --lora_dirs "$PAINTERLY_DIR" \
    --lora_scales $SCALES \
    --prompt "$PROMPT_CITY" --prompt "$PROMPT_NATURE" \
    --seeds $SEED \
    --save_dir "$SAVE_PAINTERLY" \
    --height $HEIGHT --width $WIDTH \
    --steps $STEPS --guidance_scale $GUIDANCE \
    --rank 16 --alpha 1 --train_method xattn

echo ""
echo "=================================================================="
echo "=== DONE — outputs (16 imgs per dir, 64 totali):               ==="
echo "  age       : $SAVE_AGE"
echo "  smile     : $SAVE_SMILE"
echo "  curly     : $SAVE_CURLY"
echo "  painterly : $SAVE_PAINTERLY"
echo "=================================================================="
