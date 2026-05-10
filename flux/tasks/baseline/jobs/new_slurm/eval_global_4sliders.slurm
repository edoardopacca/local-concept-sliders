#!/bin/bash
#SBATCH --job-name=flux_global_4sliders
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=flux/tasks/baseline/logs/%x_%j.out
#SBATCH --error=flux/tasks/baseline/logs/%x_%j.err

# =============================================================================
# eval_global_4sliders.slurm
# =============================================================================
# Generazione GLOBALE (no mask) — INFERENZA, no training.
# 4 slider × 8 scale × 2 prompt, ognuno in cartella separata.
#
# 4 invocazioni separate (1 carico Flux per slider) per chiarezza output.
#
# Slider e prompt:
#   age       (symmetric, rank 16, xattn)   ← 2 prompt person
#   smile     (symmetric, rank 16, xattn)   ← 2 prompt person
#   curly     (custom v2, rank 16, noxattn) ← 2 prompt person
#   painterly (custom v2, rank 16, noxattn) ← 2 prompt landscape
#
# Scale: 0 0.5 1 1.5 2 3 4 5 (8 valori)
#
# Output: 4 dir, 16 imgs per dir = 64 imgs totali
#
# Compute: 4 carichi Flux (~7 min) + 64 img × ~12s a 1024 = ~50-57 min totali
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

# -----------------------------------------------------------------------------
# Slider paths (struttura: <slider_name>_rank<R>_<method>_alpha<A>/flux-<slider_name>/slider_0.pt)
# -----------------------------------------------------------------------------
SLIDERS_ROOT="flux/trained_sliders/sliders"
AGE_DIR="${SLIDERS_ROOT}/age_flux_v1_symmetric_rank16_xattn_alpha1/flux-age_flux_v1_symmetric"
SMILE_DIR="${SLIDERS_ROOT}/smile_flux_v5_symmetric_rank16_xattn_alpha1/flux-smile_flux_v5_symmetric"
CURLY_DIR="${SLIDERS_ROOT}/curly_hair_flux_v2_custom_rank16_noxattn_alpha1/flux-curly_hair_flux_v2_custom"
PAINTERLY_DIR="${SLIDERS_ROOT}/pittoric_style_flux_v2_custom_rank16_noxattn_alpha1/flux-pittoric_style_flux_v2_custom"

# -----------------------------------------------------------------------------
# Parametri comuni
# -----------------------------------------------------------------------------
SCALES="0 0.5 1 1.5 2 3 4 5"
SEED=777
HEIGHT=1024
WIDTH=1024
STEPS=30
GUIDANCE=3.5
TS=$(date +%Y%m%d_%H%M)

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
PROMPT_2P="two people facing the camera, photorealistic portrait, 4k"
PROMPT_3P="three people facing the camera, photorealistic portrait, 4k"
PROMPT_2P_NEUTRAL="two people facing the camera, neutral expression, photorealistic portrait, 4k"
PROMPT_3P_NEUTRAL="three people facing the camera, neutral expression, photorealistic portrait, 4k"
PROMPT_CITY="a city street with buildings, photorealistic, 4k"
PROMPT_NATURE="a panoramic landscape with mountains and valleys, photorealistic, 4k"

echo "=================================================================="
echo "=== eval_global_4sliders : 4 invocations separate              ==="
echo "=================================================================="
nvidia-smi
echo ""

# -----------------------------------------------------------------------------
# 1/4 — AGE (symmetric, xattn, rank 16) on person prompts
# -----------------------------------------------------------------------------
SAVE_AGE="flux/tasks/baseline/outputs/age_global_${TS}"
mkdir -p "$SAVE_AGE"
echo "----------------------------------------"
echo "[1/4] AGE on person prompts (xattn, rank 16)"
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

# -----------------------------------------------------------------------------
# 2/4 — SMILE (symmetric, xattn, rank 16) on person prompts
# -----------------------------------------------------------------------------
SAVE_SMILE="flux/tasks/baseline/outputs/smile_global_${TS}"
mkdir -p "$SAVE_SMILE"
echo ""
echo "----------------------------------------"
echo "[2/4] SMILE on person prompts (xattn, rank 16)"
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

# -----------------------------------------------------------------------------
# 3/4 — CURLY (custom v2, noxattn, rank 16) on person prompts
# -----------------------------------------------------------------------------
SAVE_CURLY="flux/tasks/baseline/outputs/curly_global_${TS}"
mkdir -p "$SAVE_CURLY"
echo ""
echo "----------------------------------------"
echo "[3/4] CURLY on person prompts (noxattn, rank 16)"
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
    --rank 16 --alpha 1 --train_method noxattn

# -----------------------------------------------------------------------------
# 4/4 — PAINTERLY (custom v2, noxattn, rank 16) on landscape prompts
# -----------------------------------------------------------------------------
SAVE_PAINTERLY="flux/tasks/baseline/outputs/painterly_global_${TS}"
mkdir -p "$SAVE_PAINTERLY"
echo ""
echo "----------------------------------------"
echo "[4/4] PAINTERLY on landscape prompts (noxattn, rank 16)"
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
    --rank 16 --alpha 1 --train_method noxattn

echo ""
echo "=================================================================="
echo "=== DONE — outputs (16 imgs per dir, 64 totali):               ==="
echo "  age       : $SAVE_AGE"
echo "  smile     : $SAVE_SMILE"
echo "  curly     : $SAVE_CURLY"
echo "  painterly : $SAVE_PAINTERLY"
echo "=================================================================="
