#!/bin/bash
#SBATCH --job-name=mlf_p1_two_subjects
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

# =============================================================================
# phase1_two_subjects.slurm — Phase 1: genera base image con 2 soggetti
# =============================================================================
# Stessa scena del sweep shop_concept (man + woman, studio portrait), ma
# via the masked_lora pipeline (manual SAM masking afterwards, then Van Gogh edit
# in phase3).
#
# Output: flux/tasks/masked_lora/outputs/<RUN_ID>/base.png + metadata.json
# =============================================================================

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_flux_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p flux/tasks/masked_lora/logs flux/tasks/masked_lora/outputs

PROMPT="a man and a woman facing the camera, studio portrait, neutral expression"
SEED=42
RUN_ID="two_subjects_masked_$(date +%Y%m%d_%H%M)"

echo "=============================================="
echo "=== masked_lora Flux: Phase 1 base image   ==="
echo "=============================================="
echo "prompt  : $PROMPT"
echo "seed    : $SEED"
echo "run_id  : $RUN_ID"
echo ""

python -u flux/tasks/masked_lora/scripts/01_generate_base.py \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --steps 30 \
    --guidance_scale 3.5 \
    --max_sequence_length 256 \
    --height 512 \
    --width 512 \
    --model_id black-forest-labs/FLUX.1-dev \
    --run_id "$RUN_ID" \
    --output_root flux/tasks/masked_lora/outputs \
    --device cuda

echo ""
echo "=== DONE phase1 ==="
ls -la "flux/tasks/masked_lora/outputs/$RUN_ID"
echo ""
echo "Next steps:"
echo "  1. On the local machine: ./tools/pull_from_hpc.sh"
echo "  2. Sul Mac (interattivo): python mask_SAM/segment_with_sam.py \\"
echo "       --run_dir flux/tasks/masked_lora/outputs/$RUN_ID \\"
echo "       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \\"
echo "       --image_name base.png --mode interactive"
echo "  3. On the local machine: ./tools/push_to_hpc.sh   (push the mask to HPC)"
echo "  4. Crea phase3 SLURM e lancialo (slider Van Gogh + RUN_ID)"
