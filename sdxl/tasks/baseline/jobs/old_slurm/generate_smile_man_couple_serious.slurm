#!/bin/bash
#SBATCH --job-name=smile_couple_srs
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=sdxl/tasks/baseline/logs/%x_%j.out
#SBATCH --error=sdxl/tasks/baseline/logs/%x_%j.err

set -euo pipefail
source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env
export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python sdxl/tasks/baseline/scripts/generate_with_sliders.py \
    --prompt "a portrait photo of a serious man and a serious woman with neutral expression, shoulders up, looking at the camera" \
    --slider sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/smile_man_alpha1.0_rank4_noxattn_last.safetensors \
    --save_path sdxl/trained_sliders/training/outputs/smile_man_alpha1.0_rank4_noxattn/generations_couple_serious \
    --scales -3 -1.5 0 1.5 3 \
    --start_noise 700 \
    --seed 42 \
    --rank 4 \
    --alpha 1.0 \
    --steps 50
