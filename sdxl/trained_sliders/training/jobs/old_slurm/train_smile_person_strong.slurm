#!/bin/bash
#SBATCH --job-name=smile_person_strong_train
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=sdxl/trained_sliders/training/logs/%x_%j.out
#SBATCH --error=sdxl/trained_sliders/training/logs/%x_%j.err

set -euo pipefail

source $SLURM_SUBMIT_DIR/tools/set_slurms.sh
cd $FERT_REPO
activate_sdxl_env

export HF_HOME=$FERT_HF_CACHE
export HF_HUB_CACHE=$FERT_HF_CACHE/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Baseline replica of Gandikota's smile slider, anchored to "person".
# 10 entries (5 ethnicities x M/F), identical to upstream recipe except the
# buggy `guidance: 4` key has been fixed to `guidance_scale: 1` (which matches
# the EFFECTIVE value upstream actually uses — see comments in the YAML).
# This is the reference slider against which smile_man_strong is compared.

python sdxl/trained_sliders/training/scripts/train.py \
    --config_file sdxl/trained_sliders/training/configs/smile_person_strong.yaml \
    --name smile_person_strong \
    --rank 4 \
    --alpha 1.0 \
    --device 0
