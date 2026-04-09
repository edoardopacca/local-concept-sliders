#!/bin/bash
#SBATCH --job-name=smile_man_strong_v2_guidance4_train
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

# Controlled ablation vs train_smile_man_strong_v2.slurm. The *only* variable
# changed between the two training runs is the per-entry guidance_scale
# declared in the prompts YAML (1 in v2, 4 in v2_guidance4). Same preservation
# set (10 entries), same network, same optimizer, same number of iterations.
#
# If the resulting slider is stronger without artefacts → Gandikota's written
# intent of gs=4 was correct and the bug-at-effective-1.0 was a lucky accident
# that left strength on the table. If instead we reproduce the asymmetric /
# off-manifold behaviour seen in our first training attempt, that confirms
# gs=1 is the right regime for this prompt family regardless of entry count.
python sdxl/trained_sliders/training/scripts/train.py \
    --config_file sdxl/trained_sliders/training/configs/smile_man_strong_v2_guidance4.yaml \
    --name smile_man_strong_v2_guidance4 \
    --rank 4 \
    --alpha 1.0 \
    --device 0
