#!/bin/bash
#SBATCH --job-name=smile_man_strong_v3_yaml_trick
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

# v3_yaml_trick — preservation via YAML-level degeneration of Eq.7.
# Uses UPSTREAM train.py unchanged. All the preservation magic is in
# the prompts YAML: 10 woman entries where target=positive=unconditional=
# neutral, which makes Eq.7's enhance loss collapse to
#   MSE(eps_target_with_LoRA, eps_target_without_LoRA)
# automatically. No code fork needed.
#
# Same 1000 iterations, same cost per iteration as v2_guidance4 (single
# prompt_pair sampled per iter). Walltime 1h is generous — v2_guidance4
# ran in ~45 min with 10 entries; now we have 20 entries but each iter
# still touches exactly one entry so total cost is unchanged.
# Memory 16G is fine (same as v2_guidance4).

python sdxl/trained_sliders/training/scripts/train.py \
    --config_file sdxl/trained_sliders/training/configs/smile_man_strong_v3_yaml_trick.yaml \
    --rank 4 \
    --alpha 1.0
