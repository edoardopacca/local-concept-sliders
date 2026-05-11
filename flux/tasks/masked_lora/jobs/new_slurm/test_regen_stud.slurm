#!/bin/bash
#SBATCH --job-name=test_regen
#SBATCH --account=3377169
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=flux/tasks/masked_lora/logs/%x_%j.out
#SBATCH --error=flux/tasks/masked_lora/logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/tools/set_slurms.sh"
cd "$FERT_REPO"
activate_flux_env

export HF_HOME="$FERT_HF_CACHE"
export HF_HUB_CACHE="$FERT_HF_CACHE/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p flux/tasks/masked_lora/logs

python - <<'EOF'
import sys, torch
from pathlib import Path
from diffusers import FluxPipeline

sys.path.insert(0, str(Path(".").resolve()))
import flux.tasks.shop_concept  # SDPA shim

RUNS = [
    {"run_dir": "flux/tasks/masked_lora/runs/eval_age_person_01",
     "seed": 1001, "prompt": "two women, photorealistic portrait, 4k"},
    {"run_dir": "flux/tasks/masked_lora/runs/eval_age_person_10",
     "seed": 1010, "prompt": "an indian woman and a man, photorealistic portrait, 4k"},
]

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

for r in RUNS:
    run_dir = Path(r["run_dir"])
    print(f"[{run_dir.name}] seed={r['seed']}")
    g = torch.Generator(device=pipe._execution_device).manual_seed(r["seed"])
    img = pipe(prompt=r["prompt"], num_inference_steps=30, guidance_scale=3.5,
               max_sequence_length=256, height=1024, width=1024, generator=g).images[0]
    img.save(run_dir / "base_regen_stud.png")
    print(f"  -> {run_dir / 'base_regen_stud.png'}")

print("DONE")
EOF
