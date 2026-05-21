#!/usr/bin/env python3
"""
Phase 1 of the external mask-guided pipeline on Flux: generate the base
image without any slider, and save the metadata needed to rerun the same
denoising trajectory in phase 3.

Outputs:
  - base.png        decoded image
  - metadata.json   seed, prompt, height, width, steps, scheduler_config, ...

The seed and scheduler config are reused in phase 3 so that the two passes
follow exactly the same denoising trajectory; this is what allows a clean
comparison between baseline and masked-edit on identical inputs.

Initial latents are NOT saved to disk: on Flux it is simpler to regenerate
them from the seed inside phase 3, which also avoids shape mismatches
between the packed and unpacked layouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

# Import flux.tasks.shop_concept BEFORE torch/diffusers so the SDPA
# compatibility shim is installed: diffusers >= 0.36 passes `enable_gqa`
# to F.scaled_dot_product_attention, but that kwarg only exists on
# torch >= 2.5. The shim (in shop_concept/__init__.py) drops it.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
import flux.tasks.shop_concept  # noqa: F401  (side-effect: install SDPA shim)

import torch  # noqa: E402
from diffusers import FluxPipeline  # noqa: E402


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase 1 - Flux base image generation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--model_id", type=str, default="black-forest-labs/FLUX.1-dev"
    )
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("flux/tasks/masked_lora/outputs"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


@torch.no_grad()
def generate_base_and_save(
    pipe: FluxPipeline,
    run_dir: Path,
    seed: int,
    prompt: str,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    height: int,
    width: int,
) -> Dict:
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    image.save(run_dir / "base.png")

    metadata = {
        "phase": "base_generation",
        "created_at": datetime.utcnow().isoformat(),
        "model_id": pipe.config.get("_name_or_path", "black-forest-labs/FLUX.1-dev"),
        "seed": seed,
        "prompt": prompt,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": dict(pipe.scheduler.config),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    run_id = (
        args.run_id
        or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(args.seed)

    print(f"[phase1] loading Flux pipeline: {args.model_id}")
    pipe = FluxPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    ).to(args.device)

    metadata = generate_base_and_save(
        pipe=pipe,
        run_dir=run_dir,
        seed=args.seed,
        prompt=args.prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        height=args.height,
        width=args.width,
    )
    print(f"[OK] Base generated in: {run_dir}")
    print(
        f"[OK] metadata.json seed={metadata['seed']} "
        f"steps={metadata['steps']} res={metadata['height']}x{metadata['width']}"
    )


if __name__ == "__main__":
    main()
