#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from diffusers import StableDiffusionXLPipeline


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase 1 - Base image generation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--output_root", type=Path, default=Path("sdxl/tasks/masked_lora/outputs"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    return parser


def dtype_from_str(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


@torch.no_grad()
def generate_base_and_save(
    pipe: StableDiffusionXLPipeline,
    run_dir: Path,
    seed: int,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> Dict:
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_dtype = pipe.unet.dtype
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        width=width,
        dtype=latent_dtype,
        device=device,
        generator=generator,
    )
    init_latents = latents.detach().clone().cpu()

    # Use fully standard Diffusers call (same reliability as baseline scripts).
    # Do not inject custom latents here; this avoids accidental noisy outputs.
    # We DO pass an explicit generator freshly re-seeded from `seed` (the one
    # above was consumed by prepare_latents). With set_determinism this is
    # bit-identical to the previous behaviour (which relied on the CUDA
    # global state), but it makes the init noise contractually equal to the
    # one phase 3 reconstructs from metadata, so the "all-sliders-off" run
    # in phase 3 reproduces base.png bit-exact.
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    image.save(run_dir / "base.png")

    torch.save(init_latents, run_dir / "init_latents.pt")

    metadata = {
        "phase": "base_generation",
        "created_at": datetime.utcnow().isoformat(),
        "model_id": pipe.config.get("_name_or_path", "unknown"),
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": dict(pipe.scheduler.config),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    run_id = args.run_id or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = make_run_dir(args.output_root, run_id)
    set_determinism(args.seed)

    torch_dtype = dtype_from_str(args.dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=False)

    metadata = generate_base_and_save(
        pipe=pipe,
        run_dir=run_dir,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    print(f"[OK] Base generated in: {run_dir}")
    print(f"[OK] metadata.json seed={metadata['seed']} steps={metadata['steps']}")


if __name__ == "__main__":
    main()
