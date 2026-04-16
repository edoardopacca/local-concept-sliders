#!/usr/bin/env python3
"""Invert a real image into diffusion latent space.

Usage examples
--------------
# SDXL + Tight Inversion (default, with IP-Adapter)
python scripts/invert_real_image.py \
    --image path/to/photo.jpg \
    --prompt "a photo of a person" \
    --run_id my_run_001

# SDXL + Tight Inversion without IP-Adapter
python scripts/invert_real_image.py \
    --image path/to/photo.jpg \
    --prompt "a photo of a person" \
    --run_id my_run_002 \
    --no_ipa

# SDXL + pure DDIM baseline
python scripts/invert_real_image.py \
    --image path/to/photo.jpg \
    --prompt "a photo of a person" \
    --run_id my_run_003 \
    --backend ddim

# SD1.4 + Null-Text Inversion
python scripts/invert_real_image.py \
    --image path/to/photo.jpg \
    --prompt "a photo of a person" \
    --run_id my_run_004 \
    --model_type sd1x \
    --backend null_text \
    --dtype float32
"""

import argparse
import os
import sys
from pathlib import Path

# __file__ = .../sdxl/tasks/real_editing/scripts/invert_real_image.py → parents[4] = repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from sdxl.tasks.real_editing.lib.models.loader import load_model_context
from sdxl.tasks.real_editing.lib.inversion.registry import get_backend, list_backends
from sdxl.tasks.real_editing.lib.inversion.base import InversionConfig
from sdxl.tasks.real_editing.lib.io.artifacts import save_inversion_artifacts
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Invert a real image into diffusion latent space."
    )
    p.add_argument("--image", type=Path, required=True, help="Path to input image")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt describing the image")
    p.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    p.add_argument("--run_id", type=str, required=True, help="Run identifier for output directory")
    p.add_argument("--output_root", type=Path, default=Path("sdxl/tasks/real_editing/outputs"))

    # Model
    p.add_argument("--model_type", type=str, default="sdxl", choices=["sdxl", "sd1x", "sd14", "sd15"])
    p.add_argument("--model_id", type=str, default=None, help="HuggingFace model id or local path")

    # Backend
    available = ", ".join(sorted(list_backends().keys()))
    p.add_argument("--backend", type=str, default="tight_inversion",
                    help=f"Inversion backend. Available: {available}")

    # Inversion params
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                    help="CFG scale during inversion (1.0 = no CFG, recommended for DDIM/Tight)")
    p.add_argument("--seed", type=int, default=1234)

    # Tight Inversion / GD
    p.add_argument("--num_gd_steps", type=int, default=0,
                    help="Gradient descent optimisation steps per timestep (Tight Inv)")
    p.add_argument("--gd_step_size", type=float, default=0.001)
    p.add_argument("--optimization_start", type=int, default=0,
                    help="Timestep threshold below which GD kicks in")

    # IP-Adapter
    p.add_argument("--no_ipa", action="store_true",
                    help="Disable IP-Adapter (even for tight_inversion)")
    p.add_argument("--ipa_scale", type=float, default=0.4,
                    help="IP-Adapter scale (0.3-0.5 typical)")

    # Null-Text
    p.add_argument("--num_inner_steps", type=int, default=10)
    p.add_argument("--early_stop_epsilon", type=float, default=1e-5)

    # Hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--local_files_only", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"[INFO] Image:   {args.image}")
    print(f"[INFO] Prompt:  {args.prompt}")
    print(f"[INFO] Backend: {args.backend}")
    print(f"[INFO] Model:   {args.model_type} ({args.model_id or 'default'})")

    # --- Load model ---
    model_ctx = load_model_context(
        model_type=args.model_type,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    # --- Build inversion config ---
    use_ipa = (not args.no_ipa) and args.backend == "tight_inversion"
    if args.backend == "tight_inversion" and not use_ipa:
        raise ValueError(
            "This workflow is configured for Tight + IPA only. "
            "Do not pass --no_ipa with backend=tight_inversion."
        )
    config = InversionConfig(
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_gd_steps=args.num_gd_steps,
        gd_step_size=args.gd_step_size,
        optimization_start=args.optimization_start,
        use_ipa=use_ipa,
        ipa_scale=args.ipa_scale,
        num_inner_steps=args.num_inner_steps,
        early_stop_epsilon=args.early_stop_epsilon,
    )

    # --- Run inversion ---
    backend = get_backend(args.backend)
    print(f"[INFO] Backend status: {backend.status}")

    image = Image.open(args.image).convert("RGB")
    result = backend.invert(
        model_ctx=model_ctx,
        image=image,
        prompt=args.prompt,
        config=config,
        negative_prompt=args.negative_prompt,
    )

    # --- Save ---
    run_dir = args.output_root / args.run_id
    save_inversion_artifacts(
        run_dir=run_dir,
        result=result,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        extra_meta={
            "image_path": str(args.image),
            "height": model_ctx.default_resolution()[0],
            "width": model_ctx.default_resolution()[1],
            "dtype": args.dtype,
        },
    )

    print(f"[DONE] Run directory: {run_dir}")
    print(f"       Backend: {result.inversion_backend} ({result.backend_status})")


if __name__ == "__main__":
    main()
