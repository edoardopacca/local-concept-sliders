#!/usr/bin/env python3
"""Masked LoRA/slider editing on a previously inverted real image.

Usage examples
--------------
# Basic SDXL masked edit
python scripts/edit_real_image_masked.py \
    --run_dir real_editing/runs/my_run_001 \
    --slider_path sdxl/trained_sliders/sliders/smiling.pt \
    --mask_name mask_target.png

# Stronger edit with pixel compositing
python scripts/edit_real_image_masked.py \
    --run_dir real_editing/runs/my_run_001 \
    --slider_path sdxl/trained_sliders/sliders/smiling.pt \
    --slider_scale 4.0 \
    --start_noise 800 \
    --mask_name mask_target.png \
    --pixel_composite \
    --output_name edited_strong.png

# SD1.4 edit with feathered mask
python scripts/edit_real_image_masked.py \
    --run_dir real_editing/runs/my_sd14_run \
    --slider_path sdxl/trained_sliders/sliders/smiling.pt \
    --mask_name mask_target.png \
    --feather_radius 8 \
    --dtype float32
"""

import argparse
import json
import sys
from pathlib import Path

# __file__ = .../sdxl/tasks/real_editing/scripts/edit_real_image_masked.py → parents[4] = repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from sdxl.tasks.real_editing.lib.models.loader import load_model_context
from sdxl.tasks.real_editing.lib.io.artifacts import load_inversion_artifacts, save_edit_artifacts
from sdxl.tasks.real_editing.lib.io.metrics import compute_metrics
from sdxl.tasks.real_editing.lib.editing.masked_editor import MaskedLoRAEditor, EditConfig
from sdxl.tasks.real_editing.lib.editing.blending import load_mask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Masked LoRA/slider editing on an inverted real image."
    )
    p.add_argument("--run_dir", type=Path, required=True,
                    help="Directory with inversion artifacts (from invert_real_image.py)")
    p.add_argument("--slider_path", type=str, required=True,
                    help="Path to LoRA slider .pt file")
    p.add_argument("--slider_scale", type=float, default=2.0)
    p.add_argument("--rank", type=int, default=4, help="LoRA rank (must match checkpoint)")
    p.add_argument("--start_noise", type=int, default=700,
                    help="Timestep threshold: LoRA applies when t <= start_noise")

    p.add_argument("--mask_name", type=str, default="mask_target.png")
    p.add_argument("--output_name", type=str, default="edited_target_only.png")

    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=None,
                    help="Override denoising steps (default: from inversion config)")
    p.add_argument("--seed", type=int, default=None,
                    help="Override seed (default: from inversion config)")

    p.add_argument("--img2img_strength", type=float, default=0.6,
                    help="Noise strength for img2img mode (0.0-1.0, used when no null-text embeddings)")
    p.add_argument("--feather_radius", type=int, default=0,
                    help="Gaussian blur radius for soft mask edges (0 = hard)")
    p.add_argument("--pixel_composite", action="store_true",
                    help="Composite edited region with original outside mask")
    p.add_argument("--skip_metrics", action="store_true")

    # Hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--local_files_only", action="store_true")

    # Model override (usually inferred from inversion metadata)
    p.add_argument("--model_type", type=str, default=None)
    p.add_argument("--model_id", type=str, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Load inversion artifacts ---
    inv_result = load_inversion_artifacts(args.run_dir)
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))

    model_type = args.model_type or inv_result.model_family or metadata.get("model_family", "sdxl")
    model_id = args.model_id or inv_result.model_id or metadata.get("model_id")
    steps = args.steps or (inv_result.config.steps if inv_result.config else int(metadata.get("steps", 50)))
    seed = args.seed if args.seed is not None else (inv_result.config.seed if inv_result.config else int(metadata.get("seed", 1234)))

    print(f"[INFO] Run dir:  {args.run_dir}")
    print(f"[INFO] Slider:   {args.slider_path} (scale={args.slider_scale})")
    print(f"[INFO] Model:    {model_type} ({model_id or 'default'})")
    print(f"[INFO] Backend:  {inv_result.inversion_backend} ({inv_result.backend_status})")

    # --- Load model ---
    model_ctx = load_model_context(
        model_type=model_type,
        model_id=model_id,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    # Re-encode text with CFG for editing (inversion may have used do_cfg=False)
    prompt = metadata.get("prompt", "")
    negative_prompt = metadata.get("negative_prompt", "")
    inv_result.text_condition = model_ctx.encode_text(
        prompt, negative_prompt, do_cfg=True
    )

    # --- Edit ---
    editor = MaskedLoRAEditor()
    edit_config = EditConfig(
        slider_scale=args.slider_scale,
        rank=args.rank,
        start_noise=args.start_noise,
        guidance_scale=args.guidance_scale,
        steps=steps,
        seed=seed,
        feather_radius=args.feather_radius,
        pixel_composite=args.pixel_composite,
        img2img_strength=args.img2img_strength,
    )

    mask_path = args.run_dir / args.mask_name
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}\n"
            "Run SAM segmentation first (maskedLORA_editing/02_segment_with_sam.py)."
        )

    edit_result = editor.run(
        model_ctx=model_ctx,
        inv_result=inv_result,
        mask_path=mask_path,
        slider_path=args.slider_path,
        config=edit_config,
    )

    # --- Metrics ---
    prompt = metadata.get("prompt", "")
    mask_tensor = load_mask(
        mask_path,
        torch.Size([1, 1, edit_result.edited_image.size[1], edit_result.edited_image.size[0]]),
        "cpu", torch.float32,
    )
    metrics = compute_metrics(
        original=inv_result.original_image,
        edited=edit_result.edited_image,
        mask=mask_tensor,
        prompt=prompt,
        device=args.device,
        reconstruction=inv_result.reconstruction_image,
        skip_metrics=args.skip_metrics,
    )

    # --- Save ---
    save_edit_artifacts(
        run_dir=args.run_dir,
        result=edit_result,
        output_name=args.output_name,
        metrics=metrics,
    )

    print(f"[DONE] Edited image: {args.run_dir / args.output_name}")


if __name__ == "__main__":
    main()
