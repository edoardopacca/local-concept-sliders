"""
CLI wrapper that generates a grid of images (seeds x scales) with a single
Concept Slider, loading Flux only once.

Why this exists: running ``generate.py`` in a bash loop pays the ~2-3 min
Flux load cost (the model weighs ~24 GB) on every image, which makes an
8-image sweep about 3x slower than needed. This wrapper loads the pipeline
once and iterates internally over the (seed, scale) pairs.

Scope: single slider, single target. Multi-slider sweeps would be
N-dimensional and need a dedicated design; for that case call
``generate.py`` directly.

Example:
    python -m flux.tasks.shop_concept.scripts.sweep \\
        --slider_path path/to/slider.pt \\
        --target_prompt "man" \\
        --prompt "a man and a woman facing the camera" \\
        --seeds 42 123 \\
        --scales 0.0 0.3 0.7 1.0 \\
        --output_dir flux/tasks/shop_concept/outputs/sweep_xxx \\
        --height 512 --width 512 \\
        --num_inference_steps 30 \\
        --guidance_scale 3.5 \\
        --edit_start_step 8 \\
        --cache_dir flux/tasks/shop_concept/_peft_cache

Output: flux/tasks/shop_concept/outputs/sweep_xxx/seed<S>_scale<C>.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(_REPO_ROOT))
    from flux.tasks.shop_concept.lib.flux_real_pipeline import RealGenerationPipeline  # noqa: E402
    from flux.tasks.shop_concept.scripts.generate import (  # noqa: E402
        ensure_matching_lora_params,
        prepare_slider_as_safetensors,
    )
else:
    from ..lib.flux_real_pipeline import RealGenerationPipeline
    from .generate import (
        ensure_matching_lora_params,
        prepare_slider_as_safetensors,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Seed x scale sweep on a single slider, loading Flux only once."
    )

    # Flux model
    parser.add_argument(
        "--model_name", type=str, default="black-forest-labs/FLUX.1-dev"
    )
    parser.add_argument("--device", type=str, default="cuda")

    # Slider
    parser.add_argument(
        "--slider_path",
        type=str,
        required=True,
        help="Path to a single slider (.pt auto-converted or .safetensors PEFT).",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        required=True,
        help="Single target_prompt; must be an exact substring of --prompt.",
    )
    parser.add_argument("--prompt", type=str, required=True)

    # Sweep axes
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed list (axis 1 of the sweep).",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        required=True,
        help="Slider scale list (axis 2 of the sweep). 0.0 = slider off.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where seed<S>_scale<C>.png files are written.",
    )

    # Generation
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--edit_start_step", type=int, default=8)
    parser.add_argument("--lora_fill_rank", type=int, default=16)
    parser.add_argument(
        "--cache_dir", type=str, default="flux/tasks/shop_concept/_peft_cache"
    )

    # Debug / diagnostic
    parser.add_argument(
        "--dump_masks",
        action="store_true",
        help=(
            "When set, save the masks (soft + binary segmentation) used by "
            "the mask-aware blend as PNG next to each generated image. "
            "Useful for diagnosing cases in which the slider leaks outside "
            "the intended target region."
        ),
    )

    args = parser.parse_args()

    # ---- Validate ----
    if args.target_prompt not in args.prompt:
        raise ValueError(
            f"target_prompt '{args.target_prompt}' is not a literal substring "
            f"of --prompt; the mask-aware pipeline requires an exact match for "
            f"token slicing."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Flux (only once) ----
    print(f"[sweep] loading Flux pipeline from {args.model_name}")
    t0 = time.time()
    pipe = RealGenerationPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    print(f"[sweep] pipeline loaded in {time.time() - t0:.1f}s")

    # ---- Prepare slider (auto-convert .pt -> .safetensors PEFT if needed) ----
    print(f"[sweep] preparing slider: {args.slider_path}")
    slider_safetensors = prepare_slider_as_safetensors(
        args.slider_path, args.cache_dir
    )
    lora_dicts = [load_file(slider_safetensors)]
    lora_dicts = ensure_matching_lora_params(
        lora_dicts, rank=args.lora_fill_rank
    )
    pipe.load_lora_weights(lora_dicts[0])

    print("[sweep] registering transformer blocks (mask-aware)")
    pipe.register_transformer_blocks()

    # ---- Sweep loop ----
    total = len(args.seeds) * len(args.scales)
    count = 0

    print("=" * 60)
    print(f"[sweep] total images : {total}")
    print(f"[sweep] seeds        : {args.seeds}")
    print(f"[sweep] scales       : {args.scales}")
    print(f"[sweep] target       : '{args.target_prompt}'")
    print(f"[sweep] prompt       : '{args.prompt}'")
    print(f"[sweep] steps        : {args.num_inference_steps}  "
          f"edit_start: {args.edit_start_step}")
    print(f"[sweep] resolution   : {args.height}x{args.width}")
    print(f"[sweep] output_dir   : {out_dir}")
    print("=" * 60)

    t_sweep = time.time()
    for seed in args.seeds:
        for scale in args.scales:
            count += 1
            out_path = out_dir / f"seed{seed}_scale{scale}.png"

            print("-" * 60)
            print(f"[sweep] [{count}/{total}] seed={seed} scale={scale}")
            print(f"         out: {out_path}")
            print("-" * 60)

            generator = torch.Generator(device=args.device).manual_seed(seed)
            t_img = time.time()

            # When --dump_masks is set, pass an extensionless prefix path: the
            # pipeline appends _target{i}_{seg,soft}.png for each target.
            _mask_dump_path = (
                str(out_dir / f"seed{seed}_scale{scale}_mask")
                if args.dump_masks
                else None
            )

            result = pipe(
                prompt=args.prompt,
                target_prompt=[args.target_prompt],
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                height=args.height,
                width=args.width,
                generator=generator,
                edit_start_step=args.edit_start_step,
                target_lora_scales=[scale],
                mask_dump_path=_mask_dump_path,
            )
            dt_img = time.time() - t_img
            print(f"[sweep] image time: {dt_img:.1f}s")

            result.images[0].save(out_path)
            print(f"[sweep] saved -> {out_path}")

    dt_sweep = time.time() - t_sweep
    print("=" * 60)
    print(f"[sweep] DONE: {total} images in {dt_sweep:.1f}s "
          f"(avg {dt_sweep / total:.1f}s/img)")
    print(f"[sweep] output_dir: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
