#!/usr/bin/env python
# =============================================================================
# Generate images with Flux.1-dev (no slider, base model only).
#
# Uses the official diffusers FluxPipeline, bf16, with the Flux.1-dev
# default parameters (30 steps, guidance 3.5). The HF cache path follows
# the training environment.
#
# Usage:
#   python generate_flux.py \
#       --prompt "a smiling man next to a serious woman" \
#       --save_dir ../outputs/generations/smile_serious \
#       --seeds 42 7 2026 \
#       --steps 30
#
# Multiple prompts (pass --prompt several times):
#   python generate_flux.py \
#       --prompt "a smiling man" --prompt "a serious woman" \
#       --prompt "a smiling man and a serious woman" \
#       --save_dir ../outputs/generations/test \
#       --seeds 42 7
# =============================================================================

import os
import sys
import argparse
from pathlib import Path
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path",
                   default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--prompt", action="append", required=True,
                   help="Generation prompt. Pass --prompt several times for multi-prompt mode.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seed list: one image per (prompt, seed).")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--height", type=int, default=1024,
                   help="Flux.1-dev native resolution is 1024; use 512 for speed.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--cuda_device", default="0")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import torch

# -----------------------------------------------------------------------------
# Patch compat: torch 2.4 <-> diffusers 0.35+ (scaled_dot_product_attention)
# -----------------------------------------------------------------------------
# Stesso monkeypatch di train_flux_slider.py. diffusers passa enable_gqa=False
# a F.scaled_dot_product_attention, kwarg introdotto in torch 2.5. Flux usa
# MHA pura (no GQA), strippare il kwarg e' equivalente al default.
_torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _torch_ver < (2, 5):
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*sdpa_args, **sdpa_kwargs):
        sdpa_kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*sdpa_args, **sdpa_kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    print(f"[compat] torch {torch.__version__} < 2.5: monkeypatched "
          f"F.scaled_dot_product_attention per strippare enable_gqa kwarg")

from diffusers import FluxPipeline

# silenzia verbose
import transformers as _tf
_tf.logging.set_verbosity_error()
import diffusers as _df
_df.logging.set_verbosity_error()

save_dir = Path(args.save_dir).resolve()
save_dir.mkdir(parents=True, exist_ok=True)
print(f"[config] model       = {args.pretrained_model_name_or_path}")
print(f"[config] save_dir    = {save_dir}")
print(f"[config] prompts     = {len(args.prompt)}")
for i, p in enumerate(args.prompt):
    print(f"           [{i}] {p}")
print(f"[config] seeds       = {args.seeds}")
print(f"[config] resolution  = {args.width}x{args.height}")
print(f"[config] steps       = {args.steps}  guidance={args.guidance_scale}")
print()

# -----------------------------------------------------------------------------
# Load pipeline (bf16, local only)
# -----------------------------------------------------------------------------
print("=== Loading FluxPipeline ===")
t0 = time.time()
pipe = FluxPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.to("cuda:0")
print(f"[load] done in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    print(f"[mem post-load] allocated={torch.cuda.memory_allocated()/1e9:.2f} GB")

# -----------------------------------------------------------------------------
# Generation loop
# -----------------------------------------------------------------------------
print("\n=== Generating images ===")
t_total = time.time()
n_total = len(args.prompt) * len(args.seeds)
n_done = 0

for pi, prompt in enumerate(args.prompt):
    # nome file-safe dal prompt
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt)[:60]
    for seed in args.seeds:
        n_done += 1
        tag = f"p{pi}_{slug}__seed{seed}"
        out_path = save_dir / f"{tag}.png"
        if out_path.exists():
            print(f"  [{n_done}/{n_total}] SKIP (gia' esistente): {out_path.name}")
            continue
        gen = torch.Generator("cuda").manual_seed(seed)
        t_g = time.time()
        img = pipe(
            prompt=prompt,
            height=args.height, width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            max_sequence_length=args.max_sequence_length,
            generator=gen,
        ).images[0]
        img.save(out_path)
        dt = time.time() - t_g
        print(f"  [{n_done}/{n_total}] {out_path.name}  ({dt:.1f}s)")

print(f"\n=== Done in {(time.time() - t_total)/60:.1f} min ===")
print(f"Output in: {save_dir}")
