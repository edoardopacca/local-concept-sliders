#!/usr/bin/env python
# =============================================================================
# generate_flux_slider.py
# =============================================================================
# Generazione immagini con Flux.1-dev + LoRA concept slider applicata.
#
# Supporta il pattern di validazione "panel" multi-checkpoint + multi-scale:
# carica Flux UNA volta, poi per ogni (checkpoint, scale, prompt, seed)
# ricarica solo i pesi LoRA e genera. Memory-efficient, pipe loaded once.
#
# Uso tipico (panel multi-checkpoint + multi-scale):
#   python generate_flux_slider.py \
#       --lora_dirs .../smile_flux_v1_custom_rank16_xattn_alpha1/flux-smile_flux_v1_custom_step200 \
#                   .../smile_flux_v1_custom_rank16_xattn_alpha1/flux-smile_flux_v1_custom_step300 \
#                   .../smile_flux_v1_custom_rank16_xattn_alpha1/flux-smile_flux_v1_custom_step400 \
#                   .../smile_flux_v1_custom_rank16_xattn_alpha1/flux-smile_flux_v1_custom \
#       --lora_scales 0.0 0.5 1.0 1.5 \
#       --prompt "a photo of a person" \
#       --seeds 42 7 \
#       --save_dir .../outputs/generations/smile_flux_v1_panel \
#       --rank 16 --alpha 1.0 --train_method xattn
#
# Output: PNG con naming "ckpt{NAME}__scale{X.XX}__seed{N}__p{I}_{slug}.png"
# Naming permette sort naturale per ispezione visiva a griglia.
#
# Note: --lora_scales 0.0 disabilita la LoRA (baseline di confronto).
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
    p.add_argument("--lora_dirs", nargs="+", required=True,
                   help="Directories containing slider_0.pt; one per checkpoint.")
    p.add_argument("--lora_scales", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5],
                   help="Scales to sweep (0.0 = LoRA disabled = baseline).")
    p.add_argument("--prompt", action="append", required=True,
                   help="Prompt. May be passed several times for multi-prompt mode.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds (same set across checkpoint/scale combinations).")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--height", type=int, default=512,
                   help="512 for speed (default), 1024 for quality.")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    # LoRA config (must match the training-time config).
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train_method", default="xattn",
                   choices=["xattn", "noxattn", "full"])
    p.add_argument("--cuda_device", default="0")
    p.add_argument("--skip_slider_timestep_till", type=int, default=8,
                   help="Step threshold from which the slider becomes active "
                        "during denoising: LoRA off for i <= threshold, on for "
                        "i > threshold. Default 8 (aligned with shop_concept "
                        "and masked_lora for fair comparisons). The original "
                        "Concept Sliders setting is 0 (LoRA active from step 1).")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import torch

# -----------------------------------------------------------------------------
# Compatibility patch: torch 2.4 <-> diffusers 0.35+ (scaled_dot_product_attention)
# -----------------------------------------------------------------------------
_torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _torch_ver < (2, 5):
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*sdpa_args, **sdpa_kwargs):
        sdpa_kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*sdpa_args, **sdpa_kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    print(f"[compat] torch {torch.__version__} < 2.5: monkeypatched "
          f"F.scaled_dot_product_attention to drop the enable_gqa kwarg")

print(f"[path] CWD = {os.getcwd()}")

from flux.core.custom_flux_pipeline import FluxPipeline
from flux.core.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
)

# Silence verbose logs
import transformers as _tf
_tf.logging.set_verbosity_error()
import diffusers as _df
_df.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Config print
# -----------------------------------------------------------------------------
save_dir = Path(args.save_dir).resolve()
save_dir.mkdir(parents=True, exist_ok=True)

# Validate that every lora_dir exists and contains slider_0.pt.
lora_paths = []
for d in args.lora_dirs:
    p = Path(d) / "slider_0.pt"
    if not p.exists():
        sys.exit(f"[ERROR] Non trovato: {p}")
    lora_paths.append(p)

print(f"[config] model        = {args.pretrained_model_name_or_path}")
print(f"[config] save_dir     = {save_dir}")
print(f"[config] lora_dirs    = {len(lora_paths)}")
for i, p in enumerate(lora_paths):
    print(f"           [{i}] {p.parent.name} -> {p.name}")
print(f"[config] lora_scales  = {args.lora_scales}")
print(f"[config] rank/alpha   = {args.rank} / {args.alpha}  train_method={args.train_method}")
print(f"[config] prompts      = {len(args.prompt)}")
for i, pp in enumerate(args.prompt):
    print(f"           [{i}] {pp}")
print(f"[config] seeds        = {args.seeds}")
print(f"[config] resolution   = {args.width}x{args.height}")
print(f"[config] steps/cfg    = {args.steps} / {args.guidance_scale}")
print(f"[config] skip_slider  = {args.skip_slider_timestep_till} "
      f"(LoRA on per i > {args.skip_slider_timestep_till})")
n_total = len(lora_paths) * len(args.lora_scales) * len(args.prompt) * len(args.seeds)
print(f"[config] total images = {n_total}")
print()

# -----------------------------------------------------------------------------
# Load pipeline ONCE (bf16, local only)
# -----------------------------------------------------------------------------
print("=== Loading FluxPipeline ===")
t0 = time.time()
pipe = FluxPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.to("cuda:0")
pipe.set_progress_bar_config(disable=True)
print(f"[load] done in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    print(f"[mem post-load] allocated={torch.cuda.memory_allocated()/1e9:.2f} GB")

# -----------------------------------------------------------------------------
# Crea LoRANetwork ONCE (wrapping dei moduli in-place sul transformer).
# Poi per ogni checkpoint ri-carichiamo solo i pesi via load_state_dict.
# -----------------------------------------------------------------------------
print("\n=== Creating LoRANetwork wrapper ===")
net = LoRANetwork(
    pipe.transformer,
    rank=args.rank, multiplier=1.0, alpha=args.alpha,
    train_method=args.train_method,
).to("cuda:0", dtype=torch.bfloat16)
print(f"[lora] wrapper created (will be repopulated for every checkpoint)")

# === BUGFIX: clean baseline at scale=0.0 ===
# LoRAModule.__init__ sets multiplier=1.0 by default, and apply_to()
# already monkey-patches the .forward of the target modules. Without
# zeroing the multiplier here, the first pipe call with scale=0.0
# (use_network=None) would still run the LoRA at multiplier 1.0,
# producing a "baseline" image polluted by the loaded LoRA. The
# `with network:` context manager in custom_flux_pipeline resets the
# multiplier to 0 inside __exit__, but it is never entered when
# scale=0.0. We zero it explicitly here; for scale>0 the context
# manager raises it to the correct value and zeroes it again on exit.
for _m in net.unet_loras:
    _m.multiplier = 0
print(f"[lora] multiplier forced to 0 (clean baseline at scale=0.0)")

# -----------------------------------------------------------------------------
# Generation loop: per ogni checkpoint, carica pesi -> per ogni scale -> gen
# -----------------------------------------------------------------------------
print(f"\n=== Generating {n_total} images ===")
t_total = time.time()
n_done = 0

for ci, lora_path in enumerate(lora_paths):
    ckpt_tag = lora_path.parent.name
    # Extract the "step100" / "step200" / ... suffix, or fall back to "final".
    if "_step" in ckpt_tag:
        ckpt_short = "step" + ckpt_tag.split("_step")[-1]
    else:
        ckpt_short = "final"

    print(f"\n--- [{ci+1}/{len(lora_paths)}] Loading LoRA weights: {ckpt_tag} ---")
    # torch.load -> load_state_dict on the LoRA wrapper.
    state = torch.load(str(lora_path), map_location="cuda:0")
    net.load_state_dict(state)
    print(f"[lora] loaded {lora_path.name}")

    for scale in args.lora_scales:
        # scale=0 -> baseline: do NOT pass the network to the pipe (LoRA disabled).
        # scale>0 -> set_lora_slider(scale) + pass network=net:
        #           the custom pipeline executes `with network:`, which
        #           propagates the multiplier to the wrappers (see
        #           LoRANetwork.__enter__).
        if scale == 0.0:
            use_network = None
        else:
            net.set_lora_slider(float(scale))
            use_network = net

        for pi, prompt in enumerate(args.prompt):
            slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt)[:40]
            for seed in args.seeds:
                n_done += 1
                tag = f"ckpt{ckpt_short}__scale{scale:.2f}__seed{seed}__p{pi}_{slug}"
                out_path = save_dir / f"{tag}.png"
                if out_path.exists():
                    print(f"  [{n_done}/{n_total}] SKIP: {out_path.name}")
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
                    network=use_network,
                    skip_slider_timestep_till=args.skip_slider_timestep_till,
                ).images[0]
                img.save(out_path)
                dt = time.time() - t_g
                print(f"  [{n_done}/{n_total}] {out_path.name}  ({dt:.1f}s)")

print(f"\n=== Done in {(time.time() - t_total)/60:.1f} min ===")
print(f"Output in: {save_dir}")
print(f"\nTip: per vedere rapidamente la griglia, ordina per nome e scorri:")
print(f"  ls {save_dir}/*.png | sort")
