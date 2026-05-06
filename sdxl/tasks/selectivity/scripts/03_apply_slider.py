#!/usr/bin/env python3
"""
03_apply_slider.py — applica un LoRA slider GLOBALMENTE (senza maschera spaziale).

Usato nel task selectivity per testare se uno slider subject-specific modifica
selettivamente il soggetto target senza toccare il soggetto non-target.
Le metriche regionali (target vs non-target) sono calcolate separatamente da
eval_selectivity.py, che legge mask_target.png e mask_nontarget.png.

Usage:
  python sdxl/tasks/selectivity/scripts/03_apply_slider.py \
      --run_dir  sdxl/tasks/selectivity/runs/eval_age_01 \
      --slider_path sdxl/trained_sliders/sliders/age_woman_sdxl_v1_alpha1.0_rank4_noxattn/age_woman_sdxl_v1_alpha1.0_rank4_noxattn_last.safetensors \
      --slider_scales 0.5 1.0 1.5 \
      --output_prefix edited_age_specific \
      --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# __file__ = .../sdxl/tasks/selectivity/scripts/03_apply_slider.py → parents[4] = repo root
REPO_ROOT = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO_ROOT)

from sdxl.core.lora import (  # noqa: E402
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
    LoRANetwork,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Phase 3 selectivity — global slider application")
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--slider_path", type=str, required=True)
    p.add_argument("--slider_scales", type=float, nargs="+", default=[1.0],
                   help="One or more slider scales to apply (model loaded once)")
    p.add_argument("--output_prefix", type=str, required=True,
                   help="Prefix for output filenames, e.g. 'edited_age_specific'")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--start_noise", type=int, default=750,
                   help="Timestep above which LoRA is NOT applied (same as masked_lora)")
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"],
                   default="float16")
    return p


def dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict_noise(
    pipe, network, latents, t,
    prompt_embeds, add_text_embeds, add_time_ids,
    guidance_scale: float, lora_scale: float,
) -> torch.Tensor:
    network.set_lora_slider(scale=lora_scale)
    do_cfg = guidance_scale > 1.0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    with network:
        noise_pred = pipe.unet(
            latent_model_input, t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    if do_cfg:
        uncond, cond = noise_pred.chunk(2)
        noise_pred = uncond + guidance_scale * (cond - uncond)
    return noise_pred


def run_denoising(pipe, network, latents, timesteps, prompt_embeds,
                  add_text_embeds, add_time_ids, guidance_scale: float,
                  slider_scale: float, start_noise: int,
                  extra_step_kwargs: Dict) -> torch.Tensor:
    """Denoising loop senza maschera spaziale: LoRA applicato globalmente."""
    with torch.no_grad():
        for t in timesteps:
            if int(t.item()) > start_noise:
                eps = predict_noise(pipe, network, latents, t,
                                    prompt_embeds, add_text_embeds, add_time_ids,
                                    guidance_scale, lora_scale=0.0)
            else:
                eps = predict_noise(pipe, network, latents, t,
                                    prompt_embeds, add_text_embeds, add_time_ids,
                                    guidance_scale, lora_scale=slider_scale)
            latents = pipe.scheduler.step(eps, t, latents,
                                          **extra_step_kwargs, return_dict=False)[0]
    return latents


def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        needs_upcasting = (pipe.vae.dtype == torch.float16 and
                           getattr(pipe.vae.config, "force_upcast", False))
        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def main() -> None:
    args = build_parser().parse_args()

    metadata_path = args.run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    set_determinism(int(metadata["seed"]))
    torch.set_grad_enabled(False)

    device = args.device
    device_obj = torch.device(device)
    torch_dtype = dtype_from_str(args.dtype)

    resolved_model_id = (args.model_id or metadata.get("model_id") or
                         "stabilityai/stable-diffusion-xl-base-1.0")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        resolved_model_id, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=False)
    if "scheduler_config" in metadata:
        pipe.scheduler = pipe.scheduler.from_config(metadata["scheduler_config"])

    for module_name in UNET_TARGET_REPLACE_MODULE_CONV:
        if module_name not in DEFAULT_TARGET_REPLACE:
            DEFAULT_TARGET_REPLACE.append(module_name)

    network = LoRANetwork(pipe.unet, rank=args.rank, multiplier=1.0,
                          alpha=1.0, train_method="noxattn").to(device, dtype=torch_dtype)
    if str(args.slider_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        network.load_state_dict(load_file(str(args.slider_path)))
    else:
        network.load_state_dict(torch.load(args.slider_path, map_location=device))

    prompt = metadata["prompt"]
    negative_prompt = metadata.get("negative_prompt", "")
    guidance_scale = float(metadata["guidance_scale"])
    steps = int(metadata["steps"])
    height = int(metadata["height"])
    width = int(metadata["width"])
    do_cfg = guidance_scale > 1.0

    prompt_embeds, neg_embeds, pooled_embeds, neg_pooled = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt, negative_prompt_2=None,
    )
    add_text_embeds = pooled_embeds
    proj_dim = (int(pooled_embeds.shape[-1]) if getattr(pipe, "text_encoder_2", None) is None
                else pipe.text_encoder_2.config.projection_dim)
    add_time_ids = pipe._get_add_time_ids(
        (height, width), (0, 0), (height, width),
        dtype=prompt_embeds.dtype, text_encoder_projection_dim=proj_dim,
    )
    if do_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds])
        add_text_embeds = torch.cat([neg_pooled, add_text_embeds])
        add_time_ids = torch.cat([add_time_ids, add_time_ids])
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

    for scale in args.slider_scales:
        # Reset scheduler state (step_index) for each scale to avoid out-of-bounds
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        print(f"\n  [scale={scale}]  {args.run_dir.name}")
        # Ricrea i latent iniziali dallo stesso seed (same as 03_masked_edit.py)
        generator = torch.Generator(device=device_obj).manual_seed(int(metadata["seed"]))
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=height, width=width,
            dtype=pipe.unet.dtype, device=device_obj, generator=generator,
        )

        latents = run_denoising(
            pipe, network, latents, timesteps,
            prompt_embeds, add_text_embeds, add_time_ids,
            guidance_scale, scale, args.start_noise, extra_step_kwargs,
        )
        edited_pil = decode_latents(pipe, latents)

        out_name = f"{args.output_prefix}_s{scale:.1f}.png"
        edited_pil.save(args.run_dir / out_name)
        print(f"  [OK] → {out_name}")

        edit_meta = {
            "phase": "apply_slider",
            "output_image": out_name,
            "slider_path": args.slider_path,
            "slider_scale": scale,
            "start_noise": args.start_noise,
            "spatial_mask": "none (global application)",
        }
        (args.run_dir / f"edit_meta_{args.output_prefix}_s{scale:.1f}.json").write_text(
            json.dumps(edit_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
