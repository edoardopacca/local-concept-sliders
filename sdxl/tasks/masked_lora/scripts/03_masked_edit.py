#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPModel, CLIPProcessor

# __file__ = .../sdxl/tasks/masked_lora/scripts/03_masked_edit.py → parents[4] = repo root
REPO_ROOT = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO_ROOT)

from sdxl.core.lora import (  # noqa: E402
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
    LoRANetwork,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase 3 - Masked LoRA edit")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--slider_path", type=str, required=True)
    parser.add_argument("--slider_scale", type=float, default=2.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--start_noise", type=int, default=700)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output_name", type=str, default="edited.png")
    parser.add_argument("--mask_name", type=str, default="mask.png")
    parser.add_argument("--skip_metrics", action="store_true")
    return parser


def dtype_from_str(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_run_inputs(run_dir: Path, mask_name: str) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    metadata_path = run_dir / "metadata.json"
    init_latents_path = run_dir / "init_latents.pt"
    mask_path = run_dir / mask_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    if not init_latents_path.exists():
        raise FileNotFoundError(f"Missing {init_latents_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing {mask_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    init_latents = torch.load(init_latents_path, map_location="cpu")
    mask_img = Image.open(mask_path).convert("L")
    mask_np = (np.array(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    mask_tensor = torch.from_numpy(mask_np)[None, None, ...]
    return metadata, init_latents, mask_tensor


def predict_noise(
    pipe: StableDiffusionXLPipeline,
    network: LoRANetwork,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    add_text_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
    lora_scale: float,
) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0
    network.set_lora_slider(scale=lora_scale)
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    with network:
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def compute_clip_similarity(clip_model, clip_processor, image: Image.Image, text: str, device: str) -> float:
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
        image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_emb * text_emb).sum(dim=-1).item())


def main() -> None:
    args = build_parser().parse_args()
    metadata, init_latents_cpu, mask_img_tensor = load_run_inputs(args.run_dir, args.mask_name)
    set_determinism(int(metadata["seed"]))
    torch.set_grad_enabled(False)

    device = args.device
    device_obj = torch.device(device)
    torch_dtype = dtype_from_str(args.dtype)
    resolved_model_id = args.model_id or metadata.get("model_id") or "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(resolved_model_id, torch_dtype=torch_dtype).to(device)
    pipe.set_progress_bar_config(disable=False)
    # Rebuild scheduler from saved config to keep the same denoising trajectory contract.
    if "scheduler_config" in metadata:
        pipe.scheduler = pipe.scheduler.from_config(metadata["scheduler_config"])

    # Match baseline training/inference setup: include conv targets in LoRA patching.
    for module_name in UNET_TARGET_REPLACE_MODULE_CONV:
        if module_name not in DEFAULT_TARGET_REPLACE:
            DEFAULT_TARGET_REPLACE.append(module_name)

    network = LoRANetwork(
        pipe.unet,
        rank=args.rank,
        multiplier=1.0,
        alpha=1.0,
        train_method="noxattn",
    ).to(device, dtype=torch_dtype)
    # Sniff estensione: .safetensors usa safetensors.torch.load_file, .pt usa torch.load
    if str(args.slider_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        _ckpt = load_file(str(args.slider_path))
    else:
        _ckpt = torch.load(args.slider_path, map_location=device)
    network.load_state_dict(_ckpt)

    prompt = metadata["prompt"]
    negative_prompt = metadata.get("negative_prompt", "")
    guidance_scale = float(metadata["guidance_scale"])
    steps = int(metadata["steps"])
    height = int(metadata["height"])
    width = int(metadata["width"])
    do_cfg = guidance_scale > 1.0

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
    )
    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = (
        int(pooled_prompt_embeds.shape[-1])
        if getattr(pipe, "text_encoder_2", None) is None
        else pipe.text_encoder_2.config.projection_dim
    )
    add_time_ids = pipe._get_add_time_ids(
        (height, width), (0, 0), (height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    # Recreate initial noise from the same seed used in phase 1.
    # This is more reliable than loading a potentially mismatched cached tensor.
    generator = torch.Generator(device=device_obj).manual_seed(int(metadata["seed"]))
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        width=width,
        dtype=pipe.unet.dtype,
        device=device_obj,
        generator=generator,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

    latent_h = height // pipe.vae_scale_factor
    latent_w = width // pipe.vae_scale_factor
    latent_mask = F.interpolate(mask_img_tensor, size=(latent_h, latent_w), mode="nearest").to(device=device, dtype=latents.dtype)

    with torch.no_grad():
        for t in timesteps:
            eps_base = predict_noise(
                pipe=pipe,
                network=network,
                latents=latents,
                t=t,
                prompt_embeds=prompt_embeds,
                add_text_embeds=add_text_embeds,
                add_time_ids=add_time_ids,
                guidance_scale=guidance_scale,
                lora_scale=0.0,
            )
            if int(t.item()) > args.start_noise:
                eps_lora = eps_base
            else:
                eps_lora = predict_noise(
                    pipe=pipe,
                    network=network,
                    latents=latents,
                    t=t,
                    prompt_embeds=prompt_embeds,
                    add_text_embeds=add_text_embeds,
                    add_time_ids=add_time_ids,
                    guidance_scale=guidance_scale,
                    lora_scale=args.slider_scale,
                )
            eps_blend = latent_mask * eps_lora + (1.0 - latent_mask) * eps_base
            latents = pipe.scheduler.step(eps_blend, t, latents, **extra_step_kwargs, return_dict=False)[0]

        needs_upcasting = pipe.vae.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False)
        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)

        edited_pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
    edited_path = args.run_dir / args.output_name
    edited_pil.save(edited_path)

    metrics_path = args.run_dir / "metrics.json"
    if args.skip_metrics:
        metrics = {
            "skipped": True,
            "reason": "skip_metrics flag enabled (offline cluster safe mode)",
            "slider_scale": args.slider_scale,
            "start_noise": args.start_noise,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        base_pil = Image.open(args.run_dir / "base.png").convert("RGB")
        base_t = torch.from_numpy(np.array(base_pil)).permute(2, 0, 1).float() / 255.0
        edit_t = torch.from_numpy(np.array(edited_pil)).permute(2, 0, 1).float() / 255.0
        mask_full = mask_img_tensor.squeeze(0)

        lpips_model = lpips.LPIPS(net="alex").to(device)
        base_lp = (base_t * 2.0 - 1.0).unsqueeze(0).to(device)
        edit_lp = (edit_t * 2.0 - 1.0).unsqueeze(0).to(device)
        in_mask = mask_full.to(device)
        out_mask = 1.0 - in_mask

        lpips_inside = float(lpips_model(base_lp * in_mask, edit_lp * in_mask).item())
        lpips_outside = float(lpips_model(base_lp * out_mask, edit_lp * out_mask).item())

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        base_np = np.array(base_pil, dtype=np.uint8)
        edit_np = np.array(edited_pil, dtype=np.uint8)
        mask_np = mask_full.squeeze(0).cpu().numpy()[..., None]

        gray_bg = np.full_like(base_np, 127, dtype=np.uint8)
        inside_img = (edit_np * mask_np + gray_bg * (1.0 - mask_np)).astype(np.uint8)
        outside_img = (edit_np * (1.0 - mask_np) + gray_bg * mask_np).astype(np.uint8)
        clip_inside = compute_clip_similarity(clip_model, clip_processor, Image.fromarray(inside_img), prompt, device)
        clip_outside = compute_clip_similarity(clip_model, clip_processor, Image.fromarray(outside_img), prompt, device)

        metrics = {
            "lpips_inside": lpips_inside,
            "lpips_outside": lpips_outside,
            "lpips_localization_ratio": lpips_inside / (lpips_outside + 1e-8),
            "clip_inside": clip_inside,
            "clip_outside": clip_outside,
            "slider_scale": args.slider_scale,
            "start_noise": args.start_noise,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    edit_meta = {
        "phase": "masked_edit",
        "output_image": args.output_name,
        "slider_path": args.slider_path,
        "slider_scale": args.slider_scale,
        "start_noise": args.start_noise,
    }
    (args.run_dir / "edit_meta.json").write_text(json.dumps(edit_meta, indent=2), encoding="utf-8")
    print(f"[OK] Edited image: {edited_path}")
    print(f"[OK] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
