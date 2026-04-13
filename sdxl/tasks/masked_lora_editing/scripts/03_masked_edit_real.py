#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# __file__ = .../sdxl/tasks/masked_lora_editing/scripts/03_masked_edit_real.py → parents[4] = repo root
REPO_ROOT = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO_ROOT)

from sdxl.core.lora import (  # noqa: E402
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
    LoRANetwork,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase 3 (real editing): masked LoRA edit from inversion artifacts")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--slider_path", type=str, required=True)
    parser.add_argument("--slider_scale", type=float, default=2.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--start_noise", type=int, default=500)
    parser.add_argument("--mask_name", type=str, default="mask_target.png")
    parser.add_argument("--output_name", type=str, default="edited_target_only.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float32")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    return parser


def dtype_from_str(dtype_name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[dtype_name]


def _noise_pred(unet, network, latent_model_input, t, text_embeddings, lora_scale: float) -> torch.Tensor:
    network.set_lora_slider(scale=lora_scale)
    with network:
        pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    return pred


def main() -> None:
    args = build_parser().parse_args()
    torch.set_grad_enabled(False)

    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    x_t = torch.load(args.run_dir / "x_t.pt", map_location="cpu")
    uncond_embeddings = torch.load(args.run_dir / "uncond_embeddings.pt", map_location="cpu")
    mask_img = Image.open(args.run_dir / args.mask_name).convert("L")
    mask_np = (np.array(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    mask_full = torch.from_numpy(mask_np)[None, None, ...]

    device = args.device
    weight_dtype = dtype_from_str(args.dtype)

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", local_files_only=args.local_files_only)
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", local_files_only=args.local_files_only)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", local_files_only=args.local_files_only)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", local_files_only=args.local_files_only)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    for module_name in UNET_TARGET_REPLACE_MODULE_CONV:
        if module_name not in DEFAULT_TARGET_REPLACE:
            DEFAULT_TARGET_REPLACE.append(module_name)

    network = LoRANetwork(
        unet,
        rank=args.rank,
        multiplier=1.0,
        alpha=1.0,
        train_method="noxattn",
    ).to(device, dtype=weight_dtype)
    # Sniff estensione: .safetensors usa safetensors.torch.load_file, .pt usa torch.load
    if str(args.slider_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        _ckpt = load_file(str(args.slider_path))
    else:
        _ckpt = torch.load(args.slider_path, map_location=device)
    network.load_state_dict(_ckpt)

    text_input = tokenizer(
        metadata["prompt"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings_cond = text_encoder(text_input.input_ids.to(device))[0]

    noise_scheduler.set_timesteps(int(metadata["steps"]))
    latents = x_t.to(device=device, dtype=weight_dtype) * noise_scheduler.init_noise_sigma
    latent_mask = F.interpolate(mask_full, size=latents.shape[-2:], mode="nearest").to(device=device, dtype=latents.dtype)

    with torch.no_grad():
        for step_idx, t in enumerate(noise_scheduler.timesteps):
            uncond_step = uncond_embeddings[step_idx : step_idx + 1].to(device=device, dtype=weight_dtype)
            text_embeddings = torch.cat([uncond_step.expand(*text_embeddings_cond.shape), text_embeddings_cond], dim=0)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            pred_base = _noise_pred(unet, network, latent_model_input, t, text_embeddings, lora_scale=0.0)
            pred_base_uncond, pred_base_text = pred_base.chunk(2)
            eps_base = pred_base_uncond + args.guidance_scale * (pred_base_text - pred_base_uncond)

            if int(t.item()) > args.start_noise:
                eps_lora = eps_base
            else:
                pred_lora = _noise_pred(unet, network, latent_model_input, t, text_embeddings, lora_scale=args.slider_scale)
                pred_lora_uncond, pred_lora_text = pred_lora.chunk(2)
                eps_lora = pred_lora_uncond + args.guidance_scale * (pred_lora_text - pred_lora_uncond)

            eps_blend = latent_mask * eps_lora + (1.0 - latent_mask) * eps_base
            latents = noise_scheduler.step(eps_blend, t, latents).prev_sample

        decoded = vae.decode((1 / 0.18215) * latents).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded_np = decoded.detach().cpu().permute(0, 2, 3, 1).to(torch.float32).numpy()
        out = (decoded_np[0] * 255).round().astype(np.uint8)
        Image.fromarray(out).save(args.run_dir / args.output_name)

    metrics = {
        "skipped": bool(args.skip_metrics),
        "reason": "real-edit pipeline currently exports image only; metrics optional extension",
        "slider_scale": args.slider_scale,
        "start_noise": args.start_noise,
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.run_dir / "edit_meta.json").write_text(
        json.dumps(
            {
                "phase": "masked_real_edit",
                "run_dir": str(args.run_dir),
                "mask_name": args.mask_name,
                "output_name": args.output_name,
                "slider_path": args.slider_path,
                "slider_scale": args.slider_scale,
                "start_noise": args.start_noise,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Edited image: {args.run_dir / args.output_name}")
    print(f"[OK] Metadata: {args.run_dir / 'edit_meta.json'}")


if __name__ == "__main__":
    main()
