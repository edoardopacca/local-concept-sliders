#!/usr/bin/env python3
"""
SDXL phase 3 of the external mask-guided pipeline: multi-mask + multi-LoRA
masked edit.

Two CLI modes:

(1) Legacy single-mask single-slider:
      --slider_path X.pt --mask_name M.png --slider_scale S

(2) Multi-mask + multi-LoRA per mask:
      --slider_paths X1.pt X2.pt X3.pt
      --mask_names M1.png M2.png
      --slider_to_mask 0 0 1
      --slider_scales 1.0 2.0 0.5

In mode (2): N disjoint masks, M sliders per mask (summed additively via
N nested LoRANetwork instances). For each denoising step:
  1 base forward      (every LoRA off)
  N styled forwards   (one per mask, with only its sliders active)
  blend: out = (1 - sum_i mask_i) * base + sum_i (mask_i * styled_i)

In mode (1): one base + one styled + blend (operator of eq. (2) in the paper).
"""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(
        "Phase 3 - Masked LoRA edit (legacy single + multi-mask multi-LoRA)"
    )
    parser.add_argument("--run_dir", type=Path, required=True)

    # Legacy single-mask single-slider
    parser.add_argument("--slider_path", type=str, default=None,
                        help="Legacy: a single slider. Mutually exclusive with --slider_paths.")
    parser.add_argument("--slider_scale", type=float, default=2.0,
                        help="Legacy: scale for the single slider.")
    parser.add_argument("--mask_name", type=str, default=None,
                        help="Legacy: a single mask (default mask.png). "
                             "Mutually exclusive with --mask_names.")

    # Multi-mask multi-LoRA
    parser.add_argument("--slider_paths", type=str, nargs="+", default=None,
                        help="Multi-mode: list of sliders (.pt or .safetensors).")
    parser.add_argument("--mask_names", type=str, nargs="+", default=None,
                        help="Multi-mode: list of N masks (binary PNGs in the run dir).")
    parser.add_argument("--slider_to_mask", type=int, nargs="+", default=None,
                        help="Multi-mode: slider_to_mask[i]=j means slider i is applied "
                             "inside mask j. Length must match --slider_paths. Several "
                             "sliders on the same mask compose additively.")
    parser.add_argument("--slider_scales", type=float, nargs="+", default=None,
                        help="Multi-mode: one scale per slider (1:1 with --slider_paths).")

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--start_noise", type=int, default=700,
                        help="Timestep threshold: for t > start_noise the styled forward "
                             "is skipped (LoRA inactive, output = base). Default 700.")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output_name", type=str, default="edited.png")
    parser.add_argument("--skip_metrics", action="store_true")
    return parser


def _normalize_args(args) -> Tuple[List[str], List[str], List[int], List[float], bool]:
    """Normalise legacy and multi CLI into the same multi-mode form.

    Returns:
      slider_paths   : List[str]
      mask_names     : List[str]
      slider_to_mask : List[int]   (mapping slider_idx -> mask_idx)
      slider_scales  : List[float]
      is_multi       : bool        (True if running in multi-mode)
    """
    has_slider_paths = args.slider_paths is not None
    has_slider_path = args.slider_path is not None
    has_mask_names = args.mask_names is not None
    has_mask_name = args.mask_name is not None
    has_slider_to_mask = args.slider_to_mask is not None
    has_slider_scales = args.slider_scales is not None

    is_multi = has_slider_paths or has_mask_names or has_slider_to_mask or has_slider_scales

    if is_multi:
        if has_slider_path or has_mask_name:
            raise ValueError(
                "Mixing legacy (--slider_path/--mask_name) and multi "
                "(--slider_paths/--mask_names/--slider_to_mask/--slider_scales) is "
                "ambiguous. Use ONE form or the other."
            )
        if not has_slider_paths:
            raise ValueError("Multi-mode requires --slider_paths.")
        if not has_mask_names:
            raise ValueError("Multi-mode requires --mask_names.")
        if not has_slider_to_mask:
            raise ValueError("Multi-mode requires --slider_to_mask (slider->mask mapping).")
        if not has_slider_scales:
            raise ValueError("Multi-mode requires --slider_scales (one value per slider).")
        n_sliders = len(args.slider_paths)
        n_masks = len(args.mask_names)
        if len(args.slider_to_mask) != n_sliders:
            raise ValueError(
                f"--slider_to_mask has {len(args.slider_to_mask)} values, expected "
                f"{n_sliders} (one per --slider_paths)."
            )
        if len(args.slider_scales) != n_sliders:
            raise ValueError(
                f"--slider_scales has {len(args.slider_scales)} values, expected "
                f"{n_sliders} (one per --slider_paths)."
            )
        for s_idx, m_idx in enumerate(args.slider_to_mask):
            if not (0 <= m_idx < n_masks):
                raise ValueError(
                    f"--slider_to_mask[{s_idx}]={m_idx} is out of range "
                    f"[0, {n_masks}); there are {n_masks} masks."
                )
        masks_used = set(args.slider_to_mask)
        for m in range(n_masks):
            if m not in masks_used:
                raise ValueError(
                    f"Mask {m} ('{args.mask_names[m]}') has no slider mapped to it. "
                    f"Add at least one slider that maps to this mask."
                )
        return (list(args.slider_paths), list(args.mask_names),
                list(args.slider_to_mask), list(args.slider_scales), True)

    # ---- Legacy mode ----
    if not has_slider_path:
        raise ValueError(
            "Need --slider_path (legacy) or --slider_paths (multi)."
        )
    mask_name = args.mask_name if has_mask_name else "mask.png"
    return ([args.slider_path], [mask_name], [0], [args.slider_scale], False)


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


def load_run_inputs(run_dir: Path, mask_names: List[str]) -> Tuple[Dict, torch.Tensor, List[torch.Tensor]]:
    """Load metadata.json, init_latents (if present) and a LIST of masks.

    Returns ``(metadata, init_latents_or_None, [mask_tensor_1, ...])`` where
    each mask_tensor has shape ``(1, 1, H, W)`` with values in ``[0, 1]``.
    """
    metadata_path = run_dir / "metadata.json"
    init_latents_path = run_dir / "init_latents.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # init_latents may be absent — when missing it is rebuilt from the seed.
    init_latents = None
    if init_latents_path.exists():
        init_latents = torch.load(init_latents_path, map_location="cpu")

    masks = []
    for mname in mask_names:
        mp = run_dir / mname
        if not mp.exists():
            raise FileNotFoundError(f"Missing mask: {mp}")
        mask_img = Image.open(mp).convert("L")
        mask_np = (np.array(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
        masks.append(torch.from_numpy(mask_np)[None, None, ...])
    return metadata, init_latents, masks


def _unet_forward(
    pipe: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    add_text_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Single UNet forward with CFG. The LoRA state (multiplier on every
    active network) is implicit at runtime — the caller is responsible
    for setting the multipliers beforehand (e.g. via ExitStack on a
    subset of LoRANetwork)."""
    do_cfg = guidance_scale > 1.0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
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


def predict_noise_base(
    pipe: StableDiffusionXLPipeline,
    all_networks: List[LoRANetwork],
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    add_text_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Forward con TUTTI i LoRA OFF (multiplier=0)."""
    for net in all_networks:
        for lora in net.unet_loras:
            lora.multiplier = 0.0
    return _unet_forward(pipe, latents, t, prompt_embeds,
                         add_text_embeds, add_time_ids, guidance_scale)


def predict_noise_styled(
    pipe: StableDiffusionXLPipeline,
    networks_to_activate: List[LoRANetwork],
    scales_to_activate: List[float],
    all_networks: List[LoRANetwork],
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    add_text_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Forward with ONLY ``networks_to_activate`` active, each at its
    own scale. Every other network in ``all_networks`` is disabled
    (multiplier=0).

    Underneath, the N nested LoRANetwork instances wrap the UNet forward
    additively: ``out = base(x) + sum_i s_i * LoRA_i(x)``.
    """
    # Disable every network first.
    for net in all_networks:
        for lora in net.unet_loras:
            lora.multiplier = 0.0
    # ExitStack: each activated network's __enter__ sets
    # lora.multiplier = 1.0 * net.lora_scale on its unet_loras, and
    # __exit__ resets them to 0.
    with ExitStack() as stack:
        for net, scale in zip(networks_to_activate, scales_to_activate):
            net.set_lora_slider(scale=float(scale))
            stack.enter_context(net)
        return _unet_forward(pipe, latents, t, prompt_embeds,
                             add_text_embeds, add_time_ids, guidance_scale)


def compute_clip_similarity(clip_model, clip_processor, image: Image.Image, text: str, device: str) -> float:
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
        image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_emb * text_emb).sum(dim=-1).item())


def main() -> None:
    args = build_parser().parse_args()

    (slider_paths, mask_names, slider_to_mask,
     slider_scales, is_multi) = _normalize_args(args)
    num_sliders = len(slider_paths)
    num_masks = len(mask_names)

    print(f"[phase3] run_dir: {args.run_dir}")
    print(f"[phase3] mode: {'multi' if is_multi else 'legacy'}  "
          f"(num_masks={num_masks}, num_sliders={num_sliders})")

    metadata, _init_latents_cpu, mask_img_tensors = load_run_inputs(args.run_dir, mask_names)
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

    # ---- Load N distinct LoRANetwork instances ----
    # Each LoRANetwork calls apply_to() on its unet_loras (see
    # sdxl/core/lora.py), hooking into the UNet forward. The N instances
    # nest in cascade: out_module(x) = base(x) + sum_i (mult_i * LoRA_i(x)),
    # which is the additive aggregation used in the paper. The
    # multipliers are set per-network via __enter__ (= 1.0 * lora_scale)
    # and zeroed on inactive networks so that the right subset is active
    # in each masked region.
    print(f"[phase3] loading {num_sliders} slider(s) as separate LoRANetwork instances")
    all_networks: List[LoRANetwork] = []
    for i, sp in enumerate(slider_paths):
        net = LoRANetwork(
            pipe.unet,
            rank=args.rank,
            multiplier=1.0,
            alpha=1.0,
            train_method="noxattn",
        ).to(device, dtype=torch_dtype)
        if str(sp).endswith(".safetensors"):
            from safetensors.torch import load_file
            _ckpt = load_file(str(sp))
        else:
            _ckpt = torch.load(sp, map_location=device)
        net.load_state_dict(_ckpt)
        all_networks.append(net)
        m_idx = slider_to_mask[i]
        print(f"  [slider {i}] -> mask[{m_idx}] '{mask_names[m_idx]}'  "
              f"scale={slider_scales[i]}  path={sp}")
    # Disable every multiplier up front; they are activated selectively
    # inside the per-mask loop.
    for net in all_networks:
        for lora in net.unet_loras:
            lora.multiplier = 0.0

    # Map mask_idx -> list of (network_idx, scale) for that mask.
    mask_to_active: List[List[Tuple[int, float]]] = [[] for _ in range(num_masks)]
    for s_idx, m_idx in enumerate(slider_to_mask):
        mask_to_active[m_idx].append((s_idx, slider_scales[s_idx]))

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
    # Pre-compute the masks at latent resolution.
    latent_masks: List[torch.Tensor] = []
    for m in mask_img_tensors:
        lm = F.interpolate(m, size=(latent_h, latent_w), mode="nearest").to(
            device=device, dtype=latents.dtype
        )
        latent_masks.append(lm)
    # Overlap warning (sum > 1 on a pixel = masks overlap).
    if num_masks > 1:
        sum_lm = torch.zeros_like(latent_masks[0])
        for lm in latent_masks:
            sum_lm = sum_lm + lm
        n_overlap = int((sum_lm > 1).sum().item())
        if n_overlap > 0:
            print(f"[phase3] WARNING: {n_overlap} latent pixel(s) have "
                  f"overlapping masks (sum>1). The formula assumes "
                  f"disjoint masks; on overlapping pixels the effects "
                  f"add up — check the SAM masks.")

    print(f"[phase3] denoising: steps={steps}  num_masks={num_masks}  "
          f"forward_per_step={1 + num_masks} (1 base + {num_masks} per-mask)")

    with torch.no_grad():
        for t in timesteps:
            # ---- 1 base forward (every LoRA off) ----
            eps_base = predict_noise_base(
                pipe=pipe, all_networks=all_networks,
                latents=latents, t=t,
                prompt_embeds=prompt_embeds,
                add_text_embeds=add_text_embeds,
                add_time_ids=add_time_ids,
                guidance_scale=guidance_scale,
            )

            # ---- start_noise cutoff: per i timestep iniziali (rumore alto)
            # saltiamo i forward styled, usiamo solo base ovunque.
            # Equivalente al comportamento legacy "if t > start_noise: eps_lora = eps_base".
            if int(t.item()) > args.start_noise:
                eps_pred = eps_base
            else:
                # ---- N forward styled (1 per mask, con sub-set di LoRA attivi) ----
                eps_styled_list: List[torch.Tensor] = []
                for mask_idx in range(num_masks):
                    active = mask_to_active[mask_idx]
                    nets = [all_networks[s_idx] for (s_idx, _s) in active]
                    scales = [s for (_s_idx, s) in active]
                    eps_styled_list.append(
                        predict_noise_styled(
                            pipe=pipe,
                            networks_to_activate=nets,
                            scales_to_activate=scales,
                            all_networks=all_networks,
                            latents=latents, t=t,
                            prompt_embeds=prompt_embeds,
                            add_text_embeds=add_text_embeds,
                            add_time_ids=add_time_ids,
                            guidance_scale=guidance_scale,
                        )
                    )
                # ---- Blend iterativo (assume mask disgiunte) ----
                # eps_pred = (1 - sum mask_i) * eps_base + sum (mask_i * eps_styled_i)
                eps_pred = eps_base.clone()
                for mask_idx in range(num_masks):
                    m = latent_masks[mask_idx]
                    eps_pred = eps_pred - m * eps_base + m * eps_styled_list[mask_idx]

            latents = pipe.scheduler.step(eps_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

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
    if args.skip_metrics or is_multi:
        metrics = {
            "skipped": True,
            "reason": ("multi-mode (metrics non implementate per multi-mask)"
                       if is_multi else
                       "skip_metrics flag enabled (offline cluster safe mode)"),
            "mode": "multi" if is_multi else "legacy",
            "slider_paths": slider_paths,
            "slider_scales": slider_scales,
            "mask_names": mask_names,
            "slider_to_mask": slider_to_mask,
            "start_noise": args.start_noise,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        # Legacy mode: 1 mask, 1 slider — same metrics as before.
        mask_img_tensor = mask_img_tensors[0]
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
            "slider_scale": slider_scales[0],
            "start_noise": args.start_noise,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    edit_meta = {
        "phase": "masked_edit",
        "mode": "multi" if is_multi else "legacy",
        "output_image": args.output_name,
        "slider_paths": slider_paths,
        "slider_scales": slider_scales,
        "mask_names": mask_names,
        "slider_to_mask": slider_to_mask,
        "start_noise": args.start_noise,
    }
    (args.run_dir / "edit_meta.json").write_text(json.dumps(edit_meta, indent=2), encoding="utf-8")
    print(f"[OK] Edited image: {edited_path}")
    print(f"[OK] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
