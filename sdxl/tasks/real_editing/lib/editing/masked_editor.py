"""Masked LoRA editor: apply a slider only inside a spatial ROI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from sdxl.tasks.real_editing.lib.models.base import ModelContext, TextCondition
from sdxl.tasks.real_editing.lib.inversion.base import InversionResult
from .blending import feather_mask, load_mask, noise_blend, pixel_composite
from .slider_loader import load_slider

from sdxl.core.lora import LoRANetwork


@dataclass
class EditConfig:
    slider_scale: float = 2.0
    rank: int = 4
    start_noise: int = 700
    guidance_scale: float = 7.5
    steps: int = 50
    seed: int = 1234
    feather_radius: int = 0
    pixel_composite: bool = False
    train_method: str = "noxattn"
    img2img_strength: float = 0.6


@dataclass
class EditResult:
    edited_image: Image.Image
    composite_image: Optional[Image.Image] = None
    config: Optional[EditConfig] = None
    edit_meta: Dict = None


class MaskedLoRAEditor:
    """Apply LoRA/slider editing only inside a masked region.

    The core formula applied at each denoising timestep:
        eps_blend = mask * eps_lora + (1 - mask) * eps_base
    """

    def run(
        self,
        model_ctx: ModelContext,
        inv_result: InversionResult,
        mask_path: str | Path,
        slider_path: str,
        config: EditConfig,
    ) -> EditResult:
        torch.set_grad_enabled(False)
        device = model_ctx.device

        # --- Load and prepare slider ---
        network = load_slider(
            slider_path, model_ctx, rank=config.rank, train_method=config.train_method
        )

        # --- Text conditioning ---
        text_cond = inv_result.text_condition
        prompt = ""  # retrieved from inv_result metadata
        do_cfg = config.guidance_scale > 1.0

        is_tight_run = inv_result.inversion_backend == "tight_inversion"
        has_null_embeds = inv_result.uncond_embeddings is not None
        use_per_step_uncond = has_null_embeds

        # Build conditioning dict for CFG
        if do_cfg:
            cfg_dict = text_cond.cfg_embeds(device)
        else:
            cfg_dict = text_cond.nocfg_embeds(device)

        # --- IP-Adapter for tight inversion runs ---
        # Tight inversion inverts x_t WITH IP-Adapter image conditioning.
        # The editing pass MUST also use IP-Adapter; otherwise the denoising
        # trajectory diverges from the inversion trajectory → black output.
        ipa_image_embeds = None
        if is_tight_run and inv_result.original_image is not None:
            inv_cfg = inv_result.config
            ipa_scale = inv_cfg.ipa_scale if inv_cfg else 0.4
            ipa_weight = (
                inv_cfg.ipa_weight_name
                if inv_cfg
                else "ip-adapter-plus_sdxl_vit-h.safetensors"
            )
            ipa_result = model_ctx.prepare_ip_adapter(
                inv_result.original_image,
                ipa_scale=ipa_scale,
                weight_name=ipa_weight,
            )
            if ipa_result is not None:
                ipa_image_embeds = ipa_result["image_embeds"]
                print(
                    f"[INFO] IP-Adapter re-loaded for editing "
                    f"(scale={ipa_scale}, weight={ipa_weight})"
                )

        # --- Latents ---
        from diffusers import DDIMScheduler
        if not isinstance(model_ctx.scheduler, DDIMScheduler):
            model_ctx.scheduler = DDIMScheduler.from_config(model_ctx.scheduler.config)
            model_ctx.pipe.scheduler = model_ctx.scheduler

        model_ctx.scheduler.set_timesteps(config.steps, device=device)
        timesteps = model_ctx.scheduler.timesteps

        if is_tight_run and inv_result.x_t is None:
            raise ValueError(
                "Tight run requires x_t.pt artifacts. Missing x_t in inversion result."
            )

        if (is_tight_run or use_per_step_uncond) and inv_result.x_t is not None:
            # Null-text mode: start from x_t (the DDIM-inverted latent)
            latents = inv_result.x_t.to(device=device, dtype=model_ctx.dtype)
        else:
            # Img2img mode: re-encode original + add noise at controlled strength.
            # This is necessary because without per-step uncond embeddings,
            # denoising from DDIM x_t with CFG diverges (produces black images).
            source_latents = model_ctx.encode_image(inv_result.original_image)
            generator = torch.Generator(device=device).manual_seed(config.seed)
            noise = torch.randn(source_latents.shape, generator=generator,
                                device=device, dtype=source_latents.dtype)
            init_timestep = min(int(config.steps * config.img2img_strength), config.steps)
            t_start = max(config.steps - init_timestep, 0)
            timesteps = timesteps[t_start:]
            latents = model_ctx.scheduler.add_noise(source_latents, noise, timesteps[0])

        # --- Mask ---
        latent_mask = load_mask(mask_path, latents.shape, device, latents.dtype)
        if config.feather_radius > 0:
            latent_mask = feather_mask(latent_mask, config.feather_radius)

        extra_step_kwargs = model_ctx.pipe.prepare_extra_step_kwargs(
            generator=None, eta=0.0
        )

        # --- Denoising loop ---
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                # Build cond dict for this step (null-text: per-step uncond)
                step_cond = _build_step_cond(
                    text_cond, inv_result, step_idx, do_cfg, device,
                    use_per_step_uncond,
                )

                # Inject IP-Adapter embeddings for tight inversion runs.
                # This keeps the editing trajectory consistent with inversion.
                if ipa_image_embeds is not None:
                    step_cond["image_embeds"] = ipa_image_embeds

                # Base prediction (no LoRA)
                eps_base = _predict_noise(
                    model_ctx, network, latents, t, step_cond,
                    config.guidance_scale, lora_scale=0.0,
                )

                # LoRA prediction (only when t <= start_noise)
                if int(t.item()) > config.start_noise:
                    eps_lora = eps_base
                else:
                    eps_lora = _predict_noise(
                        model_ctx, network, latents, t, step_cond,
                        config.guidance_scale, lora_scale=config.slider_scale,
                    )

                eps_blend = noise_blend(eps_base, eps_lora, latent_mask)
                latents = model_ctx.scheduler.step(
                    eps_blend, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

        # --- Cleanup IP-Adapter after editing ---
        if ipa_image_embeds is not None:
            model_ctx.unload_ip_adapter()

        edited_pil = model_ctx.decode_latents(latents)

        # --- Optional pixel composite ---
        comp_pil = None
        if config.pixel_composite and inv_result.original_image is not None:
            mask_full = load_mask(
                mask_path,
                torch.Size([1, 1, edited_pil.size[1], edited_pil.size[0]]),
                "cpu",
                torch.float32,
            )
            comp_pil = pixel_composite(edited_pil, inv_result.original_image, mask_full)

        return EditResult(
            edited_image=edited_pil,
            composite_image=comp_pil,
            config=config,
            edit_meta={
                "slider_path": slider_path,
                "slider_scale": config.slider_scale,
                "start_noise": config.start_noise,
                "guidance_scale": config.guidance_scale,
                "steps": config.steps,
                "feather_radius": config.feather_radius,
                "pixel_composite": config.pixel_composite,
                "model_family": model_ctx.model_type,
                "ipa_used_in_edit": ipa_image_embeds is not None,
            },
        )


# ---------------------------------------------------------------------------
def _build_step_cond(
    text_cond: TextCondition,
    inv_result: InversionResult,
    step_idx: int,
    do_cfg: bool,
    device: torch.device,
    use_per_step_uncond: bool,
) -> Dict[str, torch.Tensor]:
    """Build the conditioning dict for a single denoising step.

    For null-text inversion results, injects the per-step optimised
    uncond embedding instead of the generic negative prompt embedding.
    """
    if not use_per_step_uncond or not do_cfg:
        if do_cfg:
            return text_cond.cfg_embeds(device)
        return text_cond.nocfg_embeds(device)

    uncond_t = inv_result.uncond_embeddings[step_idx].to(device)
    cond = text_cond.prompt_embeds.to(device)
    uncond_t = uncond_t.expand_as(cond)
    out: Dict[str, torch.Tensor] = {
        "prompt_embeds": torch.cat([uncond_t, cond], dim=0)
    }
    if text_cond.pooled_prompt_embeds is not None:
        out["add_text_embeds"] = torch.cat([
            text_cond.negative_pooled_prompt_embeds.to(device),
            text_cond.pooled_prompt_embeds.to(device),
        ], dim=0)
    if text_cond.add_time_ids is not None:
        out["add_time_ids"] = torch.cat([
            text_cond.add_time_ids.to(device),
            text_cond.add_time_ids.to(device),
        ], dim=0)
    return out


def _predict_noise(
    model_ctx: ModelContext,
    network: LoRANetwork,
    latents: torch.Tensor,
    t: torch.Tensor,
    cond_dict: Dict[str, torch.Tensor],
    guidance_scale: float,
    lora_scale: float,
) -> torch.Tensor:
    """Single noise prediction with optional LoRA and CFG."""
    do_cfg = guidance_scale > 1.0
    network.set_lora_slider(scale=lora_scale)
    lat_in = torch.cat([latents] * 2) if do_cfg else latents
    lat_in = model_ctx.scheduler.scale_model_input(lat_in, t)
    with network:
        noise_pred = model_ctx.unet_forward(lat_in, t, cond_dict)
    if do_cfg:
        nu, nt = noise_pred.chunk(2)
        noise_pred = nu + guidance_scale * (nt - nu)
    return noise_pred
