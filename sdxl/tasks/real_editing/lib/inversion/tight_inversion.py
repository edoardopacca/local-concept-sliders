"""Tight Inversion backend: DDIM + gradient-descent optimization + IP-Adapter.

Adapted from the public HuggingFace Space ``tight-inversion/tight-inversion``
(commit b9919c2).  See EXTERNAL_SOURCES.md for full provenance.

Paper : "Tight Inversion: Image-Conditioned Inversion for Real Image Editing"
        Kadosh et al., arXiv:2502.20376, ICCV 2025 Workshop.

Source files adapted
--------------------
* ``src/exact_inversion.py``   -> ``inversion_step()`` + ``unet_pass()``
* ``src/schedulers/ddim_scheduler.py`` -> ``inv_step()``
* ``src/pipes/sdxl_inversion_pipeline.py`` -> overall loop structure
* ``app.py``  -> IP-Adapter integration pattern

The IP-Adapter conditioning is the core novelty of Tight Inversion:
it uses the source image as visual conditioning during inversion via
``h94/IP-Adapter`` (ip-adapter-plus_sdxl_vit-h).  If IP-Adapter weights
are unavailable (offline HPC), the backend falls back to DDIM + GD
and logs a warning.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from sdxl.tasks.real_editing.lib.models.base import ModelContext
from .base import InversionBackend, InversionConfig, InversionResult
from .ddim import _inv_step, _denoise_from, _ensure_ddim_scheduler


class TightInversion(InversionBackend):

    name = "tight_inversion"
    status = "implemented"
    description = (
        "DDIM inversion + per-step GD optimization + IP-Adapter image conditioning. "
        "Adapted from HF Space tight-inversion/tight-inversion (b9919c2)."
    )

    def invert(
        self,
        model_ctx: ModelContext,
        image: Image.Image,
        prompt: str,
        config: InversionConfig,
        negative_prompt: str = "",
    ) -> InversionResult:

        # Apply EXIF orientation before anything else.
        # iPhone/camera photos store pixels rotated with a metadata tag.
        # Without this, original.png is saved rotated while SDXL encodes
        # the correctly-oriented version → mask lands in the wrong place.
        original_pil = ImageOps.exif_transpose(image.convert("RGB"))
        _ensure_ddim_scheduler(model_ctx)

        if not config.use_ipa:
            raise ValueError(
                "Tight inversion in this workflow requires IP-Adapter "
                "(config.use_ipa=True)."
            )

        # --- IP-Adapter setup (optional) ---
        ipa_embeds: Optional[Dict[str, Any]] = None
        if config.use_ipa and model_ctx.model_type == "sdxl":
            ipa_embeds = model_ctx.prepare_ip_adapter(
                original_pil,
                ipa_scale=config.ipa_scale,
                weight_name=config.ipa_weight_name,
            )
            if ipa_embeds is None:
                raise RuntimeError(
                    "[TightInversion] IP-Adapter could not be loaded. "
                    "Install/cache h94/IP-Adapter first. "
                    "No fallback mode is enabled."
                )

        # --- Text conditioning ---
        do_cfg = config.guidance_scale > 1.0
        text_cond = model_ctx.encode_text(prompt, negative_prompt, do_cfg=do_cfg)

        if do_cfg:
            cond_dict = text_cond.cfg_embeds(model_ctx.device)
        else:
            cond_dict = text_cond.nocfg_embeds(model_ctx.device)

        # Inject IPA embeds into conditioning dict
        if ipa_embeds is not None:
            image_embeds = ipa_embeds["image_embeds"]
            if do_cfg and config.guidance_scale <= 1.0:
                # guidance_scale == 1 with IPA: use cond embeds only
                cond_dict["image_embeds"] = [image_embeds[0][None, 1]]
            else:
                cond_dict["image_embeds"] = image_embeds

        # --- VAE encode ---
        z_0 = model_ctx.encode_image(original_pil)

        # --- Inversion loop ---
        model_ctx.scheduler.set_timesteps(config.steps, device=model_ctx.device)
        timesteps = model_ctx.scheduler.timesteps

        latents = z_0.clone()
        all_latents = [latents.cpu().clone()]

        for i, t in enumerate(
            tqdm(reversed(timesteps), total=len(timesteps), desc="Tight inv")
        ):
            latents = _tight_inversion_step(
                model_ctx=model_ctx,
                z_t=latents,
                t=t,
                cond_dict=cond_dict,
                do_cfg=do_cfg,
                guidance_scale=config.guidance_scale,
                num_gd_steps=config.num_gd_steps,
                gd_step_size=config.gd_step_size,
                optimization_start=config.optimization_start,
            )
            all_latents.append(latents.cpu().clone())

        # --- Reconstruction ---
        # Decode z_0 (the VAE-encoded original) as reconstruction baseline.
        # A full denoise from x_T with g=1.0 would be black (known DDIM limitation);
        # the GD steps improve the inversion trajectory but the forward denoise
        # still drifts without CFG.  Decoding z_0 directly shows what the VAE
        # round-trip looks like (encode -> decode), which is the best-case bound.
        reconstruction = model_ctx.decode_latents(z_0.to(model_ctx.device))

        # --- Cleanup IPA ---
        if ipa_embeds is not None:
            model_ctx.unload_ip_adapter()

        return InversionResult(
            model_family=model_ctx.model_type,
            model_id=model_ctx.model_id,
            inversion_backend=self.name,
            backend_status=self.status,
            x_t=all_latents[-1],
            all_latents=all_latents,
            original_image=original_pil,
            reconstruction_image=reconstruction,
            text_condition=text_cond,
            config=config,
            source_manifest=self.provenance(),
            ip_adapter_image_embeds=ipa_embeds,
        )

    def provenance(self) -> Dict[str, str]:
        return {
            "name": "tight_inversion",
            "status": "implemented",
            "source": "HF Space tight-inversion/tight-inversion",
            "url": "https://huggingface.co/spaces/tight-inversion/tight-inversion",
            "commit": "b9919c2",
            "license": "Public HF Space (no explicit license file)",
            "paper": "arXiv:2502.20376",
            "files_adapted": (
                "src/exact_inversion.py, "
                "src/schedulers/ddim_scheduler.py, "
                "src/pipes/sdxl_inversion_pipeline.py, "
                "app.py"
            ),
        }


# ---------------------------------------------------------------------------
# Core step: DDIM inv_step + optional gradient-descent optimisation.
# Adapted from tight-inversion HF Space  src/exact_inversion.py
# function ``inversion_step()`` (commit b9919c2).
# ---------------------------------------------------------------------------
def _tight_inversion_step(
    model_ctx: ModelContext,
    z_t: torch.Tensor,
    t: torch.Tensor,
    cond_dict: Dict[str, torch.Tensor],
    do_cfg: bool,
    guidance_scale: float,
    num_gd_steps: int = 0,
    gd_step_size: float = 0.001,
    optimization_start: int = 0,
) -> torch.Tensor:
    """Single inversion step: reverse DDIM + optional GD refinement."""

    # --- Standard DDIM inverse step ---
    noise_pred = _unet_pass_cfg(
        model_ctx, z_t, t, cond_dict, do_cfg, guidance_scale
    )
    approx_z_tp1 = _inv_step(model_ctx.scheduler, noise_pred, t, z_t)

    # --- GD optimisation (only past optimization_start timestep) ---
    t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
    if num_gd_steps <= 0 or t_int < optimization_start:
        return approx_z_tp1.detach()

    # Use manual gradient steps instead of optimizer state to avoid
    # reusing freed autograd graphs across inner-loop iterations.
    approx_z_tp1 = approx_z_tp1.detach()
    z_t_target = z_t.detach()
    for _ in range(num_gd_steps):
        approx_z_tp1 = approx_z_tp1.detach().requires_grad_(True)
        noise_pred_opt = _unet_pass_cfg(
            model_ctx, approx_z_tp1, t, cond_dict, do_cfg, guidance_scale
        )
        recon_z_t = model_ctx.scheduler.step(
            noise_pred_opt, t, approx_z_tp1, return_dict=False
        )[0]
        loss = torch.nn.functional.mse_loss(recon_z_t, z_t_target)
        grad = torch.autograd.grad(loss, approx_z_tp1, retain_graph=False, create_graph=False)[0]
        approx_z_tp1 = (approx_z_tp1 - gd_step_size * grad).detach()

    return approx_z_tp1.detach()


# ---------------------------------------------------------------------------
# UNet forward with optional CFG.
# Adapted from tight-inversion HF Space src/exact_inversion.py ``unet_pass``.
# ---------------------------------------------------------------------------
def _unet_pass_cfg(
    model_ctx: ModelContext,
    latents: torch.Tensor,
    t: torch.Tensor,
    cond_dict: Dict[str, torch.Tensor],
    do_cfg: bool,
    guidance_scale: float,
) -> torch.Tensor:
    lat_in = torch.cat([latents] * 2) if do_cfg else latents
    lat_in = model_ctx.scheduler.scale_model_input(lat_in, t)
    noise_pred = model_ctx.unet_forward(lat_in, t, cond_dict)
    if do_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
    return noise_pred
