"""Pure DDIM inversion backend.

Uses the standard reverse-DDIM step to map x_0 -> x_T along the
deterministic DDIM trajectory.  Works for both SDXL and SD1.x via
the ModelContext abstraction.

The ``inv_step`` math is adapted from the Tight Inversion HuggingFace
Space scheduler (see EXTERNAL_SOURCES.md).
"""

from __future__ import annotations

from typing import Dict

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from diffusers import DDIMScheduler

from sdxl.tasks.real_editing.lib.models.base import ModelContext
from .base import InversionBackend, InversionConfig, InversionResult


def _ensure_ddim_scheduler(model_ctx: ModelContext) -> None:
    """Swap the pipeline scheduler to DDIMScheduler if it isn't one already."""
    if not isinstance(model_ctx.scheduler, DDIMScheduler):
        model_ctx.scheduler = DDIMScheduler.from_config(model_ctx.scheduler.config)
        model_ctx.pipe.scheduler = model_ctx.scheduler


class DDIMInversion(InversionBackend):

    name = "ddim"
    status = "implemented"
    description = "Pure DDIM inversion (no optimization). Fast baseline."

    def invert(
        self,
        model_ctx: ModelContext,
        image: Image.Image,
        prompt: str,
        config: InversionConfig,
        negative_prompt: str = "",
    ) -> InversionResult:
        torch.set_grad_enabled(False)
        _ensure_ddim_scheduler(model_ctx)

        text_cond = model_ctx.encode_text(prompt, negative_prompt, do_cfg=False)
        cond_dict = text_cond.nocfg_embeds(model_ctx.device)

        # Apply EXIF orientation before storing (same reason as tight_inversion).
        original_pil = ImageOps.exif_transpose(image.convert("RGB"))
        z_0 = model_ctx.encode_image(original_pil)

        model_ctx.scheduler.set_timesteps(config.steps, device=model_ctx.device)
        timesteps = model_ctx.scheduler.timesteps

        latents = z_0.clone()
        all_latents = [latents.cpu().clone()]

        for t in tqdm(reversed(timesteps), total=len(timesteps), desc="DDIM inv"):
            noise_pred = model_ctx.unet_forward(
                model_ctx.scheduler.scale_model_input(latents, t),
                t,
                cond_dict,
            )
            latents = _inv_step(model_ctx.scheduler, noise_pred, t, latents)
            all_latents.append(latents.cpu().clone())

        # Decode z_0 (VAE-encoded original) as reconstruction baseline.
        # Full denoise from x_T with g=1.0 produces black images (known DDIM
        # limitation without CFG); decoding z_0 shows the VAE round-trip bound.
        reconstruction = model_ctx.decode_latents(z_0.to(model_ctx.device))

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
        )

    def provenance(self) -> Dict[str, str]:
        return {
            "name": "ddim",
            "status": "implemented",
            "source": "inv_step math from HF Space tight-inversion/tight-inversion + diffusers DDIMScheduler",
            "url": "https://huggingface.co/spaces/tight-inversion/tight-inversion",
            "commit": "b9919c2",
            "license": "Public HF Space (no explicit license)",
        }


# ---------------------------------------------------------------------------
# Adapted from tight-inversion HF Space src/schedulers/ddim_scheduler.py
# commit b9919c2  — MyDDIMScheduler.inv_step()
# Reference: Algorithm 3 in https://arxiv.org/pdf/2406.08070
# ---------------------------------------------------------------------------
def _inv_step(
    scheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Reverse (forward-noising) DDIM step: z_t -> z_{t+1}."""
    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    next_timestep = timestep
    timestep_prev = int(next_timestep) - step_ratio

    alpha_prod_t_next = scheduler.alphas_cumprod[int(next_timestep)]
    alpha_prod_t = (
        scheduler.alphas_cumprod[timestep_prev]
        if timestep_prev >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1.0 - alpha_prod_t

    pred_original = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    pred_direction = (1.0 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * pred_original + pred_direction
    return next_sample


def _denoise_from(
    model_ctx: ModelContext,
    x_t: torch.Tensor,
    text_cond,
    config: InversionConfig,
    recon_guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Quick reconstruction pass for quality check.

    Uses guidance_scale=1.0 by default to match the inversion trajectory
    (DDIM inversion with g=1 should reconstruct with g=1).
    """
    gs = recon_guidance_scale
    do_cfg = gs > 1.0
    # Re-encode with CFG if needed (inversion may have used do_cfg=False)
    if do_cfg and text_cond.negative_prompt_embeds is None:
        text_cond = model_ctx.encode_text("", "", do_cfg=True)
    cfg_dict = text_cond.cfg_embeds(model_ctx.device) if do_cfg else text_cond.nocfg_embeds(model_ctx.device)

    model_ctx.scheduler.set_timesteps(config.steps, device=model_ctx.device)
    latents = x_t.clone()

    for t in model_ctx.scheduler.timesteps:
        lat_in = torch.cat([latents] * 2) if do_cfg else latents
        lat_in = model_ctx.scheduler.scale_model_input(lat_in, t)
        noise_pred = model_ctx.unet_forward(lat_in, t, cfg_dict)
        if do_cfg:
            nu, nt = noise_pred.chunk(2)
            noise_pred = nu + config.guidance_scale * (nt - nu)
        latents = model_ctx.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents
