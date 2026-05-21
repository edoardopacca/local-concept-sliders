"""Null-Text Inversion backend.

Adapted from the existing ``exp_editing/edit_with_sliders.py``
NullInversion class in this repository.

For SD1.4 this is a well-tested, proven implementation.
For SDXL the same optimisation principle is applied to the larger
uncond embedding tensor; this path is marked **experimental**.

See the "Origin" section of the task README.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn.functional as nnf
from PIL import Image, ImageOps
from torch.optim.adam import Adam
from tqdm import tqdm

from sdxl.tasks.real_editing.lib.models.base import ModelContext, TextCondition
from .base import InversionBackend, InversionConfig, InversionResult
from .ddim import _inv_step, _ensure_ddim_scheduler


class NullTextInversion(InversionBackend):

    name = "null_text"
    status = "implemented"
    description = (
        "Null-Text Inversion from Mokady et al. (arXiv:2211.09794). "
        "SD1.4: implemented. SDXL: experimental."
    )

    def invert(
        self,
        model_ctx: ModelContext,
        image: Image.Image,
        prompt: str,
        config: InversionConfig,
        negative_prompt: str = "",
    ) -> InversionResult:
        original_pil = ImageOps.exif_transpose(image.convert("RGB"))
        _ensure_ddim_scheduler(model_ctx)

        text_cond = model_ctx.encode_text(prompt, negative_prompt, do_cfg=True)
        cond_embeds = text_cond.prompt_embeds          # [1, seq, dim]
        uncond_embeds = text_cond.negative_prompt_embeds  # [1, seq, dim]

        z_0 = model_ctx.encode_image(original_pil)

        model_ctx.scheduler.set_timesteps(config.steps, device=model_ctx.device)
        timesteps = model_ctx.scheduler.timesteps

        # ----- Phase 1: DDIM forward loop (cond-only) -----
        latent = z_0.clone().to(model_ctx.unet.dtype)
        ddim_latents = [latent.cpu().clone()]
        for i in range(config.steps):
            t = timesteps[len(timesteps) - i - 1]
            cond_dict = {"prompt_embeds": cond_embeds.to(model_ctx.device)}
            if text_cond.pooled_prompt_embeds is not None:
                cond_dict["add_text_embeds"] = text_cond.pooled_prompt_embeds.to(model_ctx.device)
            if text_cond.add_time_ids is not None:
                cond_dict["add_time_ids"] = text_cond.add_time_ids.to(model_ctx.device)
            scaled = model_ctx.scheduler.scale_model_input(latent, t)
            noise_pred = model_ctx.unet_forward(scaled, t, cond_dict)
            latent = _inv_step(model_ctx.scheduler, noise_pred, t, latent)
            ddim_latents.append(latent.cpu().clone())

        # ----- Phase 2: Null-text optimisation -----
        uncond_list = _null_optimization(
            model_ctx=model_ctx,
            ddim_latents=ddim_latents,
            cond_embeds=cond_embeds,
            uncond_embeds_init=uncond_embeds,
            text_cond=text_cond,
            guidance_scale=config.guidance_scale if config.guidance_scale > 1.0 else 7.5,
            num_inner_steps=config.num_inner_steps,
            epsilon=config.early_stop_epsilon,
        )

        # ----- Reconstruction -----
        recon = _reconstruct_with_null_embeds(
            model_ctx=model_ctx,
            x_t=ddim_latents[-1].to(model_ctx.device),
            cond_embeds=cond_embeds,
            uncond_list=uncond_list,
            text_cond=text_cond,
            guidance_scale=config.guidance_scale if config.guidance_scale > 1.0 else 7.5,
            steps=config.steps,
        )
        reconstruction_pil = model_ctx.decode_latents(recon)

        stacked_uncond = torch.stack([u.squeeze(0).cpu() for u in uncond_list], dim=0)

        backend_status = "implemented" if model_ctx.model_type == "sd1x" else "experimental"

        return InversionResult(
            model_family=model_ctx.model_type,
            model_id=model_ctx.model_id,
            inversion_backend=self.name,
            backend_status=backend_status,
            x_t=ddim_latents[-1],
            all_latents=ddim_latents,
            original_image=original_pil,
            reconstruction_image=reconstruction_pil,
            uncond_embeddings=stacked_uncond,
            text_condition=text_cond,
            config=config,
            source_manifest=self.provenance(),
        )

    def provenance(self) -> Dict[str, str]:
        return {
            "name": "null_text",
            "status": "implemented (SD1.4), experimental (SDXL)",
            "source": "This repo: exp_editing/edit_with_sliders.py NullInversion class",
            "url": "local",
            "commit": "--",
            "license": "--",
            "paper": "arXiv:2211.09794",
        }


# ---------------------------------------------------------------------------
# Adapted from exp_editing/edit_with_sliders.py  NullInversion.null_optimization
# ---------------------------------------------------------------------------
def _null_optimization(
    model_ctx: ModelContext,
    ddim_latents: list,
    cond_embeds: torch.Tensor,
    uncond_embeds_init: torch.Tensor,
    text_cond: TextCondition,
    guidance_scale: float,
    num_inner_steps: int,
    epsilon: float,
) -> List[torch.Tensor]:

    uncond_embeddings_list: List[torch.Tensor] = []
    latent_cur = ddim_latents[-1].to(model_ctx.device)

    model_ctx.scheduler.set_timesteps(
        len(ddim_latents) - 1, device=model_ctx.device
    )
    timesteps = model_ctx.scheduler.timesteps

    bar = tqdm(
        total=num_inner_steps * len(timesteps), desc="Null-text opt"
    )

    for i in range(len(timesteps)):
        uncond = uncond_embeds_init.clone().detach().to(model_ctx.device)
        uncond.requires_grad = True
        optimizer = Adam([uncond], lr=1e-2 * (1.0 - i / 100.0))
        latent_prev = ddim_latents[len(ddim_latents) - i - 2].to(model_ctx.device)
        t = timesteps[i]

        # Cond noise (frozen)
        cond_dict = {"prompt_embeds": cond_embeds.to(model_ctx.device)}
        if text_cond.pooled_prompt_embeds is not None:
            cond_dict["add_text_embeds"] = text_cond.pooled_prompt_embeds.to(model_ctx.device)
        if text_cond.add_time_ids is not None:
            cond_dict["add_time_ids"] = text_cond.add_time_ids.to(model_ctx.device)
        with torch.no_grad():
            scaled = model_ctx.scheduler.scale_model_input(latent_cur, t)
            noise_pred_cond = model_ctx.unet_forward(scaled, t, cond_dict)

        for j in range(num_inner_steps):
            uncond_dict = {"prompt_embeds": uncond}
            if text_cond.pooled_prompt_embeds is not None:
                uncond_dict["add_text_embeds"] = text_cond.negative_pooled_prompt_embeds.to(model_ctx.device)
            if text_cond.add_time_ids is not None:
                uncond_dict["add_time_ids"] = text_cond.add_time_ids.to(model_ctx.device)
            scaled = model_ctx.scheduler.scale_model_input(latent_cur, t)
            noise_pred_uncond = model_ctx.unet_forward(scaled, t, uncond_dict)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            latent_prev_rec = model_ctx.scheduler.step(
                noise_pred, t, latent_cur, return_dict=False
            )[0]
            loss = nnf.mse_loss(latent_prev_rec, latent_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.update()
            if loss.item() < epsilon + i * 2e-5:
                break

        for j2 in range(j + 1, num_inner_steps):
            bar.update()

        uncond_embeddings_list.append(uncond[:1].detach())

        # Update latent_cur with optimised uncond
        with torch.no_grad():
            cfg_embeds = torch.cat([uncond.detach(), cond_embeds.to(model_ctx.device)])
            full_dict = {"prompt_embeds": cfg_embeds}
            if text_cond.pooled_prompt_embeds is not None:
                full_dict["add_text_embeds"] = torch.cat([
                    text_cond.negative_pooled_prompt_embeds.to(model_ctx.device),
                    text_cond.pooled_prompt_embeds.to(model_ctx.device),
                ])
            if text_cond.add_time_ids is not None:
                full_dict["add_time_ids"] = torch.cat([
                    text_cond.add_time_ids.to(model_ctx.device),
                    text_cond.add_time_ids.to(model_ctx.device),
                ])
            lat_in = torch.cat([latent_cur] * 2)
            lat_in = model_ctx.scheduler.scale_model_input(lat_in, t)
            noise_pred = model_ctx.unet_forward(lat_in, t, full_dict)
            nu, nt = noise_pred.chunk(2)
            noise_pred = nu + guidance_scale * (nt - nu)
            latent_cur = model_ctx.scheduler.step(
                noise_pred, t, latent_cur, return_dict=False
            )[0]

    bar.close()
    return uncond_embeddings_list


def _reconstruct_with_null_embeds(
    model_ctx: ModelContext,
    x_t: torch.Tensor,
    cond_embeds: torch.Tensor,
    uncond_list: List[torch.Tensor],
    text_cond: TextCondition,
    guidance_scale: float,
    steps: int,
) -> torch.Tensor:
    model_ctx.scheduler.set_timesteps(steps, device=model_ctx.device)
    latents = x_t.clone()
    for cnt, t in enumerate(model_ctx.scheduler.timesteps):
        uncond_t = uncond_list[cnt].to(model_ctx.device)
        uncond_t = uncond_t.expand_as(cond_embeds.to(model_ctx.device))
        cfg_embeds = torch.cat([uncond_t, cond_embeds.to(model_ctx.device)])
        full_dict: Dict = {"prompt_embeds": cfg_embeds}
        if text_cond.pooled_prompt_embeds is not None:
            full_dict["add_text_embeds"] = torch.cat([
                text_cond.negative_pooled_prompt_embeds.to(model_ctx.device),
                text_cond.pooled_prompt_embeds.to(model_ctx.device),
            ])
        if text_cond.add_time_ids is not None:
            full_dict["add_time_ids"] = torch.cat([
                text_cond.add_time_ids.to(model_ctx.device),
                text_cond.add_time_ids.to(model_ctx.device),
            ])
        lat_in = torch.cat([latents] * 2)
        lat_in = model_ctx.scheduler.scale_model_input(lat_in, t)
        with torch.no_grad():
            noise_pred = model_ctx.unet_forward(lat_in, t, full_dict)
        nu, nt = noise_pred.chunk(2)
        noise_pred = nu + guidance_scale * (nt - nu)
        latents = model_ctx.scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]
    return latents
