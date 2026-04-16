"""SDXL-specific ModelContext implementation."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection

from .base import ModelContext, TextCondition

DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class SDXLContext(ModelContext):

    model_type = "sdxl"

    def __init__(
        self,
        model_id: str = DEFAULT_SDXL_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.local_files_only = local_files_only

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            use_safetensors=True,
        ).to(self.device)

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler

        # SDXL VAE is numerically unstable in float16 — always use float32.
        # The memory overhead is ~320 MB (negligible vs the UNet).
        # This avoids the fragile `upcast_vae()` partial-upcast logic and
        # eliminates black/NaN decode outputs that plague float16 SDXL VAE.
        if self.dtype == torch.float16:
            self.vae.to(dtype=torch.float32)
            print("[INFO] SDXL VAE upcast to float32 for numerical stability")

    # ------------------------------------------------------------------
    def encode_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        do_cfg: bool = True,
    ) -> TextCondition:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
        )
        h, w = self.default_resolution()
        # Build add_time_ids manually to avoid version-dependent private API.
        # SDXL time_ids = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        add_time_ids = torch.tensor(
            [[h, w, 0, 0, h, w]], dtype=prompt_embeds.dtype, device=self.device
        )
        return TextCondition(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            add_time_ids=add_time_ids.to(self.device),
        )

    # ------------------------------------------------------------------
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        h, w = self.default_resolution()
        image = _resize_center_crop(image, w, h)
        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_t = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self.vae.dtype)  # float32 (VAE always f32)
        )
        with torch.no_grad():
            latents = self.vae.encode(image_t).latent_dist.mean
            latents = latents * self.vae.config.scaling_factor
        # Cast to model dtype for UNet compatibility downstream
        return latents.to(dtype=self.dtype)

    # ------------------------------------------------------------------
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        # VAE is kept in float32 permanently (see __init__), so just
        # cast input latents to match and decode.  No fragile upcast logic.
        with torch.no_grad():
            latents_f32 = (latents / self.vae.config.scaling_factor).to(
                dtype=self.vae.dtype, device=self.device
            )
            image = self.vae.decode(latents_f32, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(
            image.float(), output_type="pil"
        )[0]

    # ------------------------------------------------------------------
    def unet_forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        text_cond_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        prompt_embeds = text_cond_dict["prompt_embeds"]
        added_cond_kwargs: Dict[str, torch.Tensor] = {}
        if "add_text_embeds" in text_cond_dict:
            added_cond_kwargs["text_embeds"] = text_cond_dict["add_text_embeds"]
        if "add_time_ids" in text_cond_dict:
            added_cond_kwargs["time_ids"] = text_cond_dict["add_time_ids"]
        if "image_embeds" in text_cond_dict:
            added_cond_kwargs["image_embeds"] = text_cond_dict["image_embeds"]
        return self.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
            return_dict=False,
        )[0]

    # ------------------------------------------------------------------
    def default_resolution(self) -> Tuple[int, int]:
        return (1024, 1024)

    # ------------------------------------------------------------------
    def prepare_ip_adapter(
        self,
        image: Image.Image,
        ipa_scale: float = 0.4,
        weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
    ) -> Optional[Dict[str, Any]]:
        try:
            is_vit_h = "vit-h" in weight_name
            image_encoder_folder = (
                "models/image_encoder" if is_vit_h else "sdxl_models/image_encoder"
            )

            # "plus_*_vit-h" weights expect ViT-H projection dims (1280).
            # SDXL pipeline can ship with a different default vision encoder.
            if is_vit_h:
                self.pipe.image_encoder = (
                    CLIPVisionModelWithProjection.from_pretrained(
                        "h94/IP-Adapter",
                        subfolder=image_encoder_folder,
                        local_files_only=self.local_files_only,
                        torch_dtype=self.dtype,
                    ).to(self.device)
                )

            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=weight_name,
                image_encoder_folder=image_encoder_folder,
                local_files_only=self.local_files_only,
            )
            self.pipe.set_ip_adapter_scale(ipa_scale)
        except Exception as e:
            warnings.warn(
                f"[TightInversion] IP-Adapter load failed ({e}). "
                "Falling back to DDIM+GD without image conditioning."
            )
            return None

        image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=image,
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        return {"image_embeds": image_embeds}

    def unload_ip_adapter(self) -> None:
        try:
            self.pipe.unload_ip_adapter()
        except Exception:
            pass


# ---------------------------------------------------------------------------
def _resize_center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    src_w, src_h = img.size
    target_ratio = width / height
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((width, height), Image.Resampling.LANCZOS)
