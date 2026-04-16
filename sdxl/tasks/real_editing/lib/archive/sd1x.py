"""SD 1.x-specific ModelContext implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from diffusers import DDIMScheduler, StableDiffusionPipeline

from .base import ModelContext, TextCondition

DEFAULT_SD14_MODEL = "CompVis/stable-diffusion-v1-4"
_VAE_SCALE_SD1X = 0.18215


class SD1xContext(ModelContext):

    model_type = "sd1x"

    def __init__(
        self,
        model_id: str = DEFAULT_SD14_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        try:
            self.pipe.disable_xformers_memory_efficient_attention()
        except AttributeError:
            pass

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

    # ------------------------------------------------------------------
    def encode_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        do_cfg: bool = True,
    ) -> TextCondition:
        uncond_input = self.tokenizer(
            [negative_prompt] if do_cfg else [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_emb = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_emb = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
        return TextCondition(
            prompt_embeds=text_emb,
            negative_prompt_embeds=uncond_emb,
        )

    # ------------------------------------------------------------------
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        image = _resize_square(image, 512)
        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_t = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self.vae.dtype)
        )
        with torch.no_grad():
            latents = self.vae.encode(image_t)["latent_dist"].mean
            latents = latents * _VAE_SCALE_SD1X
        return latents

    # ------------------------------------------------------------------
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        latents = (1.0 / _VAE_SCALE_SD1X) * latents.detach()
        latents = latents.to(self.vae.dtype)
        with torch.no_grad():
            image = self.vae.decode(latents)["sample"]
        image = (image / 2.0 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).to(torch.float32).numpy()[0]
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    # ------------------------------------------------------------------
    def unet_forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        text_cond_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.unet(
            latents,
            t,
            encoder_hidden_states=text_cond_dict["prompt_embeds"],
            return_dict=False,
        )[0]

    # ------------------------------------------------------------------
    def default_resolution(self) -> Tuple[int, int]:
        return (512, 512)

    # ------------------------------------------------------------------
    def prepare_ip_adapter(
        self,
        image: Image.Image,
        ipa_scale: float = 0.4,
        weight_name: str = "",
    ) -> Optional[Dict[str, Any]]:
        return None  # IP-Adapter not supported for SD1.x in this pipeline


# ---------------------------------------------------------------------------
def _resize_square(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to square then resize (mirrors load_512 in exp_editing)."""
    img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    arr = np.array(img)[:, :, :3]
    h, w, _ = arr.shape
    if h < w:
        offset = (w - h) // 2
        arr = arr[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        arr = arr[offset : offset + w]
    return Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)
