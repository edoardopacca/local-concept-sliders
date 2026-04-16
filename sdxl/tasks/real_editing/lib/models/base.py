"""Abstract model context that isolates SD1.x vs SDXL differences."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image


@dataclass
class TextCondition:
    """Unified container for all text-conditioning tensors needed by the UNet."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    # SDXL-only fields (None for SD1.x)
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    add_time_ids: Optional[torch.Tensor] = None

    def cfg_embeds(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return prompt_embeds concatenated for CFG [uncond, cond]."""
        pe = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0).to(device)
        out: Dict[str, torch.Tensor] = {"prompt_embeds": pe}
        if self.pooled_prompt_embeds is not None:
            out["add_text_embeds"] = torch.cat(
                [self.negative_pooled_prompt_embeds, self.pooled_prompt_embeds], dim=0
            ).to(device)
        if self.add_time_ids is not None:
            out["add_time_ids"] = torch.cat(
                [self.add_time_ids, self.add_time_ids], dim=0
            ).to(device)
        return out

    def nocfg_embeds(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return prompt_embeds without CFG duplication (cond only)."""
        out: Dict[str, torch.Tensor] = {"prompt_embeds": self.prompt_embeds.to(device)}
        if self.pooled_prompt_embeds is not None:
            out["add_text_embeds"] = self.pooled_prompt_embeds.to(device)
        if self.add_time_ids is not None:
            out["add_time_ids"] = self.add_time_ids.to(device)
        return out


class ModelContext(ABC):
    """Abstraction over a diffusion model family (SD1.x / SDXL).

    Sub-classes must implement every @abstractmethod.  This lets the
    inversion and editing code stay model-agnostic.
    """

    model_type: str  # "sdxl" | "sd1x"
    model_id: str
    device: torch.device
    dtype: torch.dtype

    # ---- Pipeline components (set by sub-class __init__) ----
    pipe: Any  # the underlying DiffusionPipeline
    unet: Any
    vae: Any
    scheduler: Any

    @abstractmethod
    def encode_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        do_cfg: bool = True,
    ) -> TextCondition:
        """Encode text prompts into model-specific conditioning tensors."""

    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """VAE-encode an RGB PIL image to latent tensor (scaled)."""

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """VAE-decode latent tensor to a PIL image (handles upcast if needed)."""

    @abstractmethod
    def unet_forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        text_cond_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Single UNet forward pass returning raw noise prediction."""

    @abstractmethod
    def default_resolution(self) -> Tuple[int, int]:
        """Return (height, width) native to this model family."""

    @abstractmethod
    def prepare_ip_adapter(
        self,
        image: Image.Image,
        ipa_scale: float,
        weight_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Load IP-Adapter weights and return image_embeds dict, or None if unsupported."""
