"""Abstract inversion backend and result data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from sdxl.tasks.real_editing.lib.models.base import ModelContext, TextCondition


@dataclass
class InversionConfig:
    """Parameters shared by all inversion backends."""

    steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 1234
    # Tight Inversion / GD-specific
    num_gd_steps: int = 0
    gd_step_size: float = 0.001
    optimization_start: int = 0
    # IP-Adapter
    use_ipa: bool = False
    ipa_scale: float = 0.4
    ipa_weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    # Null-text specific
    num_inner_steps: int = 10
    early_stop_epsilon: float = 1e-5


@dataclass
class InversionResult:
    """Standardised output of any inversion backend."""

    schema_version: str = "1.0"
    model_family: str = ""
    model_id: str = ""
    inversion_backend: str = ""
    backend_status: str = "implemented"

    x_t: Optional[torch.Tensor] = None
    all_latents: Optional[List[torch.Tensor]] = None
    original_image: Optional[Image.Image] = None
    reconstruction_image: Optional[Image.Image] = None

    uncond_embeddings: Optional[torch.Tensor] = None
    text_condition: Optional[TextCondition] = None

    config: Optional[InversionConfig] = None
    source_manifest: Dict[str, Any] = field(default_factory=dict)

    # Tight Inversion IP-Adapter specific
    ip_adapter_image_embeds: Optional[Any] = None


class InversionBackend(ABC):
    """Interface that every inversion method must implement."""

    name: str = "base"
    status: str = "stub"  # "implemented" | "experimental" | "stub"
    description: str = ""

    @abstractmethod
    def invert(
        self,
        model_ctx: ModelContext,
        image: Image.Image,
        prompt: str,
        config: InversionConfig,
        negative_prompt: str = "",
    ) -> InversionResult:
        """Run inversion and return standardised artifacts."""

    def provenance(self) -> Dict[str, str]:
        """Return provenance metadata for EXTERNAL_SOURCES.md tracking."""
        return {
            "name": self.name,
            "status": self.status,
            "source": "unknown",
            "url": "",
            "commit": "",
            "license": "",
        }
