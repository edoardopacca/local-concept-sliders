"""Factory to instantiate the correct ModelContext for a given model family."""

from __future__ import annotations

import torch

from .base import ModelContext


def load_model_context(
    model_type: str = "sdxl",
    model_id: str | None = None,
    device: str = "cuda",
    dtype: str = "float16",
    local_files_only: bool = False,
) -> ModelContext:
    """Load and return a ModelContext for the requested model family.

    Parameters
    ----------
    model_type : "sdxl" | "sd1x"
    model_id : HuggingFace model id or local snapshot path.  ``None`` picks
               the canonical default for the chosen *model_type*.
    """
    torch_dtype = _str_to_dtype(dtype)

    if model_type == "sdxl":
        from .sdxl import SDXLContext, DEFAULT_SDXL_MODEL

        return SDXLContext(
            model_id=model_id or DEFAULT_SDXL_MODEL,
            device=device,
            dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    raise ValueError(
        f"Unknown model_type={model_type!r}. Supported: sdxl"
    )


def _str_to_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
