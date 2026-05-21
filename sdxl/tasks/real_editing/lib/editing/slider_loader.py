"""Load and validate LoRA/slider weights against the active model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

# __file__ = .../sdxl/tasks/real_editing/lib/editing/slider_loader.py → parents[5] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sdxl.core.lora import (  # noqa: E402
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
    LoRANetwork,
)
from sdxl.tasks.real_editing.lib.models.base import ModelContext


def load_slider(
    slider_path: str,
    model_ctx: ModelContext,
    rank: int = 4,
    train_method: str = "noxattn",
) -> LoRANetwork:
    """Load a LoRA slider and attach it to *model_ctx.unet*.

    Validates that the checkpoint key shapes are compatible with the
    active UNet.  Raises ``RuntimeError`` on mismatch.
    """
    # Extend target modules to include conv layers (matches baseline scripts)
    for mod_name in UNET_TARGET_REPLACE_MODULE_CONV:
        if mod_name not in DEFAULT_TARGET_REPLACE:
            DEFAULT_TARGET_REPLACE.append(mod_name)

    network = LoRANetwork(
        model_ctx.unet,
        rank=rank,
        multiplier=1.0,
        alpha=1.0,
        train_method=train_method,
    ).to(model_ctx.device, dtype=model_ctx.dtype)

    ckpt = _load_slider_state_dict(slider_path, map_location=model_ctx.device)
    _validate_shapes(ckpt, network, slider_path, model_ctx.model_type)
    network.load_state_dict(ckpt)
    return network


def _load_slider_state_dict(path, map_location=None):
    """Load a slider state_dict from either a .pt (torch.load) or a
    .safetensors file (safetensors.torch.load_file). .safetensors always
    loads on CPU; call .to(device) on the network afterwards if needed.
    """
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(p)
    return torch.load(p, map_location=map_location)


def _validate_shapes(
    ckpt: dict,
    network: LoRANetwork,
    slider_path: str,
    model_type: str,
) -> None:
    """Check that checkpoint shapes match the network built for this UNet."""
    net_sd = network.state_dict()
    mismatches = []
    for key in ckpt:
        if key in net_sd:
            if ckpt[key].shape != net_sd[key].shape:
                mismatches.append(
                    f"  {key}: ckpt={list(ckpt[key].shape)} vs "
                    f"unet={list(net_sd[key].shape)}"
                )
        # Keys that exist in ckpt but not network could indicate wrong model family
    if mismatches:
        msg = (
            f"Slider checkpoint '{slider_path}' is incompatible with the "
            f"active {model_type} UNet.  Shape mismatches:\n"
            + "\n".join(mismatches)
            + "\n\nMake sure you are using an SDXL slider with an SDXL model "
            "and an SD1.x slider with an SD1.x model."
        )
        raise RuntimeError(msg)

    # Extra sanity: if ckpt has keys not in the network, warn loudly
    extra = set(ckpt.keys()) - set(net_sd.keys())
    if extra:
        import warnings
        warnings.warn(
            f"Slider '{slider_path}' has {len(extra)} keys not present in "
            f"the current LoRANetwork (first 3: {list(extra)[:3]}). "
            "This may indicate a model-family mismatch."
        )
