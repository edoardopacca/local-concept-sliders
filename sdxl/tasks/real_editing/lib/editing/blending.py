"""Blending utilities for masked editing."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def noise_blend(
    eps_base: torch.Tensor,
    eps_lora: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Blend two noise predictions using a spatial mask.

    ``eps_blend = mask * eps_lora + (1 - mask) * eps_base``
    """
    return mask * eps_lora + (1.0 - mask) * eps_base


def feather_mask(
    mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Apply Gaussian blur to a binary mask for soft edges.

    Parameters
    ----------
    mask : (1, 1, H, W) float tensor in [0, 1]
    radius : blur radius in pixels.  0 means no feathering.
    """
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    sigma = radius / 3.0
    mask_np = mask.squeeze().cpu().numpy()
    pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))
    pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=sigma))
    arr = np.array(pil_mask).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, ...].to(mask.device, dtype=mask.dtype)


def pixel_composite(
    edited: Image.Image,
    original: Image.Image,
    mask: torch.Tensor,
) -> Image.Image:
    """Final pixel-space compositing: keep original outside the mask.

    Parameters
    ----------
    mask : (1, 1, H, W) float tensor, will be resized to image size.
    """
    w, h = edited.size
    mask_np = (
        F.interpolate(mask.float(), size=(h, w), mode="bilinear", align_corners=False)
        .squeeze()
        .cpu()
        .numpy()
    )
    mask_np = np.clip(mask_np, 0, 1)[..., None]  # (H, W, 1)

    # The SD pipeline edits a center-cropped/resized view of the original.
    # To avoid stretching, composite in crop space, then paste back.
    crop_box = _center_crop_box_for_aspect(original.size, (w, h))
    original_crop = original.crop(crop_box).resize((w, h), Image.Resampling.LANCZOS)

    ed_arr = np.array(edited).astype(np.float32)
    or_arr = np.array(original_crop).astype(np.float32)
    composite_crop = mask_np * ed_arr + (1.0 - mask_np) * or_arr
    composite_crop_img = Image.fromarray(composite_crop.clip(0, 255).astype(np.uint8))

    out = original.copy()
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    composite_paste = composite_crop_img.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    out.paste(composite_paste, (crop_box[0], crop_box[1]))
    return out


def load_mask(path, target_latent_shape, device, dtype) -> torch.Tensor:
    """Load a grayscale mask image and resize to latent resolution.

    Returns (1, 1, latH, latW) float tensor on *device*.
    """
    mask_img = Image.open(path).convert("L")
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    mask_np = np.clip(mask_np, 0.0, 1.0)
    mask_full = torch.from_numpy(mask_np)[None, None, ...]
    latent_mask = F.interpolate(
        mask_full,
        size=target_latent_shape[-2:],
        mode="nearest",
    ).to(device=device, dtype=dtype)
    return latent_mask


def _center_crop_box_for_aspect(
    src_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Compute center-crop box from src_size to target aspect ratio."""
    src_w, src_h = src_size
    tgt_w, tgt_h = target_size
    target_ratio = tgt_w / tgt_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        return (left, 0, left + new_w, src_h)
    new_h = int(src_w / target_ratio)
    top = (src_h - new_h) // 2
    return (0, top, src_w, top + new_h)
