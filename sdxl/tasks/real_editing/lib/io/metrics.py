"""Compute edit-localization and reconstruction-quality metrics.

Metrics:
  - reconstruction_lpips : LPIPS(original, reconstruction)
  - reconstruction_psnr  : PSNR(original, reconstruction)
  - lpips_inside / lpips_outside / lpips_ratio : edit locality
  - clip_inside / clip_outside : semantic alignment

All metric computations are guarded: if the required model weights are
unavailable (offline HPC), the metric is marked ``"skipped"`` rather
than crashing.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image


def compute_metrics(
    original: Image.Image,
    edited: Image.Image,
    mask: torch.Tensor,
    prompt: str,
    device: str = "cuda",
    reconstruction: Optional[Image.Image] = None,
    skip_metrics: bool = False,
) -> Dict:
    """Compute all available metrics and return a dict for metrics.json."""
    if skip_metrics:
        return {"skipped": True, "reason": "skip_metrics flag set"}

    metrics: Dict = {}

    # --- Reconstruction quality ---
    if reconstruction is not None:
        psnr = _psnr(original, reconstruction)
        metrics["reconstruction_psnr"] = psnr
        lpips_recon = _safe_lpips(original, reconstruction, device)
        if lpips_recon is not None:
            metrics["reconstruction_lpips"] = lpips_recon
        else:
            metrics["reconstruction_lpips"] = "skipped (model unavailable)"

    # --- Edit localization ---
    lpips_in = _safe_lpips_masked(original, edited, mask, inside=True, device=device)
    lpips_out = _safe_lpips_masked(original, edited, mask, inside=False, device=device)
    if lpips_in is not None and lpips_out is not None:
        metrics["lpips_inside"] = lpips_in
        metrics["lpips_outside"] = lpips_out
        metrics["lpips_localization_ratio"] = lpips_in / (lpips_out + 1e-8)
    else:
        metrics["lpips_inside"] = "skipped"
        metrics["lpips_outside"] = "skipped"

    # --- CLIP similarity ---
    clip_in = _safe_clip_masked(edited, mask, prompt, inside=True, device=device)
    clip_out = _safe_clip_masked(edited, mask, prompt, inside=False, device=device)
    if clip_in is not None:
        metrics["clip_inside"] = clip_in
        metrics["clip_outside"] = clip_out
    else:
        metrics["clip_inside"] = "skipped"
        metrics["clip_outside"] = "skipped"

    return metrics


# ======================================================================
# Helpers
# ======================================================================

def _psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    a = np.array(img_a.resize(img_b.size)).astype(np.float64)
    b = np.array(img_b).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def _to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t * 2.0 - 1.0).to(device)


def _safe_lpips(
    img_a: Image.Image,
    img_b: Image.Image,
    device: str,
) -> Optional[float]:
    try:
        import lpips
        model = lpips.LPIPS(net="alex").to(device)
        a = _to_lpips_tensor(img_a.resize(img_b.size), device)
        b = _to_lpips_tensor(img_b, device)
        with torch.no_grad():
            val = model(a, b).item()
        del model
        return float(val)
    except Exception as e:
        warnings.warn(f"LPIPS computation failed: {e}")
        return None


def _safe_lpips_masked(
    img_a: Image.Image,
    img_b: Image.Image,
    mask: torch.Tensor,
    inside: bool,
    device: str,
) -> Optional[float]:
    try:
        import lpips
        model = lpips.LPIPS(net="alex").to(device)
        a = _to_lpips_tensor(img_a.resize(img_b.size), device)
        b = _to_lpips_tensor(img_b, device)
        m = torch.nn.functional.interpolate(
            mask.float(), size=a.shape[-2:], mode="nearest"
        ).to(device)
        if not inside:
            m = 1.0 - m
        with torch.no_grad():
            val = model(a * m, b * m).item()
        del model
        return float(val)
    except Exception as e:
        warnings.warn(f"LPIPS masked computation failed: {e}")
        return None


def _safe_clip_masked(
    img: Image.Image,
    mask: torch.Tensor,
    text: str,
    inside: bool,
    device: str,
) -> Optional[float]:
    try:
        from transformers import CLIPModel, CLIPProcessor

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        arr = np.array(img, dtype=np.uint8)
        mask_np = torch.nn.functional.interpolate(
            mask.float(),
            size=(arr.shape[0], arr.shape[1]),
            mode="nearest",
        ).squeeze().cpu().numpy()[..., None]
        if not inside:
            mask_np = 1.0 - mask_np
        gray = np.full_like(arr, 127, dtype=np.uint8)
        region = (arr * mask_np + gray * (1.0 - mask_np)).astype(np.uint8)
        region_pil = Image.fromarray(region)

        inputs = clip_proc(
            text=[text], images=region_pil, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            out = clip_model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        val = float((ie * te).sum(dim=-1).item())
        del clip_model, clip_proc
        return val
    except Exception as e:
        warnings.warn(f"CLIP masked computation failed: {e}")
        return None
