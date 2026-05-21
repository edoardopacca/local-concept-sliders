#!/usr/bin/env python3
"""
Phase 3 of the external mask-guided pipeline on Flux: multi-path
velocity blend with N disjoint masks and M sliders per mask.

For each denoising timestep:
  1) v_base       = transformer(latents, LoRA disabled)
  2) For each mask i:
       activate its sliders as PEFT adapters with scales `s_ij`
       v_styled_i = transformer(latents, adapter set i active)
  3) v_pred = (1 - sum_i mask_i) * v_base + sum_i (mask_i * v_styled_i)
     latents = scheduler.step(v_pred, t, latents)

The forward passes are physically independent: there is no leak via
self-attention because the blend happens OUTSIDE the transformer.
Composing several sliders inside the same mask is the PEFT equivalent of
the compositional aggregation used in shop_concept
(out = W·x + sum_j scale_j · B_j A_j · x).

Cost: (1 + N) forwards per step. N=1 -> 2x. N=2 -> 3x. N=3 -> 4x. ...

Supported modes:
  * Legacy single-mask single-slider:
      --mask_name mask.png --slider_path s.pt
      [--slider_scale 1.0 | --slider_scales -2 -1 0 1 2]
    With --slider_scales, the script SWEEPS: one PNG per scale, reusing
    the same Flux load.
  * Multi-mask multi-slider:
      --mask_names m_man.png m_woman.png
      --slider_paths smile.pt age.pt vangogh.pt
      --slider_to_mask 0 0 1
      --slider_scales 1.0 2.0 0.8
    Here --slider_scales has one value per slider (no sweep). Sliders 0
    and 1 are composed additively inside mask 0; slider 2 is applied
    inside mask 1. The region outside every mask stays at the baseline.

Inputs expected in run_dir (produced by phase 1 and 2):
  - base.png        (phase 1)
  - metadata.json   (phase 1; seed / prompt / steps / scheduler_config)
  - mask.png or mask_man.png / mask_woman.png / ... (phase 2 SAM masks)

Output:
  - edited.png, edited_scaleX.png (legacy sweep), or edited_compose.png
    (multi-mask mode)
  - edit_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxPipeline
from safetensors.torch import load_file

# Reuse the slider .pt -> PEFT safetensors conversion from shop_concept.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
from flux.tasks.shop_concept.scripts.generate import (  # noqa: E402
    ensure_matching_lora_params,
    prepare_slider_as_safetensors,
)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Phase 3: Flux multi-path mask-guided blend "
        "(N disjoint masks x M sliders per mask)."
    )
    parser.add_argument("--run_dir", type=Path, required=True)

    # Slider: 1 (legacy) or N (multi)
    parser.add_argument(
        "--slider_path", type=str, default=None,
        help="Legacy mode: a single slider. Mutually exclusive with "
             "--slider_paths.",
    )
    parser.add_argument(
        "--slider_paths", type=str, nargs="+", default=None,
        help="Multi-mode: list of sliders (.pt or .safetensors).",
    )

    # Scale: one value (legacy single) | sweep (legacy sweep) |
    #        N values (multi, one per slider)
    parser.add_argument(
        "--slider_scale", type=float, default=3.0,
        help="Legacy single-slider: one scale. Default 3.0.",
    )
    parser.add_argument(
        "--slider_scales", type=float, nargs="+", default=None,
        help=(
            "Legacy single-slider: SWEEP, one PNG per scale. "
            "Multi-mode: one scale per slider (no sweep)."
        ),
    )

    # Mask: 1 (legacy) or N (multi)
    parser.add_argument(
        "--mask_name", type=str, default=None,
        help="Legacy mode: a single mask (default mask.png). Mutually "
             "exclusive with --mask_names.",
    )
    parser.add_argument(
        "--mask_names", type=str, nargs="+", default=None,
        help="Multi-mode: list of N masks (binary PNGs inside run_dir).",
    )

    # Mapping slider -> mask (multi only)
    parser.add_argument(
        "--slider_to_mask", type=int, nargs="+", default=None,
        help="Multi-mode: slider_to_mask[i]=j means that slider i is "
             "applied inside the region of mask j. Length must match "
             "--slider_paths. Several sliders on the same mask compose "
             "additively (compositional aggregation).",
    )

    parser.add_argument(
        "--edit_start_step",
        type=int,
        default=8,
        help=(
            "Step at which the multi-path blend with LoRA becomes active. "
            "If >0, the first N steps run a single forward without LoRA. "
            "Default 8: aligned with shop_concept and the baseline so the "
            "comparisons are fair (22/30 active steps). 0 = LoRA active "
            "from step 0."
        ),
    )
    parser.add_argument(
        "--lora_fill_rank",
        type=int,
        default=16,
        help="LoRA rank used for the PEFT zero-padding.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="flux/tasks/masked_lora/_peft_cache",
    )
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="Override model_id from metadata.json when provided.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_name", type=str, default="edited.png",
                        help="Output filename (legacy single or multi).")
    return parser


# =============================================================================
# Helpers
# =============================================================================

def load_run_inputs(
    run_dir: Path, mask_names: list
) -> Tuple[Dict, list]:
    """Load metadata.json and a LIST of masks (binary grayscale PNGs).

    Returns ``(metadata, [mask_tensor_1, mask_tensor_2, ...])`` where each
    mask_tensor has shape ``(1, 1, H, W)`` with values in ``[0, 1]``.
    """
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    masks = []
    for mname in mask_names:
        mp = run_dir / mname
        if not mp.exists():
            raise FileNotFoundError(f"Missing mask: {mp}")
        mask_img = Image.open(mp).convert("L")
        mask_np = (np.array(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
        masks.append(torch.from_numpy(mask_np)[None, None, ...])
    return metadata, masks


def pack_mask_for_flux(
    mask_pixel: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a pixel-space binary mask ``(1, 1, H, W)`` into a mask in
    Flux's packed latent-token space ``(1, N_tokens, 1)``.

    Flux packing: ``(B, 16, H/8, W/8)`` unpacked -> ``(B, N_tokens, 64)``
    packed, with ``N_tokens = (H/16) * (W/16)`` (each token corresponds to
    a 2x2 block in latent space = 16x16 block in pixel space).

    The pixel mask is downscaled to token resolution with area averaging
    and then binarised at threshold 0.5, so mask boundaries that cross a
    token partially are assigned to the majority side.
    """
    token_h = height // 16
    token_w = width // 16

    mask_down = F.interpolate(
        mask_pixel, size=(token_h, token_w), mode="area"
    )  # (1, 1, token_h, token_w) soft in [0, 1]
    mask_bin = (mask_down > 0.5).to(dtype=dtype)
    # flatten to (1, N_tokens, 1)
    mask_flat = mask_bin.view(1, token_h * token_w, 1).to(device=device)
    return mask_flat


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Same shift computation as diffusers.pipelines.flux.pipeline_flux."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# =============================================================================
# Core: dual-path denoising
# =============================================================================

@torch.no_grad()
def masked_multi_path_denoise(
    pipe: FluxPipeline,
    metadata: Dict,
    mask_pixels: list,
    mask_to_adapters: list,
    edit_start_step: int,
    device: torch.device,
) -> Image.Image:
    """Multi-path velocity blend with N disjoint masks.

    Args:
        mask_pixels: list of N tensors with shape ``(1, 1, H, W)`` and
            values in ``[0, 1]``.
        mask_to_adapters: list of N tuples ``(adapter_names: List[str],
            adapter_weights: List[float])`` describing the sliders active
            inside each region.
    """
    assert len(mask_pixels) == len(mask_to_adapters), \
        f"mask_pixels ({len(mask_pixels)}) != mask_to_adapters ({len(mask_to_adapters)})"
    num_masks = len(mask_pixels)

    seed = int(metadata["seed"])
    prompt = metadata["prompt"]
    steps = int(metadata["steps"])
    guidance_scale = float(metadata["guidance_scale"])
    height = int(metadata["height"])
    width = int(metadata["width"])
    max_seq_len = int(metadata.get("max_sequence_length", 256))

    dtype = pipe.transformer.dtype

    # ---- Encode prompts (T5 + CLIP) ----
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_seq_len,
    )

    # ---- Prepare latents (packed) + image position ids ----
    num_channels_latents = pipe.transformer.config.in_channels // 4
    generator = torch.Generator(device=device).manual_seed(seed)
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
    )

    # ---- Pack each mask in Flux packed-latent-token space ----
    masks_packed = [
        pack_mask_for_flux(m, height=height, width=width,
                           device=device, dtype=latents.dtype)
        for m in mask_pixels
    ]
    for idx, mp in enumerate(masks_packed):
        cov = float(mp.float().mean().item())
        n_in = int(mp.sum().item())
        print(f"[phase3] mask[{idx}] coverage(packed)={cov:.3f}  "
              f"tokens={n_in}/{mp.shape[1]}  "
              f"adapters={mask_to_adapters[idx][0]} "
              f"scales={mask_to_adapters[idx][1]}")

    # Mask overlap warning (sum > 1 means some tokens belong to more
    # than one mask).
    if num_masks > 1:
        sum_masks = torch.zeros_like(masks_packed[0])
        for mp in masks_packed:
            sum_masks = sum_masks + mp
        n_overlap = int((sum_masks > 1).sum().item())
        if n_overlap > 0:
            print(f"[phase3] WARNING: {n_overlap} token(s) belong to "
                  f"more than one mask (sum>1). The blend formula assumes "
                  f"disjoint masks; on overlapping tokens the effect is "
                  f"the additive sum of deltas, which can saturate. "
                  f"Check the SAM masks.")

    # ---- Prepare timesteps (Flux-specific shift) ----
    import numpy as _np
    sigmas = _np.linspace(1.0, 1.0 / steps, steps)
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(image_seq_len)
    pipe.scheduler.set_timesteps(
        num_inference_steps=steps, device=device, sigmas=sigmas, mu=mu
    )
    timesteps = pipe.scheduler.timesteps

    # ---- Guidance embedding (Flux-dev uses distilled guidance) ----
    if pipe.transformer.config.guidance_embeds:
        guidance = (
            torch.tensor([guidance_scale], device=device)
            .expand(latents.shape[0])
            .to(dtype)
        )
    else:
        guidance = None

    print(
        f"[phase3] denoising: steps={steps} "
        f"edit_start_step={edit_start_step} "
        f"num_masks={num_masks}  "
        f"forward_per_step={1 + num_masks} (1 base + {num_masks} per-mask)"
    )

    def _fwd(latents_in, timestep_in):
        return pipe.transformer(
            hidden_states=latents_in,
            timestep=timestep_in,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

    # ---- Denoising loop ----
    for i, t in enumerate(timesteps):
        timestep_in = t.expand(latents.shape[0]).to(dtype) / 1000.0

        if i < edit_start_step:
            # Vanilla pass: niente LoRA, niente blend
            pipe.disable_lora()
            v_pred = _fwd(latents, timestep_in)
        else:
            # Forward base (LoRA off)
            pipe.disable_lora()
            v_base = _fwd(latents, timestep_in)

            # One forward per mask with the corresponding adapter set
            # active. PEFT.set_adapters([names], adapter_weights=[ws]) puts
            # every named adapter into the active list and the forward
            # sums their deltas, which is the compositional aggregation.
            v_styled_list = []
            for mask_idx in range(num_masks):
                names, weights = mask_to_adapters[mask_idx]
                pipe.enable_lora()
                pipe.set_adapters(list(names),
                                  adapter_weights=[float(w) for w in weights])
                v_styled_list.append(_fwd(latents, timestep_in))

            # Velocity blend: outside every mask stays on v_base; inside
            # mask i we take v_styled_i (assumes disjoint masks).
            v_pred = v_base.clone()
            sum_masks = torch.zeros_like(masks_packed[0])
            for mask_idx in range(num_masks):
                m = masks_packed[mask_idx]
                v_pred = v_pred - m * v_base + m * v_styled_list[mask_idx]
                sum_masks = sum_masks + m
            # Equivalent to: v_pred = (1 - sum_masks) * v_base
            #                       + sum_i (mask_i * v_styled_i)
            # when the masks are disjoint. If they overlap, the additive
            # sum of deltas on the overlapping tokens still corresponds to
            # the natural compositional aggregation (overlap warning has
            # already been emitted upstream).

        # Scheduler step (flow matching Euler)
        latents = pipe.scheduler.step(v_pred, t, latents, return_dict=False)[0]

        if (i + 1) % 5 == 0 or i == len(timesteps) - 1:
            print(f"  step {i + 1}/{len(timesteps)}  t={float(t):.4f}")

    # ---- Decode ----
    latents_unpacked = pipe._unpack_latents(
        latents, height, width, pipe.vae_scale_factor
    )
    latents_unpacked = (
        latents_unpacked / pipe.vae.config.scaling_factor
    ) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents_unpacked, return_dict=False)[0]
    image_pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image_pil


# =============================================================================
# Main
# =============================================================================

def _normalize_args(args) -> Tuple[list, list, list, list, bool, bool]:
    """Normalizza gli input CLI in forma multi-mode unificata.

    Ritorna:
      slider_paths      : List[str]
      mask_names        : List[str]
      slider_to_mask    : List[int]   (mapping slider_idx -> mask_idx)
      sweep_scales      : List[float] (>1 elem solo se modalita' legacy sweep)
      is_multi          : bool        (True se multi-mask o multi-slider)
      is_sweep          : bool        (True se legacy single + slider_scales sweep)
    """
    has_slider_paths = args.slider_paths is not None
    has_slider_path = args.slider_path is not None
    has_mask_names = args.mask_names is not None
    has_mask_name = args.mask_name is not None
    has_slider_to_mask = args.slider_to_mask is not None

    # Mode selection: any plural flag activates multi-mode.
    is_multi = has_slider_paths or has_mask_names or has_slider_to_mask

    if is_multi:
        if has_slider_path or has_mask_name:
            raise ValueError(
                "Mixing legacy (--slider_path/--mask_name) and multi "
                "(--slider_paths/--mask_names/--slider_to_mask) is "
                "ambiguous. Use ONE form or the other."
            )
        if not has_slider_paths:
            raise ValueError("Multi-mode richiede --slider_paths.")
        if not has_mask_names:
            raise ValueError("Multi-mode richiede --mask_names.")
        if not has_slider_to_mask:
            raise ValueError(
                "Multi-mode richiede --slider_to_mask (mapping slider->mask)."
            )
        slider_paths = list(args.slider_paths)
        mask_names = list(args.mask_names)
        slider_to_mask = list(args.slider_to_mask)
        n_sliders = len(slider_paths)
        n_masks = len(mask_names)
        if len(slider_to_mask) != n_sliders:
            raise ValueError(
                f"--slider_to_mask has {len(slider_to_mask)} values, "
                f"expected {n_sliders} (one per --slider_paths)."
            )
        for s_idx, m_idx in enumerate(slider_to_mask):
            if not (0 <= m_idx < n_masks):
                raise ValueError(
                    f"--slider_to_mask[{s_idx}]={m_idx} is out of range "
                    f"[0, {n_masks}); there are {n_masks} masks."
                )
        if args.slider_scales is None:
            raise ValueError(
                "Multi-mode requires --slider_scales (one value per slider)."
            )
        if len(args.slider_scales) != n_sliders:
            raise ValueError(
                f"Multi-mode: --slider_scales has {len(args.slider_scales)} "
                f"values, expected {n_sliders} (one per slider)."
            )
        # Each mask must have at least one slider mapped to it.
        masks_used = set(slider_to_mask)
        for m in range(n_masks):
            if m not in masks_used:
                raise ValueError(
                    f"Mask {m} ('{mask_names[m]}') has no slider mapped "
                    f"to it. Add at least one slider that maps to this "
                    f"mask, or remove the mask."
                )
        return slider_paths, mask_names, slider_to_mask, [], True, False

    # ---- Legacy mode (single mask + single slider) ----
    if not has_slider_path:
        raise ValueError(
            "Serve --slider_path (legacy) oppure --slider_paths (multi)."
        )
    slider_paths = [args.slider_path]
    mask_names = [args.mask_name if has_mask_name else "mask.png"]
    slider_to_mask = [0]
    if args.slider_scales is not None and len(args.slider_scales) > 0:
        sweep_scales = list(args.slider_scales)
        return slider_paths, mask_names, slider_to_mask, sweep_scales, False, True
    sweep_scales = [args.slider_scale]
    return slider_paths, mask_names, slider_to_mask, sweep_scales, False, False


def main() -> None:
    args = build_parser().parse_args()

    (slider_paths, mask_names, slider_to_mask,
     sweep_scales, is_multi, is_sweep) = _normalize_args(args)

    print(f"[phase3] run_dir: {args.run_dir}")
    print(f"[phase3] mode: {'multi' if is_multi else ('legacy-sweep' if is_sweep else 'legacy-single')}")

    metadata, mask_pixels = load_run_inputs(args.run_dir, mask_names)
    for idx, mp in enumerate(mask_pixels):
        print(f"[phase3] mask[{idx}] '{mask_names[idx]}' shape: "
              f"{tuple(mp.shape)} (pixel coverage: {float(mp.mean()):.3f})")

    device_obj = torch.device(args.device)
    torch.manual_seed(int(metadata["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(metadata["seed"]))

    # ---- Load Flux pipeline ----
    resolved_model_id = (
        args.model_id
        or metadata.get("model_id")
        or "black-forest-labs/FLUX.1-dev"
    )
    print(f"[phase3] loading Flux: {resolved_model_id}")
    pipe = FluxPipeline.from_pretrained(
        resolved_model_id, torch_dtype=torch.bfloat16
    ).to(args.device)

    # ---- Load slider(s) as PEFT adapter(s) ----
    # Adapter naming convention: default_0, default_1, ... aligned with
    # shop_concept. The same slider file can be loaded multiple times as
    # distinct adapters when it needs to be applied to several masks with
    # different scales.
    print(f"[phase3] preparing {len(slider_paths)} slider(s)")
    slider_safetensors_list = [
        prepare_slider_as_safetensors(p, args.cache_dir) for p in slider_paths
    ]
    lora_dicts = [load_file(p) for p in slider_safetensors_list]
    lora_dicts = ensure_matching_lora_params(
        lora_dicts, rank=args.lora_fill_rank
    )
    adapter_names = []
    for i, lora_dict in enumerate(lora_dicts):
        name = f"default_{i}"
        pipe.load_lora_weights(lora_dict, adapter_name=name)
        adapter_names.append(name)
        m_idx = slider_to_mask[i]
        print(f"  [slider {i}] adapter='{name}' -> mask[{m_idx}] "
              f"path={slider_paths[i]}")

    # ---- Run(s) ----
    per_run_outputs = []

    if is_multi:
        # Costruisci mapping mask_idx -> (adapter_names, scales)
        mask_to_adapters = []
        for m_idx in range(len(mask_names)):
            names = []
            weights = []
            for s_idx, mapped_m in enumerate(slider_to_mask):
                if mapped_m == m_idx:
                    names.append(adapter_names[s_idx])
                    weights.append(float(args.slider_scales[s_idx]))
            mask_to_adapters.append((names, weights))

        print(f"[phase3] === multi-mask compose run ===")
        for m_idx, (names, weights) in enumerate(mask_to_adapters):
            print(f"  mask[{m_idx}] '{mask_names[m_idx]}': "
                  f"adapters={names} weights={weights}")

        image_pil = masked_multi_path_denoise(
            pipe=pipe,
            metadata=metadata,
            mask_pixels=mask_pixels,
            mask_to_adapters=mask_to_adapters,
            edit_start_step=args.edit_start_step,
            device=device_obj,
        )
        out_name = args.output_name
        edited_path = args.run_dir / out_name
        image_pil.save(edited_path)
        print(f"[OK] Edited image: {edited_path}")
        per_run_outputs.append({
            "output": out_name,
            "mask_to_adapters": [
                {"mask": mask_names[m], "adapters": names, "weights": weights}
                for m, (names, weights) in enumerate(mask_to_adapters)
            ],
        })
    else:
        # Legacy: 1 slider + 1 mask, sweep o single
        scales = sweep_scales
        print(f"[phase3] {'SWEEP' if is_sweep else 'SINGLE'} mode: "
              f"scales={scales}")
        for s in scales:
            print(f"[phase3] === running scale={s} ===")
            mask_to_adapters = [(adapter_names, [float(s)])]
            image_pil = masked_multi_path_denoise(
                pipe=pipe,
                metadata=metadata,
                mask_pixels=mask_pixels,
                mask_to_adapters=mask_to_adapters,
                edit_start_step=args.edit_start_step,
                device=device_obj,
            )
            if not is_sweep:
                out_name = args.output_name
            else:
                out_name = f"edited_scale{float(s):.1f}.png"
            edited_path = args.run_dir / out_name
            image_pil.save(edited_path)
            print(f"[OK] Edited image: {edited_path}")
            per_run_outputs.append({"scale": float(s), "output": out_name})

    edit_meta = {
        "phase": "masked_edit",
        "method": "multi-path velocity-blend (MaskedLoRA Flux, "
                  "multi-mask + multi-LoRA composition paper-style)",
        "created_at": datetime.utcnow().isoformat(),
        "mode": "multi" if is_multi else ("legacy-sweep" if is_sweep else "legacy-single"),
        "slider_paths": slider_paths,
        "mask_names": mask_names,
        "slider_to_mask": slider_to_mask,
        "edit_start_step": args.edit_start_step,
        "model_id": resolved_model_id,
        "outputs": per_run_outputs,
    }
    (args.run_dir / "edit_meta.json").write_text(
        json.dumps(edit_meta, indent=2), encoding="utf-8"
    )
    print(f"[OK] edit_meta.json written")


if __name__ == "__main__":
    main()
