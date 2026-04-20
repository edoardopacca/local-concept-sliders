#!/usr/bin/env python3
"""
masked_Lora_FLUX/03_masked_edit.py
===================================
Phase 3 - MaskedLoRA dual-path ε-blend (velocity-blend in Flux, since Flux
is rectified flow) per Flux.1-dev.

Per ogni timestep del denoising, fa DUE forward pass del transformer:
  1) v_base   = transformer(latents, LoRA_off)
  2) v_styled = transformer(latents, LoRA_on_full_scale)
poi blenda nello spazio della velocity secondo la mask (convertita a
packed-latent-token space):
     v_final = mask * v_styled + (1 - mask) * v_base
     latents = scheduler.step(v_final, t, latents)

Perche' cosi' niente leak via self-attention: i due forward sono
fisicamente indipendenti, il blend avviene FUORI dal transformer.

Costo: 2x compute per step rispetto a generazione normale.

Inputs attesi nella run_dir (prodotti dai phase 1 e 2):
  - base.png        (phase 1)
  - metadata.json   (phase 1, per seed/prompt/steps/scheduler_config)
  - mask.png        (phase 2, binary mask pixel-level)

Output:
  - edited.png
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

# Riuso della conversione slider .pt -> PEFT safetensors da shop_concept
# __file__ = .../flux/tasks/masked_lora/scripts/03_masked_edit.py → parents[4] = repo root
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
        "Phase 3 - MaskedLoRA Flux dual-path blend"
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--slider_path", type=str, required=True)
    parser.add_argument("--slider_scale", type=float, default=3.0)
    parser.add_argument(
        "--slider_scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Se fornito, ignora --slider_scale e fa uno sweep salvando "
            "un file per scale (edited_scale{S}.png). Pipe+slider+mask "
            "vengono riusati: UN solo load di Flux per tutto lo sweep."
        ),
    )
    parser.add_argument(
        "--edit_start_step",
        type=int,
        default=0,
        help=(
            "Step dal quale attivare il dual-path con LoRA. "
            "Se >0, i primi N step girano single-forward senza LoRA. "
            "0 = LoRA attivo da subito (default)."
        ),
    )
    parser.add_argument(
        "--lora_fill_rank",
        type=int,
        default=16,
        help="Rank del LoRA (per il padding PEFT).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="flux/tasks/masked_lora/_peft_cache",
    )
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="Sovrascrive il model_id da metadata.json se fornito.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_name", type=str, default="edited.png")
    parser.add_argument("--mask_name", type=str, default="mask.png")
    return parser


# =============================================================================
# Helpers
# =============================================================================

def load_run_inputs(
    run_dir: Path, mask_name: str
) -> Tuple[Dict, torch.Tensor]:
    metadata_path = run_dir / "metadata.json"
    mask_path = run_dir / mask_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing {mask_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mask_img = Image.open(mask_path).convert("L")
    mask_np = (np.array(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    mask_tensor = torch.from_numpy(mask_np)[None, None, ...]  # (1, 1, H, W)
    return metadata, mask_tensor


def pack_mask_for_flux(
    mask_pixel: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Converte una mask pixel-space (1, 1, H, W) binary in una mask per packed
    Flux latent tokens (1, N_tokens, 1).

    Flux packing: (B, 16, H/8, W/8) unpacked -> (B, N_tokens, 64) packed
    dove N_tokens = (H/16) * (W/16) (ogni token = blocco 2x2 nel latent
    space = blocco 16x16 nel pixel space).

    Usiamo downscale area (media) a risoluzione token, poi binarizziamo
    con soglia 0.5. In questo modo i bordi della mask (che spesso attraversano
    token parzialmente) vengono assegnati al lato maggioritario.
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
    """Stesso calcolo di diffusers.pipelines.flux.pipeline_flux."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# =============================================================================
# Core: dual-path denoising
# =============================================================================

@torch.no_grad()
def masked_dual_path_denoise(
    pipe: FluxPipeline,
    metadata: Dict,
    mask_pixel: torch.Tensor,
    slider_scale: float,
    edit_start_step: int,
    device: torch.device,
) -> Image.Image:
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

    # ---- Prepare mask for Flux packed latent space ----
    mask_packed = pack_mask_for_flux(
        mask_pixel, height=height, width=width, device=device, dtype=latents.dtype
    )
    coverage = float(mask_packed.float().mean().item())
    print(f"[phase3] mask coverage (packed): {coverage:.3f}  "
          f"(tokens in mask = {int(mask_packed.sum().item())}/{mask_packed.shape[1]})")

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

    # ---- Determine if LoRA should be engaged from a given step ----
    # edit_start_step=0 -> dual-path from the beginning (recommended)
    # edit_start_step>0 -> first N steps single-pass vanilla, then dual-path
    print(
        f"[phase3] denoising: steps={steps} "
        f"edit_start_step={edit_start_step} "
        f"slider_scale={slider_scale}"
    )

    # ---- Denoising loop ----
    for i, t in enumerate(timesteps):
        timestep_in = t.expand(latents.shape[0]).to(dtype) / 1000.0

        if i < edit_start_step:
            # Single-forward vanilla (no LoRA, no blend)
            pipe.disable_lora()
            v_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_in,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
        else:
            # --- Path A: NO LoRA ---
            pipe.disable_lora()
            v_base = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_in,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # --- Path B: LoRA ATTIVO a slider_scale ---
            pipe.enable_lora()
            pipe.set_adapters(["slider"], adapter_weights=[float(slider_scale)])
            v_styled = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_in,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # --- BLEND nello spazio velocity ---
            v_pred = mask_packed * v_styled + (1.0 - mask_packed) * v_base

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

def main() -> None:
    args = build_parser().parse_args()

    print(f"[phase3] run_dir: {args.run_dir}")
    metadata, mask_pixel = load_run_inputs(args.run_dir, args.mask_name)
    print(f"[phase3] mask pixel shape: {tuple(mask_pixel.shape)} "
          f"(pixel coverage: {float(mask_pixel.mean()):.3f})")

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

    # ---- Load slider as PEFT adapter ----
    print(f"[phase3] preparing slider: {args.slider_path}")
    slider_safetensors = prepare_slider_as_safetensors(
        args.slider_path, args.cache_dir
    )
    lora_dicts = [load_file(slider_safetensors)]
    lora_dicts = ensure_matching_lora_params(
        lora_dicts, rank=args.lora_fill_rank
    )
    pipe.load_lora_weights(lora_dicts[0], adapter_name="slider")
    print(f"[phase3] slider loaded as adapter='slider'")

    # ---- Dual-path denoising (single or sweep) ----
    if args.slider_scales is not None and len(args.slider_scales) > 0:
        scales = list(args.slider_scales)
        print(f"[phase3] SWEEP mode: scales={scales}")
    else:
        scales = [args.slider_scale]
        print(f"[phase3] SINGLE mode: scale={args.slider_scale}")

    per_scale_outputs = []
    for s in scales:
        print(f"[phase3] === running scale={s} ===")
        image_pil = masked_dual_path_denoise(
            pipe=pipe,
            metadata=metadata,
            mask_pixel=mask_pixel,
            slider_scale=float(s),
            edit_start_step=args.edit_start_step,
            device=device_obj,
        )
        if len(scales) == 1 and args.slider_scales is None:
            out_name = args.output_name
        else:
            # sweep: un file per scale
            out_name = f"edited_scale{float(s):.1f}.png"
        edited_path = args.run_dir / out_name
        image_pil.save(edited_path)
        print(f"[OK] Edited image: {edited_path}")
        per_scale_outputs.append({"scale": float(s), "output": out_name})

    edit_meta = {
        "phase": "masked_edit",
        "method": "dual-path velocity-blend (MaskedLoRA port to Flux)",
        "created_at": datetime.utcnow().isoformat(),
        "slider_path": args.slider_path,
        "edit_start_step": args.edit_start_step,
        "mask_image": args.mask_name,
        "model_id": resolved_model_id,
        "outputs": per_scale_outputs,
    }
    (args.run_dir / "edit_meta.json").write_text(
        json.dumps(edit_meta, indent=2), encoding="utf-8"
    )
    print(f"[OK] edit_meta.json written")


if __name__ == "__main__":
    main()
