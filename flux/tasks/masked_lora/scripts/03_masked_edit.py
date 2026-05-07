#!/usr/bin/env python3
"""
masked_Lora_FLUX/03_masked_edit.py
===================================
Phase 3 - MaskedLoRA multi-path velocity-blend per Flux.1-dev.

Generalizzato a N maschere disgiunte + M slider per maschera (composizione
paper-style via PEFT multi-adapter). Per ogni timestep:
  1) v_base       = transformer(latents, LoRA disabilitato)
  2) Per ogni maschera i:
       attiva i suoi slider come adapter PEFT con scale `s_ij`
       v_styled_i = transformer(latents, adapter set i attivo)
  3) v_pred = (1 - sum_i mask_i) * v_base + sum_i (mask_i * v_styled_i)
     latents = scheduler.step(v_pred, t, latents)

I forward sono fisicamente indipendenti: nessun leak via self-attention,
il blend avviene FUORI dal transformer. La composizione di piu' slider
sulla stessa maschera e' il PEFT-equivalente del Metodo 2 di Concept
Sliders (out = W*x + sum_j scale_j * B_j A_j x), uguale a shop_concept.

Costo: (1 + N) forward per step. N=1 -> 2x (uguale al codice originale).
N=2 -> 3x. N=3 -> 4x. ecc.

Modalita' supportate:
  * Legacy single-mask single-slider:
      --mask_name mask.png --slider_path s.pt [--slider_scale 1.0 |
                                               --slider_scales -2 -1 0 1 2]
    Lo `--slider_scales` qui e' SWEEP: un PNG per scale, riusando il
    medesimo load di Flux. Comportamento identico al codice precedente.
  * Multi-mask multi-slider:
      --mask_names m_man.png m_woman.png
      --slider_paths smile.pt age.pt vangogh.pt
      --slider_to_mask 0 0 1
      --slider_scales 1.0 2.0 0.8
    qui `--slider_scales` ha 1 valore per slider (nessun sweep). Slider
    0 e 1 si compongono additivamente sulla regione di mask 0; slider 2
    sulla regione di mask 1. La regione fuori da tutte le maschere
    rimane baseline.

Inputs attesi nella run_dir (prodotti dai phase 1 e 2):
  - base.png        (phase 1)
  - metadata.json   (phase 1, per seed/prompt/steps/scheduler_config)
  - mask.png oppure mask_man.png/mask_woman.png/... (phase 2 SAM, binary)

Output:
  - edited.png oppure edited_scaleX.png (legacy sweep) oppure
    edited_compose.png (multi-mask)
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
        "Phase 3 - MaskedLoRA Flux multi-path blend "
        "(N maschere x M slider per maschera)."
    )
    parser.add_argument("--run_dir", type=Path, required=True)

    # Slider: 1 (legacy) o N (multi)
    parser.add_argument(
        "--slider_path", type=str, default=None,
        help="Legacy: 1 solo slider. Mutually exclusive con --slider_paths.",
    )
    parser.add_argument(
        "--slider_paths", type=str, nargs="+", default=None,
        help="Multi-mode: lista di slider (.pt o .safetensors).",
    )

    # Scale: 1 valore (legacy single) | sweep (legacy sweep) |
    #        N valori (multi, 1 per slider)
    parser.add_argument(
        "--slider_scale", type=float, default=3.0,
        help="Legacy single-slider: una scala. Default 3.0.",
    )
    parser.add_argument(
        "--slider_scales", type=float, nargs="+", default=None,
        help=(
            "Legacy single-slider: SWEEP, un PNG per scale. "
            "Multi-mode: 1 scale per slider (no sweep)."
        ),
    )

    # Mask: 1 (legacy) o N (multi)
    parser.add_argument(
        "--mask_name", type=str, default=None,
        help="Legacy: 1 sola mask (default mask.png). Mutually exclusive "
             "con --mask_names.",
    )
    parser.add_argument(
        "--mask_names", type=str, nargs="+", default=None,
        help="Multi-mode: lista di N maschere (PNG binari nella run_dir).",
    )

    # Mapping slider -> mask (solo multi)
    parser.add_argument(
        "--slider_to_mask", type=int, nargs="+", default=None,
        help="Multi-mode: slider_to_mask[i]=j -> slider i si applica nella "
             "regione di mask j. Lunghezza == --slider_paths. Piu' slider "
             "su stessa mask = composizione additiva (paper-style).",
    )

    parser.add_argument(
        "--edit_start_step",
        type=int,
        default=8,
        help=(
            "Step dal quale attivare il multi-path con LoRA. "
            "Se >0, i primi N step girano single-forward senza LoRA. "
            "Default 8: uniformato a shop_concept e baseline globale per "
            "confronti fair (22/30 step utili). 0 = LoRA attivo da subito."
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
    parser.add_argument("--output_name", type=str, default="edited.png",
                        help="Nome file output (legacy single o multi).")
    return parser


# =============================================================================
# Helpers
# =============================================================================

def load_run_inputs(
    run_dir: Path, mask_names: list
) -> Tuple[Dict, list]:
    """Carica metadata.json e una LISTA di maschere (PNG binari grayscale).

    Ritorna (metadata, [mask_tensor_1, mask_tensor_2, ...]) dove ogni
    mask_tensor ha shape (1, 1, H, W) in [0, 1].
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
def masked_multi_path_denoise(
    pipe: FluxPipeline,
    metadata: Dict,
    mask_pixels: list,
    mask_to_adapters: list,
    edit_start_step: int,
    device: torch.device,
) -> Image.Image:
    """Multi-path velocity-blend con N maschere disgiunte.

    Args:
        mask_pixels: list di N tensori shape (1, 1, H, W) in [0, 1].
        mask_to_adapters: list di N tuple (adapter_names: List[str],
            adapter_weights: List[float]) con gli slider attivi per
            ciascuna regione.
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

    # Avviso overlap (sum > 1 in qualche token = maschere sovrapposte)
    if num_masks > 1:
        sum_masks = torch.zeros_like(masks_packed[0])
        for mp in masks_packed:
            sum_masks = sum_masks + mp
        n_overlap = int((sum_masks > 1).sum().item())
        if n_overlap > 0:
            print(f"[phase3] WARNING: {n_overlap} token(s) hanno overlap "
                  f"di maschere (sum>1). La formula assume maschere "
                  f"disgiunte; nei token sovrapposti l'effetto e' la "
                  f"somma additiva delle delta -- puo' produrre "
                  f"saturazione. Verifica le maschere SAM.")

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

            # Forward N volte (uno per maschera) con set_adapters
            v_styled_list = []
            for mask_idx in range(num_masks):
                names, weights = mask_to_adapters[mask_idx]
                pipe.enable_lora()
                # set_adapters([n1,n2,...], weights=[w1,w2,...]):
                # PEFT mette tutti gli adapter in active e somma le delta
                # additivamente nel forward LoRA -- composizione paper-style.
                pipe.set_adapters(list(names),
                                  adapter_weights=[float(w) for w in weights])
                v_styled_list.append(_fwd(latents, timestep_in))

            # Blend velocity: regione esterna a tutte le maschere = base,
            # ogni regione mascherata = il suo v_styled (assumendo disjoint).
            v_pred = v_base.clone()
            sum_masks = torch.zeros_like(masks_packed[0])
            for mask_idx in range(num_masks):
                m = masks_packed[mask_idx]
                v_pred = v_pred - m * v_base + m * v_styled_list[mask_idx]
                sum_masks = sum_masks + m
            # Equivalente a: v_pred = (1 - sum_masks) * v_base
            #                       + sum_i (mask_i * v_styled_i)
            # quando le maschere sono disgiunte. Con overlap, la somma
            # additiva delle delta nei token sovrapposti corrisponde alla
            # composizione naturale (sopra abbiamo gia' avvertito).

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

    # Modalita': se l'utente usa una qualsiasi forma plural, va in multi.
    is_multi = has_slider_paths or has_mask_names or has_slider_to_mask

    if is_multi:
        if has_slider_path or has_mask_name:
            raise ValueError(
                "Mixing legacy (--slider_path/--mask_name) and multi "
                "(--slider_paths/--mask_names/--slider_to_mask) e' ambiguo. "
                "Usa SOLO una delle due forme."
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
                f"--slider_to_mask ha {len(slider_to_mask)} valori, attesi "
                f"{n_sliders} (uno per --slider_paths)."
            )
        for s_idx, m_idx in enumerate(slider_to_mask):
            if not (0 <= m_idx < n_masks):
                raise ValueError(
                    f"--slider_to_mask[{s_idx}]={m_idx} fuori range "
                    f"[0, {n_masks}). Hai {n_masks} maschere."
                )
        if args.slider_scales is None:
            raise ValueError(
                "Multi-mode richiede --slider_scales (1 valore per slider)."
            )
        if len(args.slider_scales) != n_sliders:
            raise ValueError(
                f"Multi-mode: --slider_scales ha {len(args.slider_scales)} "
                f"valori, attesi {n_sliders} (uno per slider)."
            )
        # Ogni mask deve avere almeno 1 slider mappato
        masks_used = set(slider_to_mask)
        for m in range(n_masks):
            if m not in masks_used:
                raise ValueError(
                    f"Maschera {m} ('{mask_names[m]}') non ha slider "
                    f"associati. Aggiungi almeno uno slider che mappi a "
                    f"questa mask o rimuovi la mask."
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
    # Convenzione nomi: default_0, default_1, ... allineata a shop_concept.
    # Lo stesso file slider puo' essere caricato N volte come adapter
    # distinti se serve applicarlo a maschere diverse con scale diverse.
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
