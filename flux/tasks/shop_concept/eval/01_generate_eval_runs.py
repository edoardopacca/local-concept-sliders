#!/usr/bin/env python3
"""
01_generate_eval_runs.py
========================
Genera 20 run di valutazione LoRAShop per un singolo concept.

Per ogni run salva in {runs_root}/eval_{concept}_{seed:04d}/:
  base.png                   generazione Flux pura (lora_scale=0)
  mask_target.png            maschera attention del target (da prior LoRAShop)
  edited_lorashop_s1.0.png   edit con scale=1.0
  edited_lorashop_s2.0.png   edit con scale=2.0
  edited_lorashop_s3.0.png   edit con scale=3.0
  metadata.json              seed, prompt, concept, slider_path, height, width

La maschera e' estratta dalla prior phase di LoRAShop (block 19 del
transformer Flux) durante la call con scale=0. I forward successivi
con scale>0 usano la stessa maschera (il prior e' ricalcolato ad ogni
call, ma il risultato e' stabile per lo stesso seed/prompt).

Uso:
  python flux/tasks/shop_concept/eval/01_generate_eval_runs.py \\
      --concept smile_woman \\
      --slider_path flux/trained_sliders/sliders/trial_v1/smile_woman_flux_v1/flux-smile_woman_flux_v1/slider_0.pt \\
      --runs_root flux/tasks/shop_concept/runs \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import yaml

# Aggiungi repo root a sys.path per import assoluti.
# __file__ = .../flux/tasks/shop_concept/eval/01_generate_eval_runs.py
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

import flux.tasks.shop_concept  # noqa: F401  side-effect: installa SDPA shim

import torch
import numpy as np
from PIL import Image

from flux.tasks.shop_concept.lib.flux_real_pipeline import RealGenerationPipeline
from flux.tasks.shop_concept.scripts.generate import (
    prepare_slider_as_safetensors,
    ensure_matching_lora_params,
)
from safetensors.torch import load_file


EVAL_SCALES = [1.0, 2.0, 3.0]
PROMPTS_YAML = Path(__file__).parent / "prompts.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera eval runs LoRAShop per un singolo concept (20 immagini)."
    )
    p.add_argument("--concept", required=True,
                   choices=["smile_woman", "curlyhair_man", "age_woman", "furlength_dog"],
                   help="Nome del concept da valutare.")
    p.add_argument("--slider_path", required=True,
                   help="Path al file .pt o .safetensors del Concept Slider addestrato.")
    p.add_argument("--runs_root", type=Path,
                   default=Path("flux/tasks/shop_concept/runs"),
                   help="Directory root dove salvare le run di eval.")
    p.add_argument("--model_id", type=str,
                   default="black-forest-labs/FLUX.1-dev",
                   help="HF model ID o path locale di Flux.1-dev.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--edit_start_step", type=int, default=8,
                   help="Step a partire dal quale inizia il blending mask-guidato.")
    p.add_argument("--peft_cache_dir", type=str,
                   default="flux/tasks/shop_concept/_peft_cache",
                   help="Cache per conversioni .pt -> safetensors PEFT.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", choices=["all", "base_only", "edited_only"],
                   default="all",
                   help="all: base+mask+edited; base_only: solo base+mask; "
                        "edited_only: solo edited (assume base/mask esistano).")
    return p


def load_concept_prompts(concept: str):
    with open(PROMPTS_YAML, "r") as f:
        data = yaml.safe_load(f)
    assert concept in data, f"Concept '{concept}' non trovato in prompts.yaml"
    prompts = data[concept]["prompts"]
    for p in prompts:
        assert "target_prompt" in p, (
            f"Entry seed={p.get('seed')} in concept '{concept}' manca di target_prompt"
        )
        assert p["target_prompt"] in p["text"], (
            f"seed={p['seed']}: target_prompt '{p['target_prompt']}' "
            f"non è sottostringa di '{p['text']}'"
        )
    return prompts


def mask_seg_to_png_resized(mask_seg_path: Path, out_path: Path, target_size: tuple):
    """Legge la seg mask (bassa risoluzione) e la ridimensiona alla target_size."""
    seg = Image.open(mask_seg_path).convert("L")
    seg_resized = seg.resize(target_size, Image.NEAREST)
    seg_resized.save(out_path)


def run_generation(
    pipe: RealGenerationPipeline,
    prompt: str,
    target_prompt: str,
    seed: int,
    lora_scale: float,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    edit_start_step: int,
    mask_dump_path: str | None,
    device: str,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        target_prompt=[target_prompt],
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=max_sequence_length,
        height=height,
        width=width,
        generator=generator,
        edit_start_step=edit_start_step,
        target_lora_scales=[lora_scale],
        mask_dump_path=mask_dump_path,
    )
    return result.images[0]


def main() -> None:
    args = build_parser().parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA non disponibile, uso CPU")
        device = "cpu"

    prompts = load_concept_prompts(args.concept)
    print(f"[eval] concept={args.concept}  n_prompts={len(prompts)}")

    # --- Carica RealGenerationPipeline ---
    print(f"[eval] loading Flux pipeline from {args.model_id} ...")
    pipe = RealGenerationPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # --- Prepara slider come PEFT safetensors e carica ---
    print(f"[eval] preparando slider {args.slider_path} ...")
    slider_st = prepare_slider_as_safetensors(args.slider_path, args.peft_cache_dir)
    lora_dict = load_file(slider_st)
    lora_dicts = ensure_matching_lora_params([lora_dict], rank=16)
    pipe.load_lora_weights(lora_dicts[0], adapter_name="default_0")

    # --- Registra transformer blocks LoRAShop-aware ---
    print("[eval] registering transformer blocks ...")
    pipe.register_transformer_blocks()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for entry in prompts:
            seed        = entry["seed"]
            prompt_text = entry["text"]
            target_prompt = entry["target_prompt"]
            run_name = f"eval_{args.concept}_{seed:04d}"
            run_dir = args.runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            base_path = run_dir / "base.png"
            mask_path = run_dir / "mask_target.png"

            base_done   = base_path.exists() and mask_path.exists()
            edited_done = all(
                (run_dir / f"edited_lorashop_s{s:.1f}.png").exists()
                for s in EVAL_SCALES
            )

            # Skip logic per mode
            if args.mode == "all" and base_done and edited_done:
                print(f"  [skip] {run_name} (già completato)")
                continue
            if args.mode == "base_only" and base_done:
                print(f"  [skip] {run_name} (base già completato)")
                continue
            if args.mode == "edited_only":
                if not base_done:
                    print(f"  [warn] {run_name} — base/mask mancante, salta")
                    continue
                if edited_done:
                    print(f"  [skip] {run_name} (edited già completato)")
                    continue

            print(f"\n  [{run_name}] seed={seed}  target='{target_prompt}'  "
                  f"prompt='{prompt_text[:70]}...'")

            # --- Step 1: genera base (scale=0) + estrai maschera ---
            if args.mode != "edited_only" and not base_done:
                mask_dump_prefix = str(Path(tmp_dir) / run_name)
                print(f"    → base + mask ...")
                base_img = run_generation(
                    pipe=pipe,
                    prompt=prompt_text,
                    target_prompt=target_prompt,
                    seed=seed,
                    lora_scale=0.0,
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    edit_start_step=args.edit_start_step,
                    mask_dump_path=mask_dump_prefix,
                    device=device,
                )
                base_img.save(base_path)
                print(f"    ✓ base.png")

                # Ridimensiona la seg mask alla dimensione dell'immagine
                seg_mask_low_res = Path(f"{mask_dump_prefix}_target0_seg.png")
                if seg_mask_low_res.exists():
                    mask_seg_to_png_resized(
                        seg_mask_low_res, mask_path, (args.width, args.height)
                    )
                    print(f"    ✓ mask_target.png")
                else:
                    print(f"    [warn] mask seg non trovata: {seg_mask_low_res}")

            # In base_only mode scrivi metadata e passa al prossimo
            if args.mode == "base_only":
                meta = {
                    "run_id":          run_name,
                    "concept":         args.concept,
                    "target_prompt":   target_prompt,
                    "seed":            seed,
                    "prompt":          prompt_text,
                    "slider_path":     str(args.slider_path),
                    "height":          args.height,
                    "width":           args.width,
                    "steps":           args.steps,
                    "guidance_scale":  args.guidance_scale,
                    "edit_start_step": args.edit_start_step,
                    "eval_scales":     EVAL_SCALES,
                }
                (run_dir / "metadata.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                print(f"    ✓ metadata.json")
                continue

            # --- Step 2: genera immagini editate per ogni scale ---
            for scale in EVAL_SCALES:
                out_path = run_dir / f"edited_lorashop_s{scale:.1f}.png"
                if out_path.exists():
                    continue
                print(f"    → edited scale={scale:.1f} ...")
                edited_img = run_generation(
                    pipe=pipe,
                    prompt=prompt_text,
                    target_prompt=target_prompt,
                    seed=seed,
                    lora_scale=scale,
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    edit_start_step=args.edit_start_step,
                    mask_dump_path=None,
                    device=device,
                )
                edited_img.save(out_path)
                print(f"    ✓ {out_path.name}")

            # --- Metadata ---
            meta = {
                "run_id":          run_name,
                "concept":         args.concept,
                "target_prompt":   target_prompt,
                "seed":            seed,
                "prompt":          prompt_text,
                "slider_path":     str(args.slider_path),
                "height":          args.height,
                "width":           args.width,
                "steps":           args.steps,
                "guidance_scale":  args.guidance_scale,
                "edit_start_step": args.edit_start_step,
                "eval_scales":     EVAL_SCALES,
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"    ✓ metadata.json")

    print(f"\n[eval] DONE  →  {args.runs_root}")


if __name__ == "__main__":
    main()
