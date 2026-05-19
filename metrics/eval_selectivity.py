#!/usr/bin/env python3
"""
eval_selectivity.py — valuta la selettività di slider subject-specific vs general.

Per ogni concept (age, curlyhair, furlength, smile) e per ogni immagine base:
  - Lo slider è applicato GLOBALMENTE (niente maschera spaziale in generazione)
  - Misuriamo separatamente cosa succede nella regione TARGET e NON-TARGET
  - Confrontiamo slider SPECIFICO (e.g. age_woman) vs GENERALE (e.g. age_person)

Metriche per run × scale × slider_type:
  - lpips_target_raw / lpips_nontarget_raw:     LPIPS grezzo nelle due regioni
  - lpips_target_norm / lpips_nontarget_norm:   normalizzato per area (comparabile)
  - lpips_selectivity:   lpips_target_norm / lpips_nontarget_norm  (> 1 = selettivo)
  - clip_delta_target / clip_delta_nontarget:   ΔCLIP nelle due regioni
  - clip_selectivity:    δtarget / (δtarget + |δnontarget|) se δtarget > 0 else 0

Usage:
  python metrics/eval_selectivity.py --concept age --device cuda

Concepts: age  curlyhair  furlength  smile
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Per-concept config
# ---------------------------------------------------------------------------

CONCEPT_EDIT_PROMPT: Dict[str, str] = {
    "age":       "an elderly person with wrinkles",
    "curlyhair": "a person with curly hair",
    "furlength": "an animal with long fluffy fur",
    "smile":     "a person smiling with a big smile",
}

CONCEPT_SPECIFIC_PREFIX: Dict[str, str] = {
    "age":       "age_specific",
    "curlyhair": "curlyhair_specific",
    "furlength": "furlength_specific",
    "smile":     "smile_specific",
}

CONCEPT_GENERAL_PREFIX: Dict[str, str] = {
    "age":       "age_general",
    "curlyhair": "curlyhair_general",
    "furlength": "furlength_general",
    "smile":     "smile_general",
}

DEFAULT_RUNS_ROOT = "sdxl/tasks/selectivity/runs"
DEFAULT_OUTPUT_DIR = "metrics/results_sdxl_selectivity"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate selectivity of subject-specific vs general sliders."
    )
    p.add_argument("--concept", required=True, choices=list(CONCEPT_EDIT_PROMPT))
    p.add_argument("--runs_root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    p.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--run_prefix", type=str, default=None,
                   help="Override run-dir prefix (default: eval_{concept}_)")
    p.add_argument("--device", type=str, default="cuda")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_sim(model: CLIPModel, processor: CLIPProcessor,
             image: Image.Image, text: str, device: str) -> float:
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_emb * txt_emb).sum())


def apply_mask(img_np: np.ndarray, mask_np: np.ndarray, gray: int = 127) -> Image.Image:
    m = mask_np[..., None]
    out = (img_np.astype(np.float32) * m + gray * (1.0 - m)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def compute_selectivity_metrics(
    base_pil: Image.Image,
    edited_pil: Image.Image,
    mask_target_np: np.ndarray,
    mask_nontarget_np: np.ndarray,
    edit_prompt: str,
    lpips_model,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
) -> Dict:
    base_np = np.array(base_pil,   dtype=np.float32) / 255.0
    edit_np = np.array(edited_pil, dtype=np.float32) / 255.0

    def _to_lp(arr: np.ndarray, m: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        return (t * torch.from_numpy(m)).unsqueeze(0).to(device)

    target_area    = float(mask_target_np.mean())
    nontarget_area = float(mask_nontarget_np.mean())

    # LPIPS in each region
    lpips_t_raw  = float(lpips_model(
        _to_lp(base_np, mask_target_np), _to_lp(edit_np, mask_target_np)).item())
    lpips_nt_raw = float(lpips_model(
        _to_lp(base_np, mask_nontarget_np), _to_lp(edit_np, mask_nontarget_np)).item())
    lpips_t_norm  = lpips_t_raw  / (target_area    + 1e-8)
    lpips_nt_norm = lpips_nt_raw / (nontarget_area + 1e-8)
    # selectivity ratio: higher = more selective (target changed much more than nontarget).
    # Parallel to LPIPS-loc (inside/outside) and consistent with paper tables (\uparrow).
    lpips_sel = lpips_t_norm / (lpips_nt_norm + 1e-8)

    # CLIP delta in each region
    base_t_img  = apply_mask(np.array(base_pil),   mask_target_np)
    base_nt_img = apply_mask(np.array(base_pil),   mask_nontarget_np)
    edit_t_img  = apply_mask(np.array(edited_pil), mask_target_np)
    edit_nt_img = apply_mask(np.array(edited_pil), mask_nontarget_np)

    cb_t  = clip_sim(clip_model, clip_processor, base_t_img,  edit_prompt, device)
    cb_nt = clip_sim(clip_model, clip_processor, base_nt_img, edit_prompt, device)
    ce_t  = clip_sim(clip_model, clip_processor, edit_t_img,  edit_prompt, device)
    ce_nt = clip_sim(clip_model, clip_processor, edit_nt_img, edit_prompt, device)

    delta_t  = ce_t  - cb_t
    delta_nt = ce_nt - cb_nt

    if delta_t <= 0:
        clip_sel = 0.0
    else:
        clip_sel = delta_t / (delta_t + abs(delta_nt) + 1e-8)

    return {
        "lpips_target_raw":    lpips_t_raw,
        "lpips_nontarget_raw": lpips_nt_raw,
        "lpips_target_norm":   lpips_t_norm,
        "lpips_nontarget_norm":lpips_nt_norm,
        "lpips_selectivity":   lpips_sel,
        "clip_delta_target":   delta_t,
        "clip_delta_nontarget":delta_nt,
        "clip_selectivity":    clip_sel,
    }


def discover_scales(run_dirs: List[Path], prefix: str) -> List[float]:
    for run_dir in run_dirs:
        scales = sorted(
            float(p.stem.split("_s")[-1])
            for p in run_dir.glob(f"edited_{prefix}_s*.png")
        )
        if scales:
            return scales
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "lpips_target_raw", "lpips_nontarget_raw",
    "lpips_target_norm", "lpips_nontarget_norm",
    "lpips_selectivity",
    "clip_delta_target", "clip_delta_nontarget",
    "clip_selectivity",
]


def main() -> None:
    args = build_parser().parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device = "cpu"

    run_prefix = args.run_prefix or f"eval_{args.concept}_"
    runs = sorted(
        d for d in args.runs_root.iterdir()
        if d.is_dir() and d.name.startswith(run_prefix)
    )
    if not runs:
        print(f"[!] No run dirs found with prefix '{run_prefix}' in {args.runs_root}")
        return

    spec_prefix = CONCEPT_SPECIFIC_PREFIX[args.concept]
    gen_prefix  = CONCEPT_GENERAL_PREFIX[args.concept]
    scales_spec = discover_scales(runs, spec_prefix)
    scales_gen  = discover_scales(runs, gen_prefix)
    scales = sorted(set(scales_spec) | set(scales_gen))
    if not scales:
        print(f"[!] No edited images found for concept '{args.concept}' — run phase3 first.")
        return

    print(f"[eval_selectivity]  concept={args.concept}  runs={len(runs)}  "
          f"scales={scales}  device={device}")

    print("  loading LPIPS (alex)...")
    lp_model = lpips_lib.LPIPS(net="alex").to(device)
    print(f"  loading CLIP ({CLIP_MODEL_ID})...")
    cl_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    cl_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    edit_prompt = CONCEPT_EDIT_PROMPT[args.concept]
    all_rows: List[Dict] = []

    for run_dir in runs:
        base_path    = run_dir / "base.png"
        mask_t_path  = run_dir / "mask_target.png"
        mask_nt_path = run_dir / "mask_nontarget.png"

        if not (base_path.exists() and mask_t_path.exists() and mask_nt_path.exists()):
            print(f"  [skip] {run_dir.name}  (missing base/mask files)")
            continue

        base_pil   = Image.open(base_path).convert("RGB")
        mask_t_pil = Image.open(mask_t_path).convert("L").resize(base_pil.size, Image.NEAREST)
        mask_nt_pil= Image.open(mask_nt_path).convert("L").resize(base_pil.size, Image.NEAREST)

        mask_t_np  = np.array(mask_t_pil,  dtype=np.float32) / 255.0
        mask_nt_np = np.array(mask_nt_pil, dtype=np.float32) / 255.0

        for scale in scales:
            for slider_type, prefix in [("specific", spec_prefix), ("general", gen_prefix)]:
                edited_path = run_dir / f"edited_{prefix}_s{scale:.1f}.png"
                if not edited_path.exists():
                    print(f"  [skip] {run_dir.name}  s={scale:.1f}  {slider_type}  (missing)")
                    continue

                edited_pil = Image.open(edited_path).convert("RGB")

                metrics = compute_selectivity_metrics(
                    base_pil, edited_pil, mask_t_np, mask_nt_np,
                    edit_prompt, lp_model, cl_model, cl_proc, device,
                )

                row = {
                    "run_id":      run_dir.name,
                    "scale":       scale,
                    "slider_type": slider_type,
                    **metrics,
                }
                all_rows.append(row)

                idx = scales.index(scale) + 1
                json_out = run_dir / f"eval_selectivity_{slider_type}_s{idx}.json"
                json_out.write_text(json.dumps(row, indent=2), encoding="utf-8")

                print(f"  [ok]  {run_dir.name}  s{scale:.1f}  {slider_type:8s}  "
                      f"lpips_sel={metrics['lpips_selectivity']:.3f}  "
                      f"clip_sel={metrics['clip_selectivity']:.4f}")

    if not all_rows:
        print("[!] No results computed — check that edited images and masks exist.")
        return

    out_dir = args.output_dir / args.concept
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[OK] {len(all_rows)} rows  →  {csv_path}")

    # Aggregate: per slider_type × scale
    print(f"\n{'='*70}")
    print(f"  AGGREGATE   concept={args.concept}")
    print(f"{'='*70}")
    agg: Dict = {}
    for slider_type in ("specific", "general"):
        agg[slider_type] = {}
        for scale in scales:
            subset = [r for r in all_rows
                      if r["slider_type"] == slider_type and r["scale"] == scale]
            if not subset:
                continue
            print(f"\n  {slider_type:8s}  scale={scale:.1f}  n={len(subset)}")
            agg[slider_type][f"scale_{scale:.1f}"] = {}
            for k in METRIC_KEYS:
                vals = [r[k] for r in subset]
                mean, std = float(np.mean(vals)), float(np.std(vals))
                agg[slider_type][f"scale_{scale:.1f}"][k] = {"mean": mean, "std": std}
                print(f"    {k:<28s}  mean={mean:.4f}  std={std:.4f}")

    agg_path = out_dir / "eval_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[OK] Aggregate  →  {agg_path}")


if __name__ == "__main__":
    main()
