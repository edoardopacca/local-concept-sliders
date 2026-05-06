#!/usr/bin/env python3
"""
eval_selectivity_loc.py — localization metrics for selectivity runs.

Re-evaluates the same 80 runs used in the selectivity experiment, but using
the localization framing (same as eval_masked.py) instead of target vs
non-target:

  inside  = mask_target region
  outside = everything OUTSIDE mask_target (complement)

Metrics per run × scale × slider_type:
  - lpips_inside / lpips_outside (raw and area-normalised)
  - lpips_localization = lpips_inside_norm / lpips_outside_norm  (> 1 = good)
  - clip_delta_in / clip_delta_out
  - clip_localization = delta_in / (delta_in + |delta_out|)  if delta_in > 0 else 0

Output: metrics/results_sdxl_selectivity_loc/{concept}/eval_results.csv + eval_aggregate.json

Usage:
  python metrics/eval_selectivity_loc.py --concept age --device cuda
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Per-concept config  (same as eval_selectivity.py)
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
DEFAULT_OUTPUT_DIR = "metrics/results_sdxl_selectivity_loc"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Localization metrics for selectivity runs (inside mask_target vs outside)."
    )
    p.add_argument("--concept", required=True, choices=list(CONCEPT_EDIT_PROMPT))
    p.add_argument("--runs_root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    p.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--run_prefix", type=str, default=None)
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


def compute_loc_metrics(
    base_pil: Image.Image,
    edited_pil: Image.Image,
    mask_inside_np: np.ndarray,   # mask_target (binary float 0/1)
    edit_prompt: str,
    lpips_model,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
) -> Dict:
    mask_outside_np = 1.0 - mask_inside_np  # complement

    base_np = np.array(base_pil,   dtype=np.float32) / 255.0
    edit_np = np.array(edited_pil, dtype=np.float32) / 255.0

    def _to_lp(arr: np.ndarray, m: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        return (t * torch.from_numpy(m)).unsqueeze(0).to(device)

    inside_area  = float(mask_inside_np.mean())
    outside_area = float(mask_outside_np.mean())

    # LPIPS
    lp_in_raw  = float(lpips_model(
        _to_lp(base_np, mask_inside_np),  _to_lp(edit_np, mask_inside_np)).item())
    lp_out_raw = float(lpips_model(
        _to_lp(base_np, mask_outside_np), _to_lp(edit_np, mask_outside_np)).item())
    lp_in_norm  = lp_in_raw  / (inside_area  + 1e-8)
    lp_out_norm = lp_out_raw / (outside_area + 1e-8)
    lpips_loc   = lp_in_norm / (lp_out_norm  + 1e-8)

    # CLIP localization
    base_in_img  = apply_mask(np.array(base_pil),   mask_inside_np)
    base_out_img = apply_mask(np.array(base_pil),   mask_outside_np)
    edit_in_img  = apply_mask(np.array(edited_pil), mask_inside_np)
    edit_out_img = apply_mask(np.array(edited_pil), mask_outside_np)

    cb_in  = clip_sim(clip_model, clip_processor, base_in_img,  edit_prompt, device)
    cb_out = clip_sim(clip_model, clip_processor, base_out_img, edit_prompt, device)
    ce_in  = clip_sim(clip_model, clip_processor, edit_in_img,  edit_prompt, device)
    ce_out = clip_sim(clip_model, clip_processor, edit_out_img, edit_prompt, device)

    delta_in  = ce_in  - cb_in
    delta_out = ce_out - cb_out

    if delta_in <= 0:
        clip_loc = 0.0
    else:
        clip_loc = delta_in / (delta_in + abs(delta_out) + 1e-8)

    return {
        "lpips_inside":       lp_in_raw,
        "lpips_outside":      lp_out_raw,
        "lpips_loc_raw":      lp_in_raw / (lp_out_raw + 1e-8),
        "lpips_inside_norm":  lp_in_norm,
        "lpips_outside_norm": lp_out_norm,
        "lpips_localization": lpips_loc,
        "clip_delta_in":      delta_in,
        "clip_delta_out":     delta_out,
        "clip_localization":  clip_loc,
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
    "lpips_inside", "lpips_outside", "lpips_loc_raw",
    "lpips_inside_norm", "lpips_outside_norm", "lpips_localization",
    "clip_delta_in", "clip_delta_out", "clip_localization",
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

    print(f"[eval_selectivity_loc]  concept={args.concept}  runs={len(runs)}  "
          f"scales={scales}  device={device}")

    print("  loading LPIPS (alex)...")
    lp_model = lpips_lib.LPIPS(net="alex").to(device)
    print(f"  loading CLIP ({CLIP_MODEL_ID})...")
    cl_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    cl_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    edit_prompt = CONCEPT_EDIT_PROMPT[args.concept]
    all_rows: List[Dict] = []

    for run_dir in runs:
        base_path   = run_dir / "base.png"
        mask_t_path = run_dir / "mask_target.png"

        if not (base_path.exists() and mask_t_path.exists()):
            print(f"  [skip] {run_dir.name}  (missing base.png or mask_target.png)")
            continue

        base_pil       = Image.open(base_path).convert("RGB")
        mask_inside_np = (
            np.array(
                Image.open(mask_t_path).convert("L").resize(base_pil.size, Image.NEAREST),
                dtype=np.float32
            ) / 255.0
        )

        for scale in scales:
            for slider_type, prefix in [("specific", spec_prefix), ("general", gen_prefix)]:
                edited_path = run_dir / f"edited_{prefix}_s{scale:.1f}.png"
                if not edited_path.exists():
                    print(f"  [skip] {run_dir.name}  s={scale:.1f}  {slider_type}  (missing)")
                    continue

                edited_pil = Image.open(edited_path).convert("RGB")

                metrics = compute_loc_metrics(
                    base_pil, edited_pil, mask_inside_np,
                    edit_prompt, lp_model, cl_model, cl_proc, device,
                )

                row = {
                    "run_id":      run_dir.name,
                    "scale":       scale,
                    "slider_type": slider_type,
                    **metrics,
                }
                all_rows.append(row)

                print(f"  [ok]  {run_dir.name}  s{scale:.1f}  {slider_type:8s}  "
                      f"lpips_loc={metrics['lpips_localization']:.3f}  "
                      f"clip_loc={metrics['clip_localization']:.4f}")

    if not all_rows:
        print("[!] No results computed.")
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
