#!/usr/bin/env python3
"""
Evaluation script for masked LoRA editing — 6 concepts, 20 images each.

Computes per (run, scale):
  - LPIPS inside/outside raw and area-normalised; raw and normalised localization ratios
  - CLIP localization: 0 if ΔCLIP_in ≤ 0, else ΔCLIP_in / (ΔCLIP_in + |ΔCLIP_out|) ∈ [0,1]

Saves:
  - per-run:      runs/eval_{concept}_{id}/eval_metrics_s{scale}.json
  - per-concept:  metrics/results/{concept}/eval_results.csv
  - aggregate:    metrics/results/{concept}/eval_aggregate.json

Usage (GPU recommended):
  python metrics/eval_masked.py --concept age_person --runs_root sdxl/tasks/masked_lora/runs

Concepts: age_person  curlyhair  daynight  furlength  painterly  smile_person
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

CONCEPT_EDIT_PREFIX: Dict[str, str] = {
    "age_person":   "age",
    "curlyhair":    "curly",
    "daynight":     "daynight",
    "furlength":    "furlength",
    "painterly":    "painterly",
    "smile_person": "smile",
}

# Text prompt measuring the direction of the edit (used for CLIP localization)
CONCEPT_EDIT_PROMPT: Dict[str, str] = {
    "age_person":   "an elderly person with wrinkles",
    "curlyhair":    "a person with curly hair",
    "daynight":     "a night sky with stars",
    "furlength":    "an animal with long fluffy fur",
    "painterly":    "an oil painting with brushstrokes",
    "smile_person": "a person smiling with a big smile",
}


DEFAULT_RUNS_ROOT = "sdxl/tasks/masked_lora/runs"
DEFAULT_OUTPUT_DIR = "metrics/results"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def discover_scales(run_dirs: List[Path], prefix: str) -> List[float]:
    """Auto-detect available slider scales from existing edited images."""
    for run_dir in run_dirs:
        scales = sorted(
            float(p.stem.split("_s")[-1])
            for p in run_dir.glob(f"edited_{prefix}_s*.png")
        )
        if scales:
            return scales
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate masked LoRA editing across 20 runs per concept."
    )
    p.add_argument("--concept", required=True, choices=list(CONCEPT_EDIT_PREFIX))
    p.add_argument("--runs_root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    p.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--run_prefix", type=str, default=None,
                   help="Override run-dir prefix (default: eval_{concept}_)")
    p.add_argument("--edit_prefix", type=str, default=None,
                   help="Override edited image prefix (default: from CONCEPT_EDIT_PREFIX). "
                        "Use to distinguish edit variants on the same concept (e.g. age_t vs age_f).")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="Restrict evaluation to these specific scales (e.g. --scales 5 6 8). "
                        "Default: auto-discover from edited image filenames.")
    p.add_argument("--device", type=str, default="cuda")
    return p


# ---------------------------------------------------------------------------
# Metric helpers
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
    """Composite image over gray background using float mask."""
    m = mask_np[..., None]
    out = (img_np.astype(np.float32) * m + gray * (1.0 - m)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def compute_metrics(
    run_dir: Path,
    concept: str,
    scale: float,
    lpips_model,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    prefix: Optional[str] = None,
) -> Optional[Dict]:
    if prefix is None:
        prefix = CONCEPT_EDIT_PREFIX[concept]
    # Bash writes scale as-is (e.g. "3" not "3.0"), so try both formats.
    edited_path = run_dir / f"edited_{prefix}_s{scale:.1f}.png"
    if not edited_path.exists() and scale == int(scale):
        edited_path = run_dir / f"edited_{prefix}_s{int(scale)}.png"
    base_path   = run_dir / "base.png"
    mask_path   = run_dir / "mask_target.png"

    if not (edited_path.exists() and base_path.exists() and mask_path.exists()):
        return None

    base_pil   = Image.open(base_path).convert("RGB")
    edited_pil = Image.open(edited_path).convert("RGB")
    mask_pil   = Image.open(mask_path).convert("L").resize(base_pil.size, Image.NEAREST)

    base_np  = np.array(base_pil,   dtype=np.float32) / 255.0
    edit_np  = np.array(edited_pil, dtype=np.float32) / 255.0
    mask_np  = np.array(mask_pil,   dtype=np.float32) / 255.0
    imask_np = 1.0 - mask_np

    # ---- LPIPS -------------------------------------------------------
    # Multiply by mask before passing to LPIPS (same convention as 03_masked_edit.py).
    def _to_lp(arr: np.ndarray, m: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0  # [-1, 1]
        return (t * torch.from_numpy(m)).unsqueeze(0).to(device)

    mask_area  = float(mask_np.mean())
    imask_area = 1.0 - mask_area

    lpips_in_raw  = float(lpips_model(_to_lp(base_np, mask_np),  _to_lp(edit_np, mask_np)).item())
    lpips_out_raw = float(lpips_model(_to_lp(base_np, imask_np), _to_lp(edit_np, imask_np)).item())
    lpips_in_norm  = lpips_in_raw  / (mask_area  + 1e-8)
    lpips_out_norm = lpips_out_raw / (imask_area + 1e-8)
    lpips_loc_raw  = lpips_in_raw  / (lpips_out_raw  + 1e-8)
    lpips_loc_norm = lpips_in_norm / (lpips_out_norm + 1e-8)

    # ---- CLIP localization -------------------------------------------
    edit_prompt = CONCEPT_EDIT_PROMPT[concept]

    base_in_img  = apply_mask(np.array(base_pil),   mask_np)
    base_out_img = apply_mask(np.array(base_pil),   imask_np)
    edit_in_img  = apply_mask(np.array(edited_pil), mask_np)
    edit_out_img = apply_mask(np.array(edited_pil), imask_np)

    cb_in  = clip_sim(clip_model, clip_processor, base_in_img,  edit_prompt, device)
    cb_out = clip_sim(clip_model, clip_processor, base_out_img, edit_prompt, device)
    ce_in  = clip_sim(clip_model, clip_processor, edit_in_img,  edit_prompt, device)
    ce_out = clip_sim(clip_model, clip_processor, edit_out_img, edit_prompt, device)

    delta_in  = ce_in  - cb_in
    delta_out = ce_out - cb_out
    # delta_in <= 0 means the edit moved the masked region away from the target
    # concept — localization is undefined, we assign 0 (edit failure).
    # abs(delta_out) captures bidirectional outside changes without sign issues.
    if delta_in <= 0:
        clip_loc = 0.0
    else:
        clip_loc = delta_in / (delta_in + abs(delta_out) + 1e-8)

    return {
        "run_id":             run_dir.name,
        "scale":              scale,
        "lpips_inside":         lpips_in_raw,
        "lpips_outside":        lpips_out_raw,
        "lpips_loc_raw":        lpips_loc_raw,
        "lpips_inside_norm":    lpips_in_norm,
        "lpips_outside_norm":   lpips_out_norm,
        "lpips_localization":   lpips_loc_norm,
        "clip_delta_in":      delta_in,
        "clip_delta_out":     delta_out,
        "clip_localization":  clip_loc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "lpips_inside", "lpips_outside", "lpips_loc_raw",
    "lpips_inside_norm", "lpips_outside_norm", "lpips_localization",
    "clip_localization",
]


def main() -> None:
    args = build_parser().parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device = "cpu"

    edit_prefix = args.edit_prefix or CONCEPT_EDIT_PREFIX[args.concept]
    run_prefix = args.run_prefix or f"eval_{args.concept}_"
    runs = sorted(
        d for d in args.runs_root.iterdir()
        if d.is_dir() and d.name.startswith(run_prefix)
    )
    if not runs:
        print(f"[!] No run dirs found with prefix '{run_prefix}' in {args.runs_root}")
        return

    scales = sorted(args.scales) if args.scales is not None else discover_scales(runs, edit_prefix)
    if not scales:
        print(f"[!] No edited images found for concept '{args.concept}' "
              f"(prefix '{edit_prefix}') — run phase3 first.")
        return
    scale_to_idx = {s: i + 1 for i, s in enumerate(scales)}

    print(f"[eval_masked]  concept={args.concept}  edit_prefix={edit_prefix}  "
          f"runs={len(runs)}  scales={scales} → labels={list(scale_to_idx.values())}  device={device}")

    # Load models once
    print("  loading LPIPS (alex)...")
    lp_model = lpips_lib.LPIPS(net="alex").to(device)
    print(f"  loading CLIP ({CLIP_MODEL_ID})...")
    cl_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True).to(device)
    cl_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    all_rows: List[Dict] = []

    for run_dir in runs:
        for scale in scales:
            row = compute_metrics(run_dir, args.concept, scale, lp_model, cl_model, cl_proc, device,
                                  prefix=edit_prefix)
            if row is None:
                print(f"  [skip] {run_dir.name}  s={scale:.1f}  (missing files)")
                continue

            idx = scale_to_idx[scale]
            row["scale"] = idx  # replace float with integer index

            # save per-run
            out_path = run_dir / f"eval_metrics_s{idx}.json"
            out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

            all_rows.append(row)
            print(f"  [ok]   {run_dir.name}  s{idx}  "
                  f"lpips_loc_raw={row['lpips_loc_raw']:.3f}  "
                  f"lpips_loc_norm={row['lpips_localization']:.3f}  "
                  f"clip_loc={row['clip_localization']:.4f}")

    if not all_rows:
        print("[!] No results computed — check that edited images exist.")
        return

    # ---- save aggregate CSV ----------------------------------------
    out_dir = args.output_dir / args.concept
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[OK] {len(all_rows)} rows  →  {csv_path}")

    # ---- print + save aggregate stats per scale --------------------
    print(f"\n{'='*62}")
    print(f"  AGGREGATE   concept={args.concept}")
    print(f"{'='*62}")
    agg: Dict = {}
    for idx in sorted(scale_to_idx.values()):
        subset = [r for r in all_rows if r["scale"] == idx]
        if not subset:
            continue
        print(f"\n  scale={idx}  n={len(subset)}")
        agg[f"scale_{idx}"] = {}
        for k in METRIC_KEYS:
            vals = [r[k] for r in subset]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            agg[f"scale_{idx}"][k] = {"mean": mean, "std": std}
            print(f"    {k:<28s}  mean={mean:.4f}  std={std:.4f}")

    agg_path = out_dir / "eval_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[OK] Aggregate  →  {agg_path}")


if __name__ == "__main__":
    main()
