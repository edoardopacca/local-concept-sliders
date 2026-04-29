#!/usr/bin/env python3
"""
Evaluation script for masked LoRA editing — 6 concepts, 20 images each.

Computes per (run, scale):
  - LPIPS_in / LPIPS_out        masked perceptual distance inside/outside mask
  - CLIP localization score      ΔCLIP_in / (ΔCLIP_in + |ΔCLIP_out|)  →  1 = perfect localization
  - Attribute score              CLIP zero-shot on edited masked crop (positive vs negative text)

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

# (positive_text, negative_text) for zero-shot attribute score
CONCEPT_ATTR_TEXTS: Dict[str, tuple] = {
    "age_person":   ("an elderly person with wrinkles and gray hair",
                     "a young person with smooth skin"),
    "curlyhair":    ("a person with curly wavy hair",
                     "a person with straight smooth hair"),
    "daynight":     ("a night sky with stars and moonlight",
                     "a bright daytime sky with sunlight"),
    "furlength":    ("an animal with long fluffy fur coat",
                     "an animal with short smooth fur"),
    "painterly":    ("an oil painting with artistic brushstrokes",
                     "a sharp photorealistic photograph"),
    "smile_person": ("a person smiling with a big happy smile",
                     "a person with a neutral expression"),
}

DEFAULT_SCALES: List[float] = [0.5, 1.0, 1.5]
DEFAULT_RUNS_ROOT = "sdxl/tasks/masked_lora/runs"
DEFAULT_OUTPUT_DIR = "metrics/results"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


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
    p.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES,
                   metavar="S", help="Slider scales to evaluate (default: 1.5 2.0 3.0)")
    p.add_argument("--run_prefix", type=str, default=None,
                   help="Override run-dir prefix (default: eval_{concept}_)")
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
) -> Optional[Dict]:
    prefix = CONCEPT_EDIT_PREFIX[concept]
    edited_path = run_dir / f"edited_{prefix}_s{scale:.1f}.png"
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

    lpips_in  = float(lpips_model(_to_lp(base_np, mask_np),  _to_lp(edit_np, mask_np)).item())
    lpips_out = float(lpips_model(_to_lp(base_np, imask_np), _to_lp(edit_np, imask_np)).item())
    lpips_loc = lpips_in / (lpips_out + 1e-8)

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
    clip_loc  = delta_in / (delta_in + abs(delta_out) + 1e-8)

    # ---- Attribute score (zero-shot) ---------------------------------
    pos_text, neg_text = CONCEPT_ATTR_TEXTS[concept]
    sim_pos = clip_sim(clip_model, clip_processor, edit_in_img, pos_text, device)
    sim_neg = clip_sim(clip_model, clip_processor, edit_in_img, neg_text, device)
    # softmax over {pos, neg}
    exp_pos = float(np.exp(sim_pos))
    exp_neg = float(np.exp(sim_neg))
    attr_score = exp_pos / (exp_pos + exp_neg)

    return {
        "run_id":            run_dir.name,
        "scale":             scale,
        "lpips_inside":      lpips_in,
        "lpips_outside":     lpips_out,
        "lpips_localization": lpips_loc,
        "clip_delta_in":     delta_in,
        "clip_delta_out":    delta_out,
        "clip_localization": clip_loc,
        "attr_score":        attr_score,
        "attr_sim_positive": sim_pos,
        "attr_sim_negative": sim_neg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "lpips_inside", "lpips_outside", "lpips_localization",
    "clip_localization", "attr_score",
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

    print(f"[eval_masked]  concept={args.concept}  runs={len(runs)}  "
          f"scales={args.scales}  device={device}")

    # Load models once
    print("  loading LPIPS (alex)...")
    lp_model = lpips_lib.LPIPS(net="alex").to(device)
    print(f"  loading CLIP ({CLIP_MODEL_ID})...")
    cl_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    cl_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    all_rows: List[Dict] = []

    for run_dir in runs:
        for scale in args.scales:
            row = compute_metrics(run_dir, args.concept, scale, lp_model, cl_model, cl_proc, device)
            if row is None:
                print(f"  [skip] {run_dir.name}  s={scale:.1f}  (missing files)")
                continue

            # save per-run
            out_path = run_dir / f"eval_metrics_s{scale:.1f}.json"
            out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

            all_rows.append(row)
            print(f"  [ok]   {run_dir.name}  s={scale:.1f}  "
                  f"lpips_in={row['lpips_inside']:.4f}  "
                  f"lpips_out={row['lpips_outside']:.4f}  "
                  f"clip_loc={row['clip_localization']:.4f}  "
                  f"attr={row['attr_score']:.4f}")

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
    for scale in args.scales:
        subset = [r for r in all_rows if r["scale"] == scale]
        if not subset:
            continue
        print(f"\n  scale={scale:.1f}  n={len(subset)}")
        agg[f"scale_{scale:.1f}"] = {}
        for k in METRIC_KEYS:
            vals = [r[k] for r in subset]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            agg[f"scale_{scale:.1f}"][k] = {"mean": mean, "std": std}
            print(f"    {k:<28s}  mean={mean:.4f}  std={std:.4f}")

    agg_path = out_dir / "eval_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[OK] Aggregate  →  {agg_path}")


if __name__ == "__main__":
    main()
