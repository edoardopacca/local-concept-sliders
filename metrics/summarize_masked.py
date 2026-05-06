#!/usr/bin/env python3
"""
summarize_masked.py — median localization metrics per concept for masked-LoRA edits.

Reads per-run CSV files produced by eval_masked.py and computes median
lpips_localization and clip_localization for each concept and scale.

Best scale per concept: the scale with the highest median clip_localization
(semantic edit most concentrated inside the mask).

Works for any results directory (SDXL or Flux).

Usage:
  python metrics/summarize_masked.py
  python metrics/summarize_masked.py --results_dir metrics/results_flux_masked
  python metrics/summarize_masked.py --json_out /tmp/masked_summary.json
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute median localization metrics per concept."
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("metrics/results_sdxl_masked"),
        help="Directory containing one sub-folder per concept with eval_results.csv",
    )
    p.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional: save best-scale summary to a JSON file",
    )
    return p


def get_medians(rows: list[dict], scale: int) -> dict | None:
    subset = [r for r in rows if int(r["scale"]) == scale]
    if not subset:
        return None
    lp_vals   = [float(r["lpips_localization"]) for r in subset]
    cl_vals   = [float(r["clip_localization"])   for r in subset]
    pct_above = sum(1 for v in cl_vals if v > 0.5) / len(cl_vals) * 100
    return {
        "lpips_loc": statistics.median(lp_vals),
        "clip_loc":  statistics.median(cl_vals),
        "pct_clip_above_0.5": pct_above,
        "n":         len(subset),
    }


def pick_best_scale(rows: list[dict], scales: list[int]) -> int:
    """Scale with highest median clip_localization."""
    best_sc, best_score = scales[0], -1.0
    for sc in scales:
        m = get_medians(rows, sc)
        if m is None:
            continue
        if m["clip_loc"] > best_score:
            best_score, best_sc = m["clip_loc"], sc
    return best_sc


def print_full_table(rows: list[dict], scales: list[int], concept: str) -> None:
    w = 64
    print(f"\n  {concept}  (all scales, n={len(set(r['run_id'] for r in rows))} runs)")
    print(f"  {'scale':<8} {'lpips_loc':>12}   {'clip_loc':>10}   {'clip>0.5':>10}   {'n':>3}")
    print(f"  {'-' * w}")
    for sc in scales:
        m = get_medians(rows, sc)
        if m is None:
            continue
        print(
            f"  {sc:<8} {m['lpips_loc']:>12.4f}   {m['clip_loc']:>10.4f}   "
            f"{m['pct_clip_above_0.5']:>9.1f}%   {m['n']:>3}"
        )


def main() -> dict:
    args = build_parser().parse_args()
    results_dir = args.results_dir

    concepts = sorted(
        p.name for p in results_dir.iterdir()
        if p.is_dir() and (p / "eval_results.csv").exists()
    )
    if not concepts:
        print(f"No concepts found in {results_dir}")
        return {}

    summary = {}

    # ── Full per-scale breakdown ──────────────────────────────────────────────
    print(f"\nFull breakdown  ·  {results_dir}")
    for concept in concepts:
        with open(results_dir / concept / "eval_results.csv") as f:
            rows = list(csv.DictReader(f))
        scales = sorted(set(int(r["scale"]) for r in rows))
        print_full_table(rows, scales, concept)
        best_sc = pick_best_scale(rows, scales)
        summary[concept] = {
            "best_scale": best_sc,
            "all_scales": {sc: get_medians(rows, sc) for sc in scales},
            "best": get_medians(rows, best_sc),
        }

    # ── Best-scale summary table ──────────────────────────────────────────────
    sep = "=" * 68
    print(f"\n{sep}")
    print("Best-scale summary  (scale = max clip_loc median)")
    print(sep)
    print(
        f"{'Concept':<16} {'Scale':<7} {'lpips_loc':>12}   {'clip_loc':>10}   "
        f"{'clip>0.5':>10}   n"
    )
    print("-" * 68)
    for concept, d in summary.items():
        b = d["best"]
        print(
            f"{concept:<16} {d['best_scale']:<7} {b['lpips_loc']:>12.4f}   "
            f"{b['clip_loc']:>10.4f}   {b['pct_clip_above_0.5']:>9.1f}%   {b['n']}"
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved JSON summary → {args.json_out}")

    return summary


if __name__ == "__main__":
    main()
