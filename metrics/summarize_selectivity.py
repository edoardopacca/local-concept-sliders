#!/usr/bin/env python3
"""
summarize_selectivity.py — median selectivity metrics per concept.

Reads per-run CSV files produced by eval_selectivity.py and computes median
lpips_selectivity and clip_selectivity for each concept, scale, and slider type
(specific / general).

Best scale per concept: the scale that maximises clip_sel_median / lpips_sel_median
for the specific slider (combined selectivity score). The same scale is then used
to report both lpips_sel and clip_sel for specific and general.

Works for any results directory (SDXL or Flux).

Usage:
  python metrics/summarize_selectivity.py
  python metrics/summarize_selectivity.py --results_dir metrics/results_flux_selectivity
  python metrics/summarize_selectivity.py --json_out /tmp/summary.json
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute median selectivity metrics per concept."
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("metrics/results_sdxl_selectivity"),
        help="Directory containing one sub-folder per concept with eval_results.csv",
    )
    p.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional: save best-scale summary to a JSON file",
    )
    return p


def get_medians(rows: list[dict], slider_type: str, scale: float) -> dict | None:
    subset = [r for r in rows
              if r["slider_type"] == slider_type and float(r["scale"]) == scale]
    if not subset:
        return None
    lpips_vals = [float(r["lpips_selectivity"]) for r in subset]
    clip_vals  = [float(r["clip_selectivity"])   for r in subset]
    return {
        "lpips_sel": statistics.median(lpips_vals),
        "clip_sel":  statistics.median(clip_vals),
        "n":         len(subset),
    }


def pick_best_scale(rows: list[dict], scales: list[float]) -> float:
    """Scale with highest clip_sel_median / lpips_sel_median for the specific slider."""
    best_sc, best_score = scales[0], -1.0
    for sc in scales:
        m = get_medians(rows, "specific", sc)
        if m is None:
            continue
        score = m["clip_sel"] / (m["lpips_sel"] + 1e-8)
        if score > best_score:
            best_score, best_sc = score, sc
    return best_sc


def print_full_table(rows: list[dict], scales: list[float], concept: str) -> None:
    w = 60
    print(f"\n  {concept}  (all scales, median over {scales[0]:.0f}-n runs)")
    print(f"  {'scale':<8} {'slider_type':<12} {'lpips_sel':>12}   {'clip_sel':>12}   {'n':>3}")
    print(f"  {'-' * w}")
    for slider_type in ["specific", "general"]:
        for sc in scales:
            m = get_medians(rows, slider_type, sc)
            if m is None:
                continue
            print(
                f"  {sc:<8.1f} {slider_type:<12} "
                f"{m['lpips_sel']:>12.4f}   {m['clip_sel']:>12.4f}   {m['n']:>3}"
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
        scales = sorted(set(float(r["scale"]) for r in rows))
        print_full_table(rows, scales, concept)
        best_sc = pick_best_scale(rows, scales)
        summary[concept] = {
            "best_scale": best_sc,
            "specific": get_medians(rows, "specific", best_sc),
            "general":  get_medians(rows, "general",  best_sc),
        }

    # ── Best-scale summary table ──────────────────────────────────────────────
    sep = "=" * 82
    print(f"\n{sep}")
    print("Best-scale summary  (scale = max clip_sel/lpips_sel for specific slider)")
    print(sep)
    print(
        f"{'Concept':<14} {'Scale':<7} "
        f"{'sp lpips_sel':>14} {'sp clip_sel':>12} "
        f"{'ge lpips_sel':>14} {'ge clip_sel':>12}   n"
    )
    print("-" * 82)
    for concept, d in summary.items():
        sp = d["specific"]
        ge = d["general"] or {}
        ge_lp = ge.get("lpips_sel", float("nan"))
        ge_cl = ge.get("clip_sel",  float("nan"))
        print(
            f"{concept:<14} {d['best_scale']:<7.1f} "
            f"{sp['lpips_sel']:>14.4f} {sp['clip_sel']:>12.4f} "
            f"{ge_lp:>14.4f} {ge_cl:>12.4f}   {sp['n']}"
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved JSON summary → {args.json_out}")

    return summary


if __name__ == "__main__":
    main()
