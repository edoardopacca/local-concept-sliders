#!/usr/bin/env python3
"""
select_best_runs.py — trova i run migliori per ogni concept basandosi su
clip_localization e lpips_localization, e salva un report per il paper.

Usage:
  # SDXL
  python metrics/select_best_runs.py \
      --results_dir metrics/results_sdxl_masked \
      --runs_root   sdxl/tasks/masked_lora/runs \
      --top_k       3

  # Flux
  python metrics/select_best_runs.py \
      --results_dir metrics/results_flux_masked \
      --runs_root   flux/tasks/masked_lora/runs \
      --top_k       3

Output:
  {results_dir}/best_runs_report.json   — top-k per concept con path immagini
  {results_dir}/best_runs_report.txt    — versione human-readable per il paper
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, required=True,
                   help="Directory con i risultati eval (es. metrics/results_sdxl_masked)")
    p.add_argument("--runs_root", type=Path, required=True,
                   help="Root delle run dir (es. sdxl/tasks/masked_lora/runs)")
    p.add_argument("--top_k", type=int, default=3,
                   help="Quante run selezionare per concept")
    p.add_argument("--clip_weight", type=float, default=0.7,
                   help="Peso clip_loc nel combined score (0-1, il resto va a lpips_loc_norm)")
    return p


def load_per_run_metrics(results_dir: Path, concept: str) -> List[Dict]:
    """Legge tutti i file eval_metrics_s*.json dalle run dir."""
    csv_path = results_dir / concept / "eval_results.csv"
    if not csv_path.exists():
        return []

    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("run_id",) else v)
                         for k, v in row.items()})
    return rows


def combined_score(row: Dict, clip_w: float) -> float:
    """Score combinato: clip_loc pesato + lpips_loc_norm normalizzato."""
    clip = row.get("clip_localization", 0.0)
    lpips = row.get("lpips_localization", 0.0)
    return clip_w * clip + (1 - clip_w) * min(lpips / 15.0, 1.0)


def find_edited_images(runs_root: Path, run_id: str, concept_prefix: str,
                       scale_idx: int) -> Optional[Path]:
    """Restituisce il path dell'immagine edited per quella run e scale."""
    run_dir = runs_root / run_id
    # prova sia float (1.5, 2.0, ...) sia int come suffisso
    for p in sorted(run_dir.glob(f"edited_{concept_prefix}_s*.png")):
        return p  # restituisce il primo trovato per quella scale
    return None


CONCEPT_PREFIX = {
    "age_person": "age",
    "curlyhair": "curly",
    "daynight": "daynight",
    "furlength": "furlength",
    "painterly": "painterly",
    "smile_person": "smile",
}


def main() -> None:
    args = build_parser().parse_args()

    concepts = [d.name for d in args.results_dir.iterdir()
                if d.is_dir() and (d / "eval_results.csv").exists()]
    concepts = sorted(concepts)

    if not concepts:
        print(f"[!] Nessun risultato trovato in {args.results_dir}")
        return

    report = {}
    txt_lines = ["=" * 70, "  BEST RUNS REPORT", f"  results: {args.results_dir}",
                 f"  runs:    {args.runs_root}", "=" * 70, ""]

    for concept in concepts:
        rows = load_per_run_metrics(args.results_dir, concept)
        if not rows:
            continue

        # Aggiungi combined score
        for r in rows:
            r["_score"] = combined_score(r, args.clip_weight)

        # Raggruppa per run_id, prendi la scala con score migliore
        best_per_run: Dict[str, Dict] = {}
        for r in rows:
            rid = r["run_id"]
            if rid not in best_per_run or r["_score"] > best_per_run[rid]["_score"]:
                best_per_run[rid] = r

        ranked = sorted(best_per_run.values(), key=lambda x: x["_score"], reverse=True)
        top = ranked[:args.top_k]

        prefix = CONCEPT_PREFIX.get(concept, concept)
        concept_report = []
        txt_lines.append(f"{'─'*60}")
        txt_lines.append(f"  {concept.upper()}")
        txt_lines.append(f"{'─'*60}")

        for rank, row in enumerate(top, 1):
            run_id = row["run_id"]
            scale = int(row["scale"])
            clip = row["clip_localization"]
            lpips_n = row["lpips_localization"]
            lpips_r = row["lpips_loc_raw"]
            score = row["_score"]

            # Trova path immagine edited
            run_dir = args.runs_root / run_id
            edited_imgs = sorted(run_dir.glob(f"edited_{prefix}_s*.png"))
            base_img = run_dir / "base.png"
            mask_img = run_dir / "mask_target.png"

            # Seleziona l'immagine alla scala corrispondente
            target_img = None
            for img in edited_imgs:
                stem = img.stem.split("_s")[-1]
                try:
                    if abs(float(stem) - scale) < 0.01 or int(float(stem)) == scale:
                        target_img = img
                        break
                except ValueError:
                    pass
            if target_img is None and edited_imgs:
                # fallback: scegli l'img con indice = scale-1
                target_img = edited_imgs[min(scale - 1, len(edited_imgs) - 1)]

            entry = {
                "rank": rank,
                "run_id": run_id,
                "best_scale": scale,
                "clip_localization": round(clip, 4),
                "lpips_loc_norm": round(lpips_n, 4),
                "lpips_loc_raw": round(lpips_r, 4),
                "combined_score": round(score, 4),
                "base_img": str(base_img) if base_img.exists() else None,
                "mask_img": str(mask_img) if mask_img.exists() else None,
                "edited_img": str(target_img) if target_img else None,
            }
            concept_report.append(entry)

            txt_lines.append(
                f"  #{rank}  {run_id}  scale={scale}"
            )
            txt_lines.append(
                f"       clip_loc={clip:.3f}  lpips_loc_norm={lpips_n:.2f}"
                f"  lpips_loc_raw={lpips_r:.2f}  score={score:.3f}"
            )
            if target_img:
                txt_lines.append(f"       edited → {target_img}")
            txt_lines.append("")

        report[concept] = concept_report

    # Salva JSON
    json_out = args.results_dir / "best_runs_report.json"
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] JSON  → {json_out}")

    # Salva TXT
    txt_out = args.results_dir / "best_runs_report.txt"
    txt_out.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"[OK] TXT   → {txt_out}")

    # Stampa a schermo
    print("\n".join(txt_lines))


if __name__ == "__main__":
    main()
