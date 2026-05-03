#!/usr/bin/env python3
"""
place_masks.py — copia mask_target.png da masks_output/ nelle run dir corrette.

La struttura attesa di masks_output/ (prodotta dalla fase 2 SAM):
  masks_output/
    eval_age_person_01/
      mask_target.png
    eval_curlyhair_07/
      mask_target.png
    ...

Lo script copia ogni mask_target.png nella run dir corrispondente:
  {runs_root}/eval_age_person_01/mask_target.png
  {runs_root}/eval_curlyhair_07/mask_target.png
  ...

Uso:
  # Flux (default)
  python mask_SAM/place_masks.py

  # SDXL
  python mask_SAM/place_masks.py --runs_root sdxl/tasks/masked_lora/runs

  # con cartella masks_output in un path diverso
  python mask_SAM/place_masks.py --masks_dir /percorso/masks_output
"""
import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Copia le maschere SAM nelle run dir.")
    p.add_argument(
        "--masks_dir", type=Path, default=Path("masks_output"),
        help="Cartella con le maschere prodotte dalla fase 2 (default: masks_output/)",
    )
    p.add_argument(
        "--runs_root", type=Path, default=Path("flux/tasks/masked_lora/runs"),
        help="Root delle run dir dove copiare le maschere "
             "(default: flux/tasks/masked_lora/runs)",
    )
    p.add_argument(
        "--mask_name", type=str, default="mask_target.png",
        help="Nome del file maschera da cercare e copiare (default: mask_target.png)",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Mostra cosa verrebbe copiato senza farlo davvero.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.masks_dir.exists():
        print(f"[ERRORE] masks_dir non trovata: {args.masks_dir}")
        print("  Decomprimi masks_output.zip nella root della repo e riprova.")
        return

    run_dirs = sorted(p for p in args.masks_dir.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"[ERRORE] Nessuna sottocartella trovata in {args.masks_dir}")
        return

    ok = 0
    skip_no_mask = 0
    skip_no_dest = 0

    for src_run in run_dirs:
        src_mask = src_run / args.mask_name
        if not src_mask.exists():
            print(f"  [SKIP] nessun {args.mask_name} in {src_run.name}")
            skip_no_mask += 1
            continue

        dst_run = args.runs_root / src_run.name
        if not dst_run.exists():
            print(f"  [SKIP] run dir destinazione non trovata: {dst_run}")
            skip_no_dest += 1
            continue

        dst_mask = dst_run / args.mask_name
        if args.dry_run:
            print(f"  [DRY]  {src_mask}  →  {dst_mask}")
        else:
            shutil.copy2(src_mask, dst_mask)
            print(f"  [OK]   {src_run.name}/{args.mask_name}  →  {dst_run}")
        ok += 1

    print()
    print(f"Copiate: {ok}  |  skip (no mask): {skip_no_mask}  |  skip (no run dir): {skip_no_dest}")
    if args.dry_run:
        print("(dry run — nessun file è stato copiato)")


if __name__ == "__main__":
    main()
