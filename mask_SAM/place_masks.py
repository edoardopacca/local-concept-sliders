#!/usr/bin/env python3
"""
place_masks.py — copy mask_target.png from masks_output/ into the
matching run directories.

Expected layout of masks_output/ (produced by SAM phase 2):
  masks_output/
    eval_age_person_01/
      mask_target.png
    eval_curlyhair_07/
      mask_target.png
    ...

The script copies every mask_target.png into the matching run directory:
  {runs_root}/eval_age_person_01/mask_target.png
  {runs_root}/eval_curlyhair_07/mask_target.png
  ...

Usage:
  # Flux (default)
  python mask_SAM/place_masks.py

  # SDXL
  python mask_SAM/place_masks.py --runs_root sdxl/tasks/masked_lora/runs

  # with masks_output stored at a different path
  python mask_SAM/place_masks.py --masks_dir /path/to/masks_output
"""
import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Copy SAM masks into the matching run directories.")
    p.add_argument(
        "--masks_dir", type=Path, default=Path("masks_output"),
        help="Folder with the masks produced by SAM phase 2 (default: masks_output/)",
    )
    p.add_argument(
        "--runs_root", type=Path, default=Path("flux/tasks/masked_lora/runs"),
        help="Root of the run dirs where the masks should be copied "
             "(default: flux/tasks/masked_lora/runs)",
    )
    p.add_argument(
        "--mask_name", type=str, default="mask_target.png",
        help="Mask filename to look for and copy (default: mask_target.png)",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be copied without actually copying anything.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.masks_dir.exists():
        print(f"[ERROR] masks_dir not found: {args.masks_dir}")
        print("  Unzip masks_output.zip in the repo root and try again.")
        return

    run_dirs = sorted(p for p in args.masks_dir.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"[ERROR] No subdirectory found in {args.masks_dir}")
        return

    ok = 0
    skip_no_mask = 0
    skip_no_dest = 0

    for src_run in run_dirs:
        src_mask = src_run / args.mask_name
        if not src_mask.exists():
            print(f"  [SKIP] no {args.mask_name} in {src_run.name}")
            skip_no_mask += 1
            continue

        dst_run = args.runs_root / src_run.name
        if not dst_run.exists():
            print(f"  [SKIP] destination run dir not found: {dst_run}")
            skip_no_dest += 1
            continue

        dst_mask = dst_run / args.mask_name
        if args.dry_run:
            print(f"  [DRY]  {src_mask}  ->  {dst_mask}")
        else:
            shutil.copy2(src_mask, dst_mask)
            print(f"  [OK]   {src_run.name}/{args.mask_name}  ->  {dst_run}")
        ok += 1

    print()
    print(f"Copied: {ok}  |  skip (no mask): {skip_no_mask}  |  skip (no run dir): {skip_no_dest}")
    if args.dry_run:
        print("(dry run -- no file was copied)")


if __name__ == "__main__":
    main()
