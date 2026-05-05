#!/usr/bin/env python3
"""
choose_masks_dual.py — carica SAM una volta, poi per ogni run dir fa:
  1. due click sul TARGET   → due opzioni maschera → scelta → mask_target.png
  2. due click NON-TARGET   → due opzioni maschera → scelta → mask_nontarget.png

Usato per il task selectivity, dove ogni immagine ha due soggetti:
  - target:     il soggetto che lo slider DEVE modificare
  - non-target: il soggetto che lo slider NON deve toccare

Uso:
  python mask_SAM/choose_masks_dual.py \
      --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
      --runs_root sdxl/tasks/selectivity/runs \
      --run_ids eval_age_01 eval_age_02 ...
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", type=Path, default=Path("sdxl/tasks/selectivity/runs"))
    p.add_argument("--sam_checkpoint", type=str, required=True)
    p.add_argument("--sam_model_type", type=str, default="vit_h")
    p.add_argument("--run_ids", nargs="+", required=True)
    return p


def load_sam(checkpoint: str, model_type: str):
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SamPredictor(sam)


def pick_point(image_np: np.ndarray, title: str) -> tuple:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image_np)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    selected = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not selected:
        raise RuntimeError("Nessun punto selezionato.")
    x, y = selected[0]
    return int(round(x)), int(round(y))


def get_mask(predictor, point_xy: tuple) -> np.ndarray:
    coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels, multimask_output=True
    )
    return masks[int(np.argmax(scores))]


def open_image(path: Path) -> None:
    import os, subprocess, sys
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


def show_comparison(
    image_np: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    run_dir: Path,
    prefix: str,
) -> None:
    def overlay(img, mask):
        out = img.copy().astype(float)
        out[mask == 0] *= 0.35
        return out.astype(np.uint8)

    p1 = run_dir / f"_{prefix}_v1_preview.png"
    p2 = run_dir / f"_{prefix}_v2_preview.png"
    Image.fromarray(overlay(image_np, mask1)).save(p1)
    Image.fromarray(overlay(image_np, mask2)).save(p2)
    open_image(p1)
    open_image(p2)


def do_two_clicks(predictor, image_np: np.ndarray, run_id: str, label: str):
    print(f"  {label} — click 1: clicca sul soggetto, poi chiudi la finestra")
    pt1 = pick_point(image_np, f"{run_id} — {label} click 1  (chiudi dopo)")
    mask1 = get_mask(predictor, pt1)
    print(f"     punto: {pt1}")

    print(f"  {label} — click 2: clicca in un altro punto sul soggetto, poi chiudi")
    pt2 = pick_point(image_np, f"{run_id} — {label} click 2  (chiudi dopo)")
    mask2 = get_mask(predictor, pt2)
    print(f"     punto: {pt2}")

    return mask1, mask2


def select_mask(predictor, image_np: np.ndarray, run_dir: Path,
                run_id: str, label: str, prefix: str) -> np.ndarray:
    """Interactive loop: two clicks, show comparison, choose or retry."""
    mask1, mask2 = do_two_clicks(predictor, image_np, run_id, label)

    while True:
        print(f"  Apro le due opzioni maschera {label} in Preview...")
        show_comparison(image_np, mask1, mask2, run_dir, prefix)

        choice = input(f"  {label} — scegli [1 / 2 / r=ripeti i click]: ").strip().lower()
        if choice == "1":
            chosen = mask1
            break
        elif choice == "2":
            chosen = mask2
            break
        elif choice == "r":
            mask1, mask2 = do_two_clicks(predictor, image_np, run_id, label)
        else:
            print("  Digita 1, 2, oppure r.")

    for tmp in (run_dir / f"_{prefix}_v1_preview.png",
                run_dir / f"_{prefix}_v2_preview.png"):
        tmp.unlink(missing_ok=True)

    return chosen


def main() -> None:
    args = build_parser().parse_args()

    print(f"Carico SAM da {args.sam_checkpoint} ...")
    predictor = load_sam(args.sam_checkpoint, args.sam_model_type)
    print("SAM caricato.\n")

    total = len(args.run_ids)
    for idx, run_id in enumerate(args.run_ids, 1):
        run_dir = args.runs_root / run_id
        image_path = run_dir / "base.png"

        print(f"\n[{idx}/{total}] === {run_id} ===")

        if not image_path.exists():
            print(f"  [SKIP] base.png non trovato in {run_dir}")
            continue

        image_np = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image_np)

        # --- TARGET mask -------------------------------------------------
        print("\n  >>> MASCHERA TARGET (soggetto che lo slider DEVE modificare) <<<")
        mask_target = select_mask(
            predictor, image_np, run_dir, run_id,
            label="TARGET", prefix="target",
        )
        out_target = run_dir / "mask_target.png"
        Image.fromarray(mask_target.astype(np.uint8) * 255).save(out_target)
        print(f"  [OK] mask_target  → {out_target}")

        # --- NON-TARGET mask ---------------------------------------------
        print("\n  >>> MASCHERA NON-TARGET (soggetto che lo slider NON deve toccare) <<<")
        mask_nontarget = select_mask(
            predictor, image_np, run_dir, run_id,
            label="NON-TARGET", prefix="nontarget",
        )
        out_nontarget = run_dir / "mask_nontarget.png"
        Image.fromarray(mask_nontarget.astype(np.uint8) * 255).save(out_nontarget)
        print(f"  [OK] mask_nontarget → {out_nontarget}")

    print("\n=== FATTO: tutte le maschere salvate ===")


if __name__ == "__main__":
    main()
