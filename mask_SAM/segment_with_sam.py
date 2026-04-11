#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase 2 (real editing): segment with SAM")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--mode", type=str, choices=["point", "box", "json", "interactive"], default="point")
    parser.add_argument("--point_x", type=int, default=None)
    parser.add_argument("--point_y", type=int, default=None)
    parser.add_argument("--box", type=int, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--json_prompts", type=Path, default=None)
    parser.add_argument("--image_name", type=str, default="original.png")
    parser.add_argument("--output_name", type=str, default="mask_target.png")
    return parser


def _load_sam_predictor(checkpoint: str, model_type: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SamPredictor(sam)


def _mask_from_point(predictor, point_xy: Tuple[int, int]) -> np.ndarray:
    point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    return masks[int(np.argmax(scores))]


def _mask_from_box(predictor, box_xyxy: List[int]) -> np.ndarray:
    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=True)
    return masks[int(np.argmax(scores))]


def _pick_point_interactively(image: np.ndarray) -> Tuple[int, int]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Click target point, then close window")
    ax.axis("on")
    selected = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not selected:
        raise RuntimeError("No point selected in interactive mode.")
    x, y = selected[0]
    return int(round(x)), int(round(y))


def main() -> None:
    args = build_parser().parse_args()
    image_path = args.run_dir / args.image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for SAM: {image_path}")

    image = np.array(Image.open(image_path).convert("RGB"))
    predictor = _load_sam_predictor(args.sam_checkpoint, args.sam_model_type)
    predictor.set_image(image)

    if args.mode == "point":
        if args.point_x is None or args.point_y is None:
            raise ValueError("point mode requires --point_x and --point_y")
        mask = _mask_from_point(predictor, (args.point_x, args.point_y))
        prompt_meta = {"mode": "point", "point": [args.point_x, args.point_y]}
    elif args.mode == "interactive":
        point_x, point_y = _pick_point_interactively(image)
        print(f"[INFO] Selected point: ({point_x}, {point_y})")
        mask = _mask_from_point(predictor, (point_x, point_y))
        prompt_meta = {"mode": "interactive", "point": [point_x, point_y]}
    elif args.mode == "box":
        if args.box is None:
            raise ValueError("box mode requires --box x1 y1 x2 y2")
        mask = _mask_from_box(predictor, args.box)
        prompt_meta = {"mode": "box", "box": args.box}
    else:
        if args.json_prompts is None:
            raise ValueError("json mode requires --json_prompts")
        payload = json.loads(args.json_prompts.read_text(encoding="utf-8"))
        mode = payload.get("mode")
        if mode == "point":
            p = payload["point"]
            mask = _mask_from_point(predictor, (int(p[0]), int(p[1])))
        elif mode == "box":
            mask = _mask_from_box(predictor, [int(v) for v in payload["box"]])
        else:
            raise ValueError("json mode requires payload mode in {'point','box'}")
        prompt_meta = payload

    out_mask = args.run_dir / args.output_name
    Image.fromarray(mask.astype(np.uint8) * 255).save(out_mask)
    meta = {
        "phase": "sam_segmentation_real",
        "image_name": args.image_name,
        "mask_image": args.output_name,
        "sam_checkpoint": args.sam_checkpoint,
        "sam_model_type": args.sam_model_type,
        "prompt": prompt_meta,
        "mask_resolution": [int(mask.shape[1]), int(mask.shape[0])],
    }
    (args.run_dir / "mask_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Mask saved to: {out_mask}")


if __name__ == "__main__":
    main()
