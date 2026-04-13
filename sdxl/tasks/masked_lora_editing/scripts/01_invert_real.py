#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline


def _load_exp_editing_module(repo_root: Path):
    """DEPRECATO. Dipendenza da exp_editing/ (cancellato nella riorganizzazione).

    Per real image editing usa il task moderno:
        sdxl/tasks/real_editing/scripts/invert_real_image.py
    """
    raise RuntimeError(
        "DEPRECATO: questo script dipendeva da exp_editing/edit_with_sliders.py "
        "(cancellato nella riorganizzazione 2026). Per la real image inversion "
        "usa invece:\n\n"
        "    python sdxl/tasks/real_editing/scripts/invert_real_image.py\n\n"
        "Se ti serve davvero il vecchio NullInversion (SD1.x), recuperalo da "
        "git: git show 0a4bdcc:exp_editing/edit_with_sliders.py"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Phase 1 (real editing): invert real image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("sdxl/tasks/masked_lora_editing/outputs"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inner_steps", type=int, default=10)
    parser.add_argument("--early_stop_epsilon", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float32")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def dtype_from_str(dtype_name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[dtype_name]


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mod = _load_exp_editing_module(repo_root)
    NullInversion = mod.NullInversion

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=dtype_from_str(args.dtype),
        local_files_only=args.local_files_only,
    ).to(args.device)
    try:
        pipe.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass

    inverter = NullInversion(pipe, num_ddim_steps=args.steps, guidance_scale=args.guidance_scale)
    (image_gt, image_rec), x_t, uncond_embeddings = inverter.invert(
        str(args.image),
        args.prompt,
        num_inner_steps=args.num_inner_steps,
        early_stop_epsilon=args.early_stop_epsilon,
    )

    Image.fromarray(image_gt).save(run_dir / "original.png")
    Image.fromarray(image_rec).save(run_dir / "reconstruction.png")
    torch.save(x_t.detach().cpu(), run_dir / "x_t.pt")
    stacked_uncond = torch.stack([u.squeeze(0).cpu() for u in uncond_embeddings], dim=0)
    torch.save(stacked_uncond, run_dir / "uncond_embeddings.pt")

    metadata = {
        "phase": "real_image_inversion",
        "created_at": datetime.utcnow().isoformat(),
        "model_id": args.model_id,
        "image_path": str(args.image),
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_inner_steps": args.num_inner_steps,
        "early_stop_epsilon": args.early_stop_epsilon,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] Inversion artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
