"""
CLI entrypoint for generating images with Flux.1-dev and one or more
Concept Sliders applied through the LoRAShop-style mask-aware pipeline
(masks derived from the cross-attention of late transformer blocks during
the first few denoising steps; blending per token onwards).

Two usage modes:

(1) One slider per target (implicit 1:1 mapping):
    --slider_paths   smile.pt vangogh.pt
    --target_prompt  man      sky
    --lora_scales    1.0      0.8
    --prompt "a photo of a man under the sky"

(2) Several sliders inside the same target (compositional aggregation:
    PEFT sums the deltas of every active adapter inside the masked
    region):
    --slider_paths     smile.pt age.pt smile.pt age.pt
    --target_prompt    man      woman
    --slider_to_target 0 0 1 1
    --lora_scales      1 1 -1 -1
    --prompt "a man and a woman"
    -> man: positive smile + positive age; woman: negative smile +
       negative age.

In both modes every value in --lora_scales is the continuous strength of
the corresponding Concept Slider (0.0 = off, 1.0 = training strength,
>1.0 = extrapolation), indexed by position in --slider_paths.

Sliders saved in the native Concept-Sliders format (.pt, kohya-ss
`LoRANetwork`) are converted to PEFT safetensors on the fly and cached
under `--cache_dir`. To pre-convert a slider once, use
`convert_slider_to_peft.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import load_file

# Importable both as a module (`python -m flux.tasks.shop_concept.scripts.generate`)
# and as a script (`python flux/tasks/shop_concept/scripts/generate.py`). When
# launched as a script, add the repository root to sys.path so the absolute
# imports below resolve.
if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(_REPO_ROOT))
    from flux.tasks.shop_concept.lib.flux_real_pipeline import RealGenerationPipeline  # noqa: E402
    from flux.tasks.shop_concept.scripts.convert_slider_to_peft import convert_slider_file  # noqa: E402
else:
    from ..lib.flux_real_pipeline import RealGenerationPipeline
    from .convert_slider_to_peft import convert_slider_file


# ---------------------------------------------------------------------------
# Utility: LoRA loading
# ---------------------------------------------------------------------------
def ensure_matching_lora_params(lora_state_dicts, rank: int = 16):
    """Align the keys of several LoRA state dicts by filling missing
    modules with zero tensors.

    Adapted from the equivalent helper in upstream LoRAShop. For Concept
    Sliders trained with the same architecture (`xattn`, same rank) the
    keys are already aligned, so this is normally a no-op.
    """
    ranks = []
    for lora_dict in lora_state_dicts:
        for key in lora_dict.keys():
            if "lora_A" in key:
                ranks.append(lora_dict[key].size(0))
                break

    all_keys = set()
    for lora_dict in lora_state_dicts:
        all_keys.update(lora_dict.keys())

    param_shapes = {}
    for lora_dict in lora_state_dicts:
        for key, param in lora_dict.items():
            if key not in param_shapes:
                param_shapes[key] = param.shape

    for key in all_keys:
        if key not in param_shapes:
            if "lora_A" in key:
                b_key = key.replace("lora_A", "lora_B")
                if b_key in param_shapes:
                    out_dim = param_shapes[b_key][1]
                    param_shapes[key] = (out_dim, rank)
            elif "lora_B" in key:
                a_key = key.replace("lora_B", "lora_A")
                if a_key in param_shapes:
                    out_dim = param_shapes[a_key][0]
                    param_shapes[key] = (rank, out_dim)

    updated_dicts = []
    for i, lora_dict in enumerate(lora_state_dicts):
        updated_dict = lora_dict.copy()
        for key in all_keys:
            if key not in updated_dict:
                if "lora_A" in key:
                    updated_dict[key] = torch.zeros(
                        (ranks[i], param_shapes[key][1]), dtype=torch.bfloat16
                    )
                elif "lora_B" in key:
                    updated_dict[key] = torch.zeros(
                        (param_shapes[key][0], ranks[i]), dtype=torch.bfloat16
                    )
        updated_dicts.append(updated_dict)
    return updated_dicts


def prepare_slider_as_safetensors(slider_path: str, cache_dir: str) -> str:
    """Return a PEFT-format safetensors path for the given slider.

    `.safetensors` inputs are returned unchanged. `.pt` inputs (kohya-ss
    Concept-Sliders native format) are converted to PEFT safetensors inside
    `cache_dir` and the cached file is returned.
    """
    p = Path(slider_path)
    if not p.exists():
        raise FileNotFoundError(f"Slider not found: {slider_path}")
    if p.suffix == ".safetensors":
        return str(p)
    if p.suffix != ".pt":
        raise ValueError(
            f"Unsupported slider extension: {p.suffix}. Expected .pt or .safetensors."
        )

    # Hash on path + mtime so the cache invalidates when the source file changes.
    key = f"{p.resolve()}|{p.stat().st_mtime}".encode()
    digest = hashlib.sha1(key).hexdigest()[:12]
    out_name = f"{p.stem}__{digest}.safetensors"
    out_path = Path(cache_dir) / out_name
    os.makedirs(cache_dir, exist_ok=True)

    if not out_path.exists():
        print(f"[shop_concept] converting {p} -> {out_path}")
        convert_slider_file(
            input_path=str(p),
            output_path=str(out_path),
            fold_alpha=True,
            strict=True,
        )
    else:
        print(f"[shop_concept] cache hit: {out_path}")

    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Genera immagini con Flux + Concept Sliders via LoRAShop masking."
    )

    # Flux model
    parser.add_argument(
        "--model_name",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="HF model name o path locale del Flux checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Flux (cuda|cpu).",
    )

    # Sliders
    parser.add_argument(
        "--slider_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more sliders. Accepts .pt (auto-converted) or .safetensors (PEFT).",
    )
    parser.add_argument(
        "--lora_scales",
        type=float,
        nargs="+",
        default=None,
        help="One continuous scale per --slider_paths (same length). "
             "0.0 = slider off, 1.0 = training strength, >1.0 = extrapolation. "
             "Defaults to 1.0 for every slider if omitted.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="flux/tasks/shop_concept/_peft_cache",
        help="Directory used to cache .pt -> .safetensors slider conversions.",
    )

    # Prompts
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Main prompt; every target_prompt must be a literal substring of it.",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        nargs="+",
        required=True,
        help="One exact substring of --prompt per masked region. Without "
             "--slider_to_target it must have the same length as --slider_paths "
             "(implicit 1:1 mapping). With --slider_to_target the length is the "
             "number of regions (can be smaller than the number of sliders, "
             "since several sliders may map to the same region).",
    )
    parser.add_argument(
        "--slider_to_target",
        type=int,
        nargs="+",
        default=None,
        help="Slider->target mapping. `--slider_to_target 0 0 1` means: "
             "slider 0 and 1 are both applied inside target_prompt 0 "
             "(compositional aggregation, PEFT sums their deltas additively), "
             "and slider 2 is applied inside target_prompt 1. "
             "Length must equal --slider_paths. "
             "If omitted, defaults to the identity (one slider per target).",
    )

    # Generation
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--edit_start_step",
        type=int,
        default=8,
        help="Step from which mask-guided blending becomes active.",
    )
    parser.add_argument(
        "--lora_fill_rank",
        type=int,
        default=16,
        help="Rank used for the zero placeholders inside ensure_matching_lora_params.",
    )

    args = parser.parse_args()

    # -------- Validate --------
    num_sliders = len(args.slider_paths)
    num_targets = len(args.target_prompt)

    if args.slider_to_target is None:
        # Identity mapping: one slider per target.
        if num_targets != num_sliders:
            raise ValueError(
                f"Without --slider_to_target, --target_prompt ({num_targets}) "
                f"must match --slider_paths ({num_sliders}). Use "
                f"--slider_to_target to apply several sliders to the same target."
            )
        slider_to_target = list(range(num_sliders))
    else:
        if len(args.slider_to_target) != num_sliders:
            raise ValueError(
                f"--slider_to_target has {len(args.slider_to_target)} entries "
                f"but --slider_paths has {num_sliders}; one target index per "
                f"slider is required."
            )
        slider_to_target = list(args.slider_to_target)
        for s_idx, t_idx in enumerate(slider_to_target):
            if not (0 <= t_idx < num_targets):
                raise ValueError(
                    f"--slider_to_target[{s_idx}]={t_idx} is out of range "
                    f"[0, {num_targets}); there are {num_targets} target_prompts."
                )

    if args.lora_scales is None:
        lora_scales = [1.0] * num_sliders
    else:
        if len(args.lora_scales) != num_sliders:
            raise ValueError(
                f"Number of --lora_scales ({len(args.lora_scales)}) must match "
                f"the number of --slider_paths ({num_sliders})."
            )
        lora_scales = list(args.lora_scales)

    # Every target_prompt must be a literal substring of the main prompt.
    for tp in args.target_prompt:
        if tp not in args.prompt:
            raise ValueError(
                f"target_prompt '{tp}' is not a literal substring of --prompt; "
                f"the mask-aware pipeline requires an exact substring match."
            )

    # -------- Load Flux pipeline --------
    print(f"[shop_concept] loading Flux pipeline from {args.model_name}")
    pipe = RealGenerationPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to(args.device)

    # -------- Prepare sliders as PEFT safetensors, load, align --------
    print(f"[shop_concept] preparing {num_sliders} slider(s)")
    slider_safetensors = [
        prepare_slider_as_safetensors(p, args.cache_dir) for p in args.slider_paths
    ]
    lora_dicts = [load_file(p) for p in slider_safetensors]
    lora_dicts = ensure_matching_lora_params(lora_dicts, rank=args.lora_fill_rank)

    for i, lora_dict in enumerate(lora_dicts):
        t_idx = slider_to_target[i]
        print(
            f"  [slider {i}] target[{t_idx}]='{args.target_prompt[t_idx]}' "
            f"scale={lora_scales[i]} path={args.slider_paths[i]}"
        )
        # Explicit adapter_name guarantees distinct names default_0..N-1 (the
        # convention expected by `_set_adapter_with_scale` in flux_blocks.py)
        # and allows loading the SAME slider file several times as distinct
        # adapters — useful when the same concept needs to be applied to
        # several targets with different scales (e.g. smile +1 on the man
        # and -1 on the woman).
        pipe.load_lora_weights(lora_dict, adapter_name=f"default_{i}")

    # -------- Patch Flux transformer blocks --------
    print("[shop_concept] registering transformer blocks (mask-aware)")
    pipe.register_transformer_blocks()

    # -------- Generator --------
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    else:
        generator = None

    # -------- Generate --------
    print(f"[shop_concept] generating: prompt='{args.prompt}'")
    print(f"                          targets         ={args.target_prompt}")
    print(f"                          scales (slider) ={lora_scales}")
    print(f"                          slider->target  ={slider_to_target}")
    # When several sliders map to the same target, the pipeline builds the
    # inverse target_to_sliders mapping and activates multiple PEFT adapters
    # simultaneously inside the matching region (their deltas are summed).
    pipe_kwargs = dict(
        prompt=args.prompt,
        target_prompt=args.target_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        height=args.height,
        width=args.width,
        generator=generator,
        edit_start_step=args.edit_start_step,
        target_lora_scales=lora_scales,
    )
    if args.slider_to_target is not None:
        pipe_kwargs["slider_to_target"] = slider_to_target
    result = pipe(**pipe_kwargs)

    # -------- Save --------
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    print(f"[shop_concept] image saved to {out_path}")


if __name__ == "__main__":
    main()
