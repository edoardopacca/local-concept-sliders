"""
Convert a Concept Slider (.pt saved by ``LoRANetwork.save_weights``) into
the PEFT-format safetensors expected by the LoRAShop-style pipeline.

Source format (slider_{i}.pt, kohya-ss / LoRANetwork):
    lora_unet_<flat_dotted_path>.lora_down.weight   shape (rank, in_dim)
    lora_unet_<flat_dotted_path>.lora_up.weight     shape (out_dim, rank)
    lora_unet_<flat_dotted_path>.alpha              scalar buffer

where ``<flat_dotted_path> = module_path.replace('.', '_')``, e.g.
``transformer_blocks_0_attn_to_q`` is the flat form of
``transformer_blocks.0.attn.to_q``.

In the original training script the LoRA contribution is applied as
    out = org(x) + lora_up(lora_down(x)) * multiplier * (alpha / rank)

Target format (PEFT diffusers safetensors):
    transformer.<dotted_path>.lora_A.weight   shape (rank, in_dim)  <- former lora_down
    transformer.<dotted_path>.lora_B.weight   shape (out_dim, rank) <- former lora_up * (alpha/rank)

Design choices:

  * ``alpha`` is FOLDED into lora_B by multiplying lora_up by
    ``(alpha / rank)``. This keeps PEFT's default scaling at 1.0 and lets
    the pipeline apply the continuous slider scale explicitly on top,
    which simplifies the per-target logic in `flux_blocks.py`.
  * The list of LoRA modules is enumerated statically: Flux 1.0 with
    ``train_method='xattn'`` produces 266 linear modules (19 double
    blocks x 8 attention linears + 38 single blocks x 3 attention
    linears). Flux itself does not need to be loaded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file


# Leaf names of the Linear modules under "attn" inside a Flux block.
# Derived from diffusers.FluxAttention (and from the SingleTransformerBlock
# variant, which only has Q/K/V).
_DOUBLE_BLOCK_ATTN_LEAVES: Tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
)
_SINGLE_BLOCK_ATTN_LEAVES: Tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
)

# Number of blocks in the Flux 1.0 / 1.0-dev architecture.
_NUM_DOUBLE_BLOCKS_FLUX1 = 19
_NUM_SINGLE_BLOCKS_FLUX1 = 38


def build_flux_lora_target_map(
    num_double_blocks: int = _NUM_DOUBLE_BLOCKS_FLUX1,
    num_single_blocks: int = _NUM_SINGLE_BLOCKS_FLUX1,
) -> Dict[str, str]:
    """Build the
        ``{kohya_flat_name -> dotted_path}``
    map covering every Linear targeted by a ``LoRANetwork`` trained with
    ``train_method='xattn'`` on Flux 1.0.

    Example::
        'lora_unet_transformer_blocks_0_attn_to_q'
            -> 'transformer_blocks.0.attn.to_q'
        'lora_unet_transformer_blocks_0_attn_to_out_0'
            -> 'transformer_blocks.0.attn.to_out.0'
    """
    mapping: Dict[str, str] = {}

    for i in range(num_double_blocks):
        base_dotted = f"transformer_blocks.{i}.attn"
        for leaf in _DOUBLE_BLOCK_ATTN_LEAVES:
            dotted = f"{base_dotted}.{leaf}"
            flat = "lora_unet_" + dotted.replace(".", "_")
            mapping[flat] = dotted

    for i in range(num_single_blocks):
        base_dotted = f"single_transformer_blocks.{i}.attn"
        for leaf in _SINGLE_BLOCK_ATTN_LEAVES:
            dotted = f"{base_dotted}.{leaf}"
            flat = "lora_unet_" + dotted.replace(".", "_")
            mapping[flat] = dotted

    return mapping


def convert_slider_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fold_alpha: bool = True,
    lora_key_prefix: str = "transformer",
    strict: bool = True,
    num_double_blocks: int = _NUM_DOUBLE_BLOCKS_FLUX1,
    num_single_blocks: int = _NUM_SINGLE_BLOCKS_FLUX1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Convert the state dict of a Concept Slider into the Flux PEFT format.

    Returns:
        ``(converted_state_dict, metadata_dict)``

    If ``fold_alpha=True`` (default): ``lora_B`` is multiplied by
    ``alpha / rank`` and the alpha tensors are discarded.
    If ``fold_alpha=False``: ``lora_A``, ``lora_B`` and ``alpha`` are all
    preserved (useful if PEFT is left to handle alpha by itself; untested).
    """
    flat_to_dotted = build_flux_lora_target_map(num_double_blocks, num_single_blocks)

    # Group source keys by kohya-flat module name.
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if "." not in key:
            if strict:
                raise ValueError(f"Unexpected key without a dot separator: {key}")
            continue
        flat_name, subkey = key.split(".", 1)
        modules.setdefault(flat_name, {})[subkey] = tensor

    converted: Dict[str, torch.Tensor] = {}
    unmatched: List[str] = []
    matched = 0
    alphas_summary: Dict[str, float] = {}

    for flat_name, parts in modules.items():
        if flat_name not in flat_to_dotted:
            unmatched.append(flat_name)
            continue

        dotted = flat_to_dotted[flat_name]
        peft_base = f"{lora_key_prefix}.{dotted}"

        lora_down = parts.get("lora_down.weight")
        lora_up = parts.get("lora_up.weight")
        alpha_tensor = parts.get("alpha")

        if lora_down is None or lora_up is None:
            raise ValueError(
                f"Module {flat_name} is missing lora_down.weight or "
                f"lora_up.weight (keys found: {list(parts.keys())})"
            )

        rank = lora_down.shape[0]
        if alpha_tensor is None:
            alpha_val = float(rank)  # kohya default: alpha == rank → scale = 1
        else:
            alpha_val = float(alpha_tensor.detach().cpu().item())
        alphas_summary[flat_name] = alpha_val

        if fold_alpha:
            scale = alpha_val / rank
            converted[f"{peft_base}.lora_A.weight"] = lora_down.detach().clone()
            converted[f"{peft_base}.lora_B.weight"] = (
                lora_up.detach().clone() * scale
            )
        else:
            converted[f"{peft_base}.lora_A.weight"] = lora_down.detach().clone()
            converted[f"{peft_base}.lora_B.weight"] = lora_up.detach().clone()
            converted[f"{peft_base}.alpha"] = (
                alpha_tensor.detach().clone() if alpha_tensor is not None
                else torch.tensor(alpha_val)
            )
        matched += 1

    if unmatched and strict:
        raise ValueError(
            f"{len(unmatched)} module(s) could not be mapped "
            f"(e.g. {unmatched[:3]}). The checkpoint has an unexpected "
            f"architectural shape (not Flux 1.0 xattn?)."
        )

    # Cast every tensor to bfloat16 for compatibility with the Flux pipeline.
    for k in list(converted.keys()):
        converted[k] = converted[k].to(torch.bfloat16).contiguous()

    metadata = {
        "format": "pt",
        "source": "concept-sliders LoRANetwork (Flux xattn)",
        "num_modules": str(matched),
        "alpha_folded_into_lora_B": str(fold_alpha),
        "distinct_alphas": json.dumps(sorted(set(alphas_summary.values()))),
    }
    return converted, metadata


def convert_slider_file(
    input_path: str,
    output_path: str,
    fold_alpha: bool = True,
    strict: bool = True,
) -> None:
    """Load a .pt slider, convert it, and save the PEFT .safetensors."""
    print(f"[convert] loading {input_path}")
    state_dict = torch.load(input_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected dict from torch.load, got {type(state_dict)}. "
            f"Is this really a LoRANetwork.save_weights() checkpoint?"
        )

    converted, metadata = convert_slider_state_dict(
        state_dict, fold_alpha=fold_alpha, strict=strict
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    save_file(converted, output_path, metadata=metadata)
    print(
        f"[convert] wrote {output_path}  "
        f"({metadata['num_modules']} modules, "
        f"alpha_folded={metadata['alpha_folded_into_lora_B']})"
    )


def _main():
    parser = argparse.ArgumentParser(
        description="Convert slider_X.pt (LoRANetwork) into PEFT safetensors."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the .pt Concept Slider checkpoint (typically slider_0.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination path (.safetensors).",
    )
    parser.add_argument(
        "--no_fold_alpha",
        action="store_true",
        help="Skip folding alpha/rank into lora_B. Discouraged: the "
             "mask-aware pipeline does not read separate alpha tensors.",
    )
    parser.add_argument(
        "--lax",
        action="store_true",
        help="Allow unmapped keys (warning instead of error).",
    )
    args = parser.parse_args()

    if not args.output.endswith(".safetensors"):
        raise ValueError("--output must end with .safetensors")

    convert_slider_file(
        input_path=args.input,
        output_path=args.output,
        fold_alpha=not args.no_fold_alpha,
        strict=not args.lax,
    )


if __name__ == "__main__":
    _main()
