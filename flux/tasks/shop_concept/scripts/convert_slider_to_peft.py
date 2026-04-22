"""
shop_concept/convert_slider_to_peft.py
=======================================
Converte un Concept Slider (.pt salvato da LoRANetwork.save_weights) nel
formato PEFT safetensors atteso da LoRAShop.

Formato sorgente (slider_{i}.pt, kohya-ss / LoRANetwork):
    lora_unet_<flat_dotted_path>.lora_down.weight   shape (rank, in_dim)
    lora_unet_<flat_dotted_path>.lora_up.weight     shape (out_dim, rank)
    lora_unet_<flat_dotted_path>.alpha              scalar buffer

dove <flat_dotted_path> = module_path.replace('.', '_')
    es. "transformer_blocks_0_attn_to_q"
        "transformer_blocks.0.attn.to_q" e' il path originale

Nel training flux_slider.py il multiplier finale del LoRA vale:
    out = org(x) + lora_up(lora_down(x)) * multiplier * (alpha / rank)

Formato destinazione (PEFT diffusers safetensors):
    transformer.<dotted_path>.lora_A.weight   shape (rank, in_dim)  <- ex lora_down
    transformer.<dotted_path>.lora_B.weight   shape (out_dim, rank) <- ex lora_up * (alpha/rank)

Scelte:
  * Alpha e' FOLDATO dentro lora_B moltiplicando lora_up per (alpha/rank).
    Cosi' PEFT scaling resta 1.0 di default e il parametro `--lora_scale`
    continuo dello slider e' moltiplicato esplicitamente sopra. Questo
    semplifica molto la logica per-target in flux_blocks.py.

  * Enumeriamo staticamente i 266 moduli LoRA attesi dal training
    `train_method='xattn'` su Flux 1.0 (19 double blocks x 8 linear attn
    + 38 single blocks x 3 linear attn). Nessun bisogno di caricare Flux.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file


# Leaf-names di moduli Linear sotto "attn" per un blocco Flux.
# Questi derivano dalla class diffusers.FluxAttention (e dalla variante
# SingleTransformerBlock che ha solo Q/K/V di base).
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

# Numero di blocchi nell'architettura Flux 1.0 / 1.0-dev.
_NUM_DOUBLE_BLOCKS_FLUX1 = 19
_NUM_SINGLE_BLOCKS_FLUX1 = 38


def build_flux_lora_target_map(
    num_double_blocks: int = _NUM_DOUBLE_BLOCKS_FLUX1,
    num_single_blocks: int = _NUM_SINGLE_BLOCKS_FLUX1,
) -> Dict[str, str]:
    """
    Costruisce la mappa
        {kohya_flat_name -> dotted_path}
    per TUTTI i Linear target di un LoRANetwork addestrato con
    `train_method='xattn'` su Flux 1.0.

    Esempio:
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
    """
    Converte lo state_dict di un Concept Slider nel formato PEFT per Flux.

    Ritorna:
        (converted_state_dict, metadata_dict)

    Se `fold_alpha=True`: lora_B viene moltiplicato per (alpha/rank) e i
    tensor alpha vengono scartati.
    Se `fold_alpha=False`: lora_A, lora_B, alpha vengono tutti preservati
    (utile se vuoi che PEFT gestisca l'alpha in autonomia; non testato).
    """
    flat_to_dotted = build_flux_lora_target_map(num_double_blocks, num_single_blocks)

    # Raggruppa le chiavi per modulo kohya-flat
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if "." not in key:
            if strict:
                raise ValueError(f"Chiave senza punto non attesa: {key}")
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
                f"Modulo {flat_name} privo di lora_down.weight o lora_up.weight "
                f"(chiavi trovate: {list(parts.keys())})"
            )

        rank = lora_down.shape[0]
        if alpha_tensor is None:
            alpha_val = float(rank)  # default kohya: alpha=rank -> scale=1
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
            f"{len(unmatched)} modulo/i non mappato/i (es. {unmatched[:3]}). "
            f"Il checkpoint ha una forma architetturale inattesa "
            f"(non Flux 1.0 xattn?)."
        )

    # Uniforma tutti i tensori in bfloat16 per compatibilita' con Flux pipeline
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
    """Load un .pt slider, converte, salva .safetensors PEFT."""
    print(f"[convert] loading {input_path}")
    state_dict = torch.load(input_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Atteso dict da torch.load, ottenuto {type(state_dict)}. "
            f"E' sicuro un checkpoint di LoRANetwork.save_weights()?"
        )

    converted, metadata = convert_slider_state_dict(
        state_dict, fold_alpha=fold_alpha, strict=strict
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    save_file(converted, output_path, metadata=metadata)
    print(
        f"[convert] wrote {output_path}  "
        f"({metadata['num_modules']} moduli, "
        f"alpha_folded={metadata['alpha_folded_into_lora_B']})"
    )


def _main():
    parser = argparse.ArgumentParser(
        description="Converte slider_X.pt (LoRANetwork) -> PEFT safetensors"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path al file .pt del concept slider (slider_0.pt tipico).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path di destinazione (.safetensors).",
    )
    parser.add_argument(
        "--no_fold_alpha",
        action="store_true",
        help="Se passato, non folda alpha/rank dentro lora_B. "
             "Sconsigliato: LoRAShop non legge alpha separati.",
    )
    parser.add_argument(
        "--lax",
        action="store_true",
        help="Consenti chiavi non mappate (warning invece di errore).",
    )
    args = parser.parse_args()

    if not args.output.endswith(".safetensors"):
        raise ValueError("--output deve terminare con .safetensors")

    convert_slider_file(
        input_path=args.input,
        output_path=args.output,
        fold_alpha=not args.no_fold_alpha,
        strict=not args.lax,
    )


if __name__ == "__main__":
    _main()
