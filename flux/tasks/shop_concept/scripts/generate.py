"""
shop_concept/generate.py
========================
Entrypoint CLI per generare immagini con Flux applicando uno o piu' Concept
Sliders alla maniera di LoRAShop (mask estratte dall'attenzione del block
19, blending per-token).

Due modalita' d'uso:

(1) 1 slider per target (retro-compat, mapping implicito 1:1):
    --slider_paths   smile.pt vangogh.pt
    --target_prompt  man      sky
    --lora_scales    1.0      0.8
    --prompt "a photo of a man under the sky"

(2) Multi slider per target (composizione paper-style, somma additiva
    delle delta nella stessa regione mascherata):
    --slider_paths     smile.pt age.pt smile.pt age.pt
    --target_prompt    man      woman
    --slider_to_target 0 0 1 1
    --lora_scales      1 1 -1 -1
    --prompt "a man and a woman"
    -> uomo: smile+age positivi compositi; donna: smile+age negativi.

In entrambe le modalita' ogni --lora_scales e' il valore continuo del
Concept Slider per il singolo slider (0.0=off, 1.0=full training strength,
>1.0=extrapolation), indicizzato per posizione in --slider_paths.

Gli slider in formato nativo (.pt, LoRANetwork kohya) vengono convertiti
on-the-fly a safetensors PEFT in una cache temporanea. Per pre-convertirli
una volta per sempre, usa convert_slider_to_peft.py.
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

# Esecuzione sia come modulo (python -m shop_concept.generate)
# sia come script (python shop_concept/generate.py).
if __package__ is None or __package__ == "":
    # Lanciato come script: aggiungi la repo root a sys.path per gli import assoluti.
    # __file__ = .../flux/tasks/shop_concept/scripts/generate.py → parents[4] = repo root
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
    """Allinea le chiavi tra piu' LoRA, riempendo con zeri i moduli mancanti.

    NOTA: derivato da LoRAShop-main/main.py. Per Concept Sliders addestrati
    con la stessa architettura (xattn, stesso rank), le chiavi dovrebbero
    gia' coincidere tra tutti gli slider, quindi questo e' di solito no-op.
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
    """Se il file e' .safetensors, ritorna path as-is.
    Se e' .pt, converte in safetensors dentro cache_dir e ritorna il path.
    """
    p = Path(slider_path)
    if not p.exists():
        raise FileNotFoundError(f"Slider non trovato: {slider_path}")
    if p.suffix == ".safetensors":
        return str(p)
    if p.suffix != ".pt":
        raise ValueError(
            f"Estensione slider non supportata: {p.suffix}. Serve .pt o .safetensors."
        )

    # Hash path+mtime per cache stabile
    key = f"{p.resolve()}|{p.stat().st_mtime}".encode()
    digest = hashlib.sha1(key).hexdigest()[:12]
    out_name = f"{p.stem}__{digest}.safetensors"
    out_path = Path(cache_dir) / out_name
    os.makedirs(cache_dir, exist_ok=True)

    if not out_path.exists():
        print(f"[shop_concept] converto {p} -> {out_path}")
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
        help="Device per Flux (cuda|cpu).",
    )

    # Sliders
    parser.add_argument(
        "--slider_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path a uno o piu' slider. Accetta .pt (auto-convert) o .safetensors PEFT.",
    )
    parser.add_argument(
        "--lora_scales",
        type=float,
        nargs="+",
        default=None,
        help="Una scale continua per ogni --slider_paths (stessa lunghezza). "
             "0.0 = slider off, 1.0 = strength training, >1.0 = extrapolation. "
             "Se omesso, default 1.0 per tutti.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="flux/tasks/shop_concept/_peft_cache",
        help="Dove mettere le conversioni .pt -> .safetensors.",
    )

    # Prompts
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt principale (ogni target_prompt deve esserne sottostringa).",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        nargs="+",
        required=True,
        help="Una sottostringa esatta di --prompt per ogni regione mascherata. "
             "Senza --slider_to_target: deve avere la stessa lunghezza di "
             "--slider_paths (mapping 1:1 implicito). "
             "Con --slider_to_target: lunghezza pari al numero di REGIONI "
             "(puo' essere minore degli slider, perche' piu' slider possono "
             "puntare alla stessa regione).",
    )
    parser.add_argument(
        "--slider_to_target",
        type=int,
        nargs="+",
        default=None,
        help="Mappa slider->target. `--slider_to_target 0 0 1` significa: "
             "slider 0 e 1 vanno entrambi sul target_prompt 0 (composizione "
             "paper-style: somma additiva delle delta nella stessa regione), "
             "slider 2 va sul target_prompt 1. "
             "Lunghezza == numero di --slider_paths. "
             "Se omesso, default identita' (1 slider per target, retro-compat).",
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
        help="Step a partire dal quale inizia il blending mask-guidato.",
    )
    parser.add_argument(
        "--lora_fill_rank",
        type=int,
        default=16,
        help="Rank per i placeholder zeri in ensure_matching_lora_params.",
    )

    args = parser.parse_args()

    # -------- Validate --------
    num_sliders = len(args.slider_paths)
    num_targets = len(args.target_prompt)

    if args.slider_to_target is None:
        # Retro-compat: 1 slider per target.
        if num_targets != num_sliders:
            raise ValueError(
                f"Senza --slider_to_target, --target_prompt ({num_targets}) deve "
                f"matchare --slider_paths ({num_sliders}). Per piu' slider sullo "
                f"stesso target usa --slider_to_target."
            )
        slider_to_target = list(range(num_sliders))
    else:
        if len(args.slider_to_target) != num_sliders:
            raise ValueError(
                f"--slider_to_target ha {len(args.slider_to_target)} elementi ma "
                f"--slider_paths ne ha {num_sliders}. Serve un target_idx per ogni "
                f"slider."
            )
        slider_to_target = list(args.slider_to_target)
        for s_idx, t_idx in enumerate(slider_to_target):
            if not (0 <= t_idx < num_targets):
                raise ValueError(
                    f"--slider_to_target[{s_idx}]={t_idx} fuori range "
                    f"[0, {num_targets}). Hai {num_targets} target_prompt."
                )

    if args.lora_scales is None:
        lora_scales = [1.0] * num_sliders
    else:
        if len(args.lora_scales) != num_sliders:
            raise ValueError(
                f"Numero di --lora_scales ({len(args.lora_scales)}) deve "
                f"matchare numero di --slider_paths ({num_sliders})."
            )
        lora_scales = list(args.lora_scales)

    # Tutti i target_prompt devono essere sottostringhe del prompt
    for tp in args.target_prompt:
        if tp not in args.prompt:
            raise ValueError(
                f"target_prompt '{tp}' non trovato letteralmente in --prompt. "
                f"LoRAShop richiede che sia una sottostringa esatta."
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
        # adapter_name esplicito: garantisce nomi distinti default_0..N-1
        # (allinea con _set_adapter_with_scale in flux_blocks.py) e supporta
        # il caricamento dello STESSO file slider piu' volte come adapter
        # distinti (utile quando un concept va applicato a target multipli
        # con scale diverse — es. smile +1 sul man e -1 sulla woman).
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
    # In modalita' multi-LoRA per target, la pipeline costruisce il mapping
    # target_to_sliders e attiva piu' adapter PEFT contemporaneamente sulla
    # stessa regione mascherata (somma additiva delle delta).
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
