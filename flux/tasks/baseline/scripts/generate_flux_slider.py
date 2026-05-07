#!/usr/bin/env python
# =============================================================================
# generate_flux_slider.py
# =============================================================================
# Generazione immagini con Flux.1-dev + LoRA concept slider applicata.
#
# Supporta il pattern di validazione "panel" multi-checkpoint + multi-scale:
# carica Flux UNA volta, poi per ogni (checkpoint, scale, prompt, seed)
# ricarica solo i pesi LoRA e genera. Memory-efficient, pipe loaded once.
#
# Uso tipico (validazione v3_yaml_trick):
#   python generate_flux_slider.py \
#       --lora_dirs .../smile_man_flux_v3_yaml_trick_rank16_xattn_alpha1/flux-smile_man_flux_v3_yaml_trick_step200 \
#                   .../smile_man_flux_v3_yaml_trick_rank16_xattn_alpha1/flux-smile_man_flux_v3_yaml_trick_step300 \
#                   .../smile_man_flux_v3_yaml_trick_rank16_xattn_alpha1/flux-smile_man_flux_v3_yaml_trick_step400 \
#                   .../smile_man_flux_v3_yaml_trick_rank16_xattn_alpha1/flux-smile_man_flux_v3_yaml_trick \
#       --lora_scales 0.0 0.5 1.0 1.5 \
#       --prompt "a photo of a man and a woman" \
#       --seeds 42 7 \
#       --save_dir .../outputs/generations/smile_man_v3_panel \
#       --rank 16 --alpha 1.0 --train_method xattn
#
# Output: PNG con naming "ckpt{NAME}__scale{X.XX}__seed{N}__p{I}_{slug}.png"
# Naming permette sort naturale per ispezione visiva a griglia.
#
# Note: --lora_scales 0.0 disabilita la LoRA (baseline di confronto).
# =============================================================================

import os
import sys
import argparse
from pathlib import Path
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path",
                   default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--lora_dirs", nargs="+", required=True,
                   help="Directory contenenti slider_0.pt. Una per checkpoint.")
    p.add_argument("--lora_scales", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5],
                   help="Scale da testare (0.0 = LoRA disabled = baseline).")
    p.add_argument("--prompt", action="append", required=True,
                   help="Prompt. Puoi passare --prompt piu' volte.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds per generazione (stessa coppia across ckpt/scale).")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--height", type=int, default=512,
                   help="512 per velocita' (default). 1024 per quality.")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    # LoRA config (deve matchare quella del training!)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train_method", default="xattn",
                   choices=["xattn", "noxattn", "full"])
    p.add_argument("--cuda_device", default="0")
    p.add_argument("--skip_slider_timestep_till", type=int, default=8,
                   help="Step soglia per attivare lo slider durante il "
                        "denoising. LoRA off per i <= soglia, on per i > soglia. "
                        "Default 8 (uniformato a edit_start_step di "
                        "shop_concept e masked_lora per confronti fair). "
                        "Setting paper Concept Sliders: 0 (LoRA dal step 1).")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import torch

# -----------------------------------------------------------------------------
# Patch compat: torch 2.4 <-> diffusers 0.35+ (scaled_dot_product_attention)
# -----------------------------------------------------------------------------
_torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _torch_ver < (2, 5):
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*sdpa_args, **sdpa_kwargs):
        sdpa_kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*sdpa_args, **sdpa_kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    print(f"[compat] torch {torch.__version__} < 2.5: monkeypatched "
          f"F.scaled_dot_product_attention per strippare enable_gqa kwarg")

print(f"[path] CWD = {os.getcwd()}")

from flux.core.custom_flux_pipeline import FluxPipeline
from flux.core.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
)

# silenzia verbose
import transformers as _tf
_tf.logging.set_verbosity_error()
import diffusers as _df
_df.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Config print
# -----------------------------------------------------------------------------
save_dir = Path(args.save_dir).resolve()
save_dir.mkdir(parents=True, exist_ok=True)

# Valida che tutti i lora_dir esistano e contengano slider_0.pt
lora_paths = []
for d in args.lora_dirs:
    p = Path(d) / "slider_0.pt"
    if not p.exists():
        sys.exit(f"[ERROR] Non trovato: {p}")
    lora_paths.append(p)

print(f"[config] model        = {args.pretrained_model_name_or_path}")
print(f"[config] save_dir     = {save_dir}")
print(f"[config] lora_dirs    = {len(lora_paths)}")
for i, p in enumerate(lora_paths):
    print(f"           [{i}] {p.parent.name} -> {p.name}")
print(f"[config] lora_scales  = {args.lora_scales}")
print(f"[config] rank/alpha   = {args.rank} / {args.alpha}  train_method={args.train_method}")
print(f"[config] prompts      = {len(args.prompt)}")
for i, pp in enumerate(args.prompt):
    print(f"           [{i}] {pp}")
print(f"[config] seeds        = {args.seeds}")
print(f"[config] resolution   = {args.width}x{args.height}")
print(f"[config] steps/cfg    = {args.steps} / {args.guidance_scale}")
print(f"[config] skip_slider  = {args.skip_slider_timestep_till} "
      f"(LoRA on per i > {args.skip_slider_timestep_till})")
n_total = len(lora_paths) * len(args.lora_scales) * len(args.prompt) * len(args.seeds)
print(f"[config] total images = {n_total}")
print()

# -----------------------------------------------------------------------------
# Load pipeline ONCE (bf16, local only)
# -----------------------------------------------------------------------------
print("=== Loading FluxPipeline ===")
t0 = time.time()
pipe = FluxPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.to("cuda:0")
pipe.set_progress_bar_config(disable=True)
print(f"[load] done in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    print(f"[mem post-load] allocated={torch.cuda.memory_allocated()/1e9:.2f} GB")

# -----------------------------------------------------------------------------
# Crea LoRANetwork ONCE (wrapping dei moduli in-place sul transformer).
# Poi per ogni checkpoint ri-carichiamo solo i pesi via load_state_dict.
# -----------------------------------------------------------------------------
print("\n=== Creating LoRANetwork wrapper ===")
net = LoRANetwork(
    pipe.transformer,
    rank=args.rank, multiplier=1.0, alpha=args.alpha,
    train_method=args.train_method,
).to("cuda:0", dtype=torch.bfloat16)
print(f"[lora] wrapper creato (sara' ri-popolato per ogni checkpoint)")

# === BUGFIX baseline scale=0.0 ===
# LoRAModule.__init__ setta multiplier=1.0 di default, e apply_to() ha gia'
# monkey-patchato i .forward dei moduli target. Senza azzerare qui, la prima
# pipe call con scale=0.0 (use_network=None) eseguirebbe comunque la LoRA al
# multiplier iniziale (1.0) -> "baseline" sporcato dalla LoRA caricata.
# Il context manager `with network:` (custom_flux_pipeline) gia' riporta a 0
# nell'__exit__, ma quello non viene mai chiamato per scale=0.0. Forziamo
# qui multiplier=0; nelle scale > 0 il context manager lo rialza al valore
# corretto e poi lo riazzera all'__exit__.
for _m in net.unet_loras:
    _m.multiplier = 0
print(f"[lora] multiplier forzato a 0 (baseline pulito quando scale=0.0)")

# -----------------------------------------------------------------------------
# Generation loop: per ogni checkpoint, carica pesi -> per ogni scale -> gen
# -----------------------------------------------------------------------------
print(f"\n=== Generating {n_total} images ===")
t_total = time.time()
n_done = 0

for ci, lora_path in enumerate(lora_paths):
    ckpt_tag = lora_path.parent.name
    # Estrae suffix "step100" / "step200" / ... o usa "final" se manca _step
    if "_step" in ckpt_tag:
        ckpt_short = "step" + ckpt_tag.split("_step")[-1]
    else:
        ckpt_short = "final"

    print(f"\n--- [{ci+1}/{len(lora_paths)}] Loading LoRA weights: {ckpt_tag} ---")
    # torch.load su un dict di pesi -> load_state_dict sull'network
    state = torch.load(str(lora_path), map_location="cuda:0")
    net.load_state_dict(state)
    print(f"[lora] caricato {lora_path.name}")

    for scale in args.lora_scales:
        # scale=0 -> baseline: NON passiamo network alla pipe (LoRA disabled).
        # scale>0 -> set_lora_slider(scale) + passiamo network=net:
        #           il custom pipeline fa `with network:` che propaga il
        #           multiplier ai wrapper (vedi LoRANetwork.__enter__).
        if scale == 0.0:
            use_network = None
        else:
            net.set_lora_slider(float(scale))
            use_network = net

        for pi, prompt in enumerate(args.prompt):
            slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt)[:40]
            for seed in args.seeds:
                n_done += 1
                tag = f"ckpt{ckpt_short}__scale{scale:.2f}__seed{seed}__p{pi}_{slug}"
                out_path = save_dir / f"{tag}.png"
                if out_path.exists():
                    print(f"  [{n_done}/{n_total}] SKIP: {out_path.name}")
                    continue
                gen = torch.Generator("cuda").manual_seed(seed)
                t_g = time.time()
                img = pipe(
                    prompt=prompt,
                    height=args.height, width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.steps,
                    max_sequence_length=args.max_sequence_length,
                    generator=gen,
                    network=use_network,
                    skip_slider_timestep_till=args.skip_slider_timestep_till,
                ).images[0]
                img.save(out_path)
                dt = time.time() - t_g
                print(f"  [{n_done}/{n_total}] {out_path.name}  ({dt:.1f}s)")

print(f"\n=== Done in {(time.time() - t_total)/60:.1f} min ===")
print(f"Output in: {save_dir}")
print(f"\nTip: per vedere rapidamente la griglia, ordina per nome e scorri:")
print(f"  ls {save_dir}/*.png | sort")
