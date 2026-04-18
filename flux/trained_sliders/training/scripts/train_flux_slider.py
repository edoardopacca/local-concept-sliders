#!/usr/bin/env python
# =============================================================================
# train_flux_slider.py
# =============================================================================
# Linearized version of flux-sliders/train-flux-concept-sliders.ipynb
# con T5 offload patch per girare su MIG A100 3g.40gb (42 GB VRAM).
#
# Differenze chiave rispetto al notebook originale:
#   (1) T5 + CLIP-L vengono cancellati dalla GPU DOPO aver pre-computato
#       gli embeddings dei 3 prompt (target/positive/negative). Libera
#       ~9.7 GB di VRAM dedicata al solo T5-XXL.
#   (2) La chiamata `pipe(...)` dentro il training loop usa
#       `prompt_embeds=` + `pooled_prompt_embeds=` invece del prompt
#       testuale. Questo evita di richiedere i text encoder DOPO averli
#       cancellati.
#   (3) `transformer.enable_gradient_checkpointing()` attivo per ridurre
#       ulteriormente l'utilizzo di memoria per le attivazioni.
#   (4) CLI argparse per override rapido di `max_train_steps`, `output_dir`,
#       `slider_name`, `target/positive/negative_prompt`, `rank`, ecc.
#
# Uso tipico (smoke test 50 step):
#   python train_flux_slider.py \
#       --max_train_steps 50 \
#       --output_dir ../outputs/smoke_test \
#       --slider_name smoke_smile
#
# Uso tipico (training reale 500 step):
#   python train_flux_slider.py \
#       --max_train_steps 500 \
#       --output_dir ../outputs/smile_man_flux_v1_rank16_xattn \
#       --slider_name smile_man_flux_v1 \
#       --rank 16 --alpha 1 --train_method xattn
#
# =============================================================================

import os
import sys
import gc
import copy
import math
import time
import argparse
from contextlib import ExitStack
from pathlib import Path

# -----------------------------------------------------------------------------
# CLI args PRIMA di toccare CUDA per poter cambiare CUDA_VISIBLE_DEVICES
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Flux Concept Slider training (MIG-friendly)")
    # modello + training
    p.add_argument("--pretrained_model_name_or_path",
                   default="black-forest-labs/FLUX.1-dev",
                   help="Repo HF o path locale al modello Flux")
    p.add_argument("--max_train_steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--train_method", default="xattn",
                   choices=["xattn", "noxattn", "full"])
    p.add_argument("--num_sliders", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--lr_scheduler", default="constant")
    p.add_argument("--eta", type=float, default=2.0,
                   help="Peso del boost nel gt Eq.7 Concept Sliders")
    # prompt (single-triple CLI, usato solo se --prompts_yaml NON e' fornito)
    p.add_argument("--target_prompt", default="picture of a person")
    p.add_argument("--positive_prompt", default="photo of a person, smiling, happy")
    p.add_argument("--negative_prompt", default="photo of a person, frowning")
    p.add_argument("--slider_name", default="person-smiling")
    # prompt YAML (SDXL-compatible: lista di entries con target/positive/
    # unconditional/neutral/guidance_scale). Se fornito, override dei
    # tre prompt CLI. Abilita multi-entry sampling + preservation via
    # Eq.7 degeneration (v3_yaml_trick style: woman entries con
    # target==positive==unconditional==neutral collapsano la loss in
    # MSE(LoRA_on, LoRA_off) = preservation pura).
    p.add_argument("--prompts_yaml", default=None,
                   help="Path a YAML con lista di entries prompt. Override "
                        "di --target/positive/negative_prompt. Abilita "
                        "multi-prompt sampling stile SDXL train.py.")
    # tecnico
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--bsz", type=int, default=1)
    # output
    p.add_argument("--output_dir", default="flux/trained_sliders/sliders/flux_slider_default")
    p.add_argument("--save_every", type=int, default=0,
                   help="Salva checkpoint intermedio ogni N step; 0 = solo alla fine")
    p.add_argument("--seed", type=int, default=None)
    # flag tecnici
    p.add_argument("--cuda_device", default="0",
                   help="CUDA_VISIBLE_DEVICES (default '0' = prima GPU MIG slice)")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Disabilita gradient checkpointing (solo se hai molta VRAM)")
    return p.parse_args()


args = parse_args()

# -----------------------------------------------------------------------------
# CUDA setup PRIMA di import torch
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import torch
import numpy as np

# -----------------------------------------------------------------------------
# Patch compat: torch 2.4 <-> diffusers 0.35+ (scaled_dot_product_attention)
# -----------------------------------------------------------------------------
# diffusers >= 0.35 passa enable_gqa=False a F.scaled_dot_product_attention,
# ma quel kwarg esiste solo da torch >= 2.5. Flux usa MHA pura (24 head,
# nessuna condivisione KV), quindi strippare il kwarg e' semanticamente
# equivalente al default. Rimuovere queste righe se/quando si upgrada a
# torch >= 2.5.
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

# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from transformers import CLIPTextModel, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from flux.core.custom_flux_pipeline import FluxPipeline
from flux.core.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)

# silenzia verbose
import transformers as _tf
_tf.logging.set_verbosity_error()
import diffusers as _df
_df.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Seed
# -----------------------------------------------------------------------------
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# -----------------------------------------------------------------------------
# Config locale (come Cell 3 del notebook)
# -----------------------------------------------------------------------------
pretrained_model_name_or_path = args.pretrained_model_name_or_path
weight_dtype = torch.bfloat16
device = "cuda:0"
max_train_steps = args.max_train_steps

num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
max_sequence_length = args.max_sequence_length
if "schnell" in pretrained_model_name_or_path:
    num_inference_steps = 4
    guidance_scale = 0
    max_sequence_length = 256

weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29
bsz = args.bsz

lr = args.lr
target_prompt = args.target_prompt
positive_prompt = args.positive_prompt
negative_prompt = args.negative_prompt
slider_name = args.slider_name
alpha = args.alpha
rank = args.rank
train_method = args.train_method
num_sliders = args.num_sliders
eta = args.eta
height = args.height
width = args.width

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print(f"[config] model         = {pretrained_model_name_or_path}")
print(f"[config] max_steps     = {max_train_steps}")
print(f"[config] slider_name   = {slider_name}")
print(f"[config] rank/alpha    = {rank}/{alpha}")
print(f"[config] train_method  = {train_method}")
print(f"[config] output_dir    = {os.path.abspath(output_dir)}")

# -----------------------------------------------------------------------------
# Prompt entries: carica da YAML (SDXL-compatible) o costruisci 1-entry da CLI
# -----------------------------------------------------------------------------
# Formato YAML atteso (lista di dict, SDXL-compatible):
#   - target: "..."
#     positive: "..."
#     unconditional: "..."   (<- usato come "negative" in Eq.7 Flux)
#     neutral: "..."         (<- default: uguale a target)
#     guidance_scale: 2.0    (<- usato come eta per questa entry)
#
# Per entries di preservation: target == positive == unconditional ==
# neutral -> (positive - unconditional) = 0 -> gt = neutral_pred
# -> loss = MSE(LoRA_on, LoRA_off) = preservation pura.
# -----------------------------------------------------------------------------
if args.prompts_yaml:
    import yaml
    with open(args.prompts_yaml, "r") as f:
        _yaml_data = yaml.safe_load(f)
    if isinstance(_yaml_data, dict) and "prompts" in _yaml_data:
        prompt_entries = _yaml_data["prompts"]
    else:
        prompt_entries = _yaml_data
    assert isinstance(prompt_entries, list) and len(prompt_entries) > 0, \
        f"[prompts] YAML {args.prompts_yaml} vuoto o non una lista"
    # Validazione entries
    for i, e in enumerate(prompt_entries):
        for k in ("target", "positive", "unconditional"):
            assert k in e, f"[prompts] entry {i} manca campo '{k}'"
        e.setdefault("neutral", e["target"])
        e.setdefault("guidance_scale", args.eta)
    print(f"[prompts] caricate {len(prompt_entries)} entries da "
          f"{args.prompts_yaml}")
else:
    prompt_entries = [{
        "target": target_prompt,
        "positive": positive_prompt,
        "unconditional": negative_prompt,
        "neutral": target_prompt,
        "guidance_scale": eta,
    }]
    print(f"[prompts] 1 entry singola da CLI (no --prompts_yaml)")

# -----------------------------------------------------------------------------
# Helpers presi dal notebook Cell 4
# -----------------------------------------------------------------------------
def flush():
    torch.cuda.empty_cache()
    gc.collect()

def mem_report(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[mem {tag}] allocated={alloc:.2f} GB  reserved={reserved:.2f} GB")

def import_model_class_from_model_name_or_path(path, subfolder="text_encoder"):
    cfg = PretrainedConfig.from_pretrained(path, subfolder=subfolder,
                                           local_files_only=True)
    cls = cfg.architectures[0]
    if cls == "CLIPTextModel":
        return CLIPTextModel
    elif cls == "T5EncoderModel":
        return T5EncoderModel
    raise ValueError(f"Text encoder class {cls} not supported")

def load_text_encoders(path, cls_one, cls_two, dtype):
    te1 = cls_one.from_pretrained(path, subfolder="text_encoder",
                                  torch_dtype=dtype, device_map=device,
                                  local_files_only=True)
    te2 = cls_two.from_pretrained(path, subfolder="text_encoder_2",
                                  torch_dtype=dtype, device_map=device,
                                  local_files_only=True)
    return te1, te2

def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    return tokenizer(prompt, padding="max_length", max_length=max_sequence_length,
                     truncation=True, return_length=False,
                     return_overflowing_tokens=False, return_tensors="pt").input_ids

def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length=512,
                           prompt=None, num_images_per_prompt=1, device=None,
                           text_input_ids=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_input_ids = tokenizer(prompt, padding="max_length",
                                   max_length=max_sequence_length, truncation=True,
                                   return_length=False, return_overflowing_tokens=False,
                                   return_tensors="pt").input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds

def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None,
                             text_input_ids=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_input_ids = tokenizer(prompt, padding="max_length", max_length=77,
                                   truncation=True, return_overflowing_tokens=False,
                                   return_length=False, return_tensors="pt").input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds

def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length,
                  device=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    pooled = _encode_prompt_with_clip(text_encoders[0], tokenizers[0], prompt,
                                      device=device or text_encoders[0].device,
                                      num_images_per_prompt=num_images_per_prompt)
    embeds = _encode_prompt_with_t5(text_encoders[1], tokenizers[1],
                                    max_sequence_length=max_sequence_length,
                                    prompt=prompt,
                                    num_images_per_prompt=num_images_per_prompt,
                                    device=device or text_encoders[1].device)
    text_ids = torch.zeros(batch_size, embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
    return embeds, pooled, text_ids

def compute_text_embeddings(prompt, text_encoders, tokenizers):
    dev = text_encoders[0].device
    with torch.no_grad():
        e, p, t = encode_prompt(text_encoders, tokenizers, prompt,
                                max_sequence_length=max_sequence_length, device=dev)
    return e.to(dev), p.to(dev), t.to(dev)

def get_sigmas(timesteps, scheduler_copy, n_dim=4, device="cuda:0", dtype=torch.bfloat16):
    sigmas = scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

# -----------------------------------------------------------------------------
# STEP A — Load tokenizers + text encoders + scheduler + VAE + transformer
# -----------------------------------------------------------------------------
print("\n=== [A] Loading tokenizers + text encoders + scheduler ===")
tokenizer_one = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer",
    local_files_only=True)
tokenizer_two = T5TokenizerFast.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer_2",
    local_files_only=True)

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler",
    local_files_only=True)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

te_cls_one = import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path)
te_cls_two = import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path, subfolder="text_encoder_2")
text_encoder_one, text_encoder_two = load_text_encoders(
    pretrained_model_name_or_path, te_cls_one, te_cls_two, weight_dtype)

print("\n=== [B] Loading VAE + transformer ===")
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae",
    torch_dtype=weight_dtype, device_map="auto",
    local_files_only=True)
transformer = FluxTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="transformer",
    torch_dtype=weight_dtype,
    local_files_only=True)

transformer.requires_grad_(False)
vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

vae.to(device)
transformer.to(device)
text_encoder_one.to(device)
text_encoder_two.to(device)

if not args.no_gradient_checkpointing:
    transformer.enable_gradient_checkpointing()
    print("[grad-ckpt] abilitato su transformer")

mem_report("dopo load full stack (T5 ancora in VRAM)")

# -----------------------------------------------------------------------------
# STEP C — Pre-compute embeddings per tutte le entries, poi FREE dei text encoder
# -----------------------------------------------------------------------------
# Cache per-string evita di ri-encodare lo stesso prompt quando compare
# in piu' entries (tipico: target == neutral, oppure woman-preservation
# dove target == positive == unconditional == neutral).
print("\n=== [C] Pre-computing text embeddings per tutte le entries ===")
_emb_cache = {}  # prompt_str -> (embeds, pooled, text_ids)

def _encode_one(prompt_str):
    """Encoder con cache: ritorna (prompt_embeds, pooled, text_ids) per il prompt."""
    if prompt_str in _emb_cache:
        return _emb_cache[prompt_str]
    with torch.no_grad():
        e, p, t = compute_text_embeddings(
            [prompt_str],
            [text_encoder_one, text_encoder_two],
            [tokenizer_one, tokenizer_two],
        )
    _emb_cache[prompt_str] = (e, p, t)
    return e, p, t

precomputed_entries = []
for i, entry in enumerate(prompt_entries):
    t_str = entry["target"]
    p_str = entry["positive"]
    u_str = entry["unconditional"]
    n_str = entry["neutral"]
    eta_i = float(entry["guidance_scale"])
    t_e, t_p, t_t = _encode_one(t_str)
    p_e, p_p, p_t = _encode_one(p_str)
    u_e, u_p, u_t = _encode_one(u_str)
    n_e, n_p, n_t = _encode_one(n_str)
    precomputed_entries.append({
        "idx": i,
        "target_str": t_str, "positive_str": p_str,
        "unconditional_str": u_str, "neutral_str": n_str,
        "target_emb": t_e, "target_pool": t_p, "target_txtid": t_t,
        "positive_emb": p_e, "positive_pool": p_p, "positive_txtid": p_t,
        "unconditional_emb": u_e, "unconditional_pool": u_p, "unconditional_txtid": u_t,
        "neutral_emb": n_e, "neutral_pool": n_p, "neutral_txtid": n_t,
        "eta": eta_i,
    })

_n_uniq = len(_emb_cache)
_sample_shape = next(iter(_emb_cache.values()))[0].shape
print(f"  entries precomputed   = {len(precomputed_entries)}")
print(f"  prompt strings unici  = {_n_uniq} (dedup via cache)")
print(f"  shape prompt_embeds   = {_sample_shape}")
del _emb_cache  # liberiamo il dict, i tensori restano referenziati dalle entries

# -----------------------------------------------------------------------------
# STEP D — T5 OFFLOAD PATCH: cancella text encoders dalla GPU
# -----------------------------------------------------------------------------
print("\n=== [D] T5 OFFLOAD: cancello text_encoder_one (CLIP-L) + text_encoder_two (T5-XXL) ===")
text_encoder_one.to("cpu")
text_encoder_two.to("cpu")
del text_encoder_one
del text_encoder_two
del tokenizer_one
del tokenizer_two
flush()
mem_report("dopo T5 offload (T5+CLIP-L fuori)")

# -----------------------------------------------------------------------------
# STEP E — Setup LoRA + optimizer + pipe
# -----------------------------------------------------------------------------
print("\n=== [E] Setup LoRA + optimizer + FluxPipeline ===")
networks = {}
params = []
modules = DEFAULT_TARGET_REPLACE + UNET_TARGET_REPLACE_MODULE_CONV
for i in range(num_sliders):
    networks[i] = LoRANetwork(
        transformer, rank=rank, multiplier=1.0, alpha=alpha,
        train_method=train_method,
    ).to(device, dtype=weight_dtype)
    params.extend(networks[i].prepare_optimizer_params())

optimizer = AdamW(params, lr=lr)
optimizer.zero_grad()

# FluxPipeline con text_encoder=None (T5/CLIP-L cancellati)
pipe = FluxPipeline(
    noise_scheduler, vae, None,  # text_encoder (CLIP-L) = None
    None,                         # tokenizer_one = None
    None,                         # text_encoder_2 (T5) = None
    None,                         # tokenizer_2 = None
    transformer,
)
pipe.set_progress_bar_config(disable=True)
mem_report("dopo LoRA + pipe init")

lr_scheduler = get_scheduler(
    args.lr_scheduler, optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=max_train_steps,
    num_cycles=1, power=1.0,
)

# -----------------------------------------------------------------------------
# STEP F — Training loop (con embeddings pre-computati passati a pipe)
# -----------------------------------------------------------------------------
print("\n=== [F] Training loop ===")
progress_bar = tqdm(range(0, max_train_steps), desc="Steps")
losses = {}
t_start = time.time()

entry_sample_count = [0] * len(precomputed_entries)  # telemetria entry-sampling

for epoch in range(max_train_steps):
    # Sample entry uniformemente a random dalla lista
    entry_idx = int(np.random.randint(len(precomputed_entries)))
    entry = precomputed_entries[entry_idx]
    entry_sample_count[entry_idx] += 1

    # Embeddings per questa entry
    target_prompt_embeds         = entry["target_emb"]
    target_pooled_prompt_embeds  = entry["target_pool"]
    target_text_ids              = entry["target_txtid"]
    entry_eta                    = entry["eta"]

    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme, batch_size=bsz,
        logit_mean=logit_mean, logit_std=logit_std, mode_scale=mode_scale,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)

    timestep_to_infer = (
        indices[0] * (num_inference_steps / noise_scheduler_copy.config.num_train_timesteps)
    ).long().item()

    with torch.no_grad():
        # Denoise usando gli embeddings del target di questa entry
        packed_noisy_model_input = pipe(
            prompt=None,
            prompt_embeds=target_prompt_embeds,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            height=height, width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=bsz,
            generator=None,
            from_timestep=0,
            till_timestep=timestep_to_infer,
            output_type="latent",
        )
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
        if epoch == 0:
            model_input = FluxPipeline._unpack_latents(
                packed_noisy_model_input,
                height=height, width=width,
                vae_scale_factor=vae_scale_factor,
            )

    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0], model_input.shape[2], model_input.shape[3],
        device, weight_dtype,
    )

    # guidance embeds (Flux.1-dev usa distilled guidance)
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=device).expand(model_input.shape[0])
    else:
        guidance = None

    _unpack_h = int(model_input.shape[2] * vae_scale_factor / 2)
    _unpack_w = int(model_input.shape[3] * vae_scale_factor / 2)

    # --- forward CON grad + LoRA attiva -> target_pred_with_slider ---
    with ExitStack() as stack:
        for net in networks:
            stack.enter_context(networks[net])
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=target_pooled_prompt_embeds,
            encoder_hidden_states=target_prompt_embeds,
            txt_ids=target_text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
    model_pred = FluxPipeline._unpack_latents(
        model_pred, height=_unpack_h, width=_unpack_w,
        vae_scale_factor=vae_scale_factor,
    )

    # --- forwards SENZA grad, LoRA OFF, DEDUPLICATI per string equality ---
    # Per v3_yaml_trick le woman entries hanno target==positive==unconditional
    # ==neutral, quindi 1 unico prompt -> 1 solo forward (invece di 4).
    # Per man entries invece ci sono 3 prompt distinti (target==neutral),
    # quindi 3 forwards. Dedup via dict keyed sulla stringa del prompt.
    with torch.no_grad():
        _off_cache = {}  # prompt_str -> pred (unpacked)

        def _forward_off(pstr, pemb, ppool, ptxtid):
            if pstr in _off_cache:
                return _off_cache[pstr]
            pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000, guidance=guidance,
                pooled_projections=ppool,
                encoder_hidden_states=pemb,
                txt_ids=ptxtid, img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            pred = FluxPipeline._unpack_latents(
                pred, height=_unpack_h, width=_unpack_w,
                vae_scale_factor=vae_scale_factor,
            )
            _off_cache[pstr] = pred
            return pred

        # 4 forward slots (con dedup: spesso sono meno forward effettivi)
        target_pred = _forward_off(
            entry["target_str"],
            entry["target_emb"], entry["target_pool"], entry["target_txtid"],
        )
        positive_pred = _forward_off(
            entry["positive_str"],
            entry["positive_emb"], entry["positive_pool"], entry["positive_txtid"],
        )
        negative_pred = _forward_off(
            entry["unconditional_str"],
            entry["unconditional_emb"], entry["unconditional_pool"], entry["unconditional_txtid"],
        )
        neutral_pred = _forward_off(
            entry["neutral_str"],
            entry["neutral_emb"], entry["neutral_pool"], entry["neutral_txtid"],
        )

        # Eq.7 Concept Sliders con neutral (non target) come base-term.
        # Quando positive == unconditional (woman-preservation case), il
        # secondo addendo e' zero -> gt = neutral_pred = target_pred (se
        # neutral == target) -> loss = MSE(LoRA_on, LoRA_off) = preservation.
        gt_pred = neutral_pred + entry_eta * (positive_pred - negative_pred)
        gt_pred = (gt_pred / gt_pred.norm()) * positive_pred.norm()

    # loss concept
    concept_loss = torch.mean(
        ((model_pred.float() - gt_pred.float()) ** 2).reshape(gt_pred.shape[0], -1),
        1,
    ).mean()
    concept_loss.backward()
    losses["concept"] = losses.get("concept", []) + [concept_loss.item()]

    logs = {"concept_loss": losses["concept"][-1],
            "lr": lr_scheduler.get_last_lr()[0],
            "entry": entry_idx,
            "uniq_fwd": len(_off_cache),
            "eta": entry_eta}
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
    progress_bar.set_postfix(**logs)

    # mem report ai primi step + periodicamente
    if epoch in (0, 1, 5, 20) or (epoch > 0 and epoch % 100 == 0):
        mem_report(f"step {epoch}")
        if epoch > 0 and epoch % 100 == 0:
            print(f"[entry-sampling] dopo {epoch} step: "
                  f"{entry_sample_count}")

    # checkpoint intermedio
    if args.save_every > 0 and (epoch + 1) % args.save_every == 0 and (epoch + 1) < max_train_steps:
        save_path = Path(output_dir) / f"flux-{slider_name}_step{epoch+1}"
        save_path.mkdir(parents=True, exist_ok=True)
        for i in range(num_sliders):
            networks[i].save_weights(
                str(save_path / f"slider_{i}.pt"), dtype=weight_dtype,
            )
        print(f"[ckpt] intermedio salvato in {save_path}")

t_end = time.time()
print(f"\n=== Training Done in {(t_end - t_start)/60:.1f} min "
      f"({(t_end - t_start)/max_train_steps:.1f} s/step) ===")

print("\n[entry-sampling] distribuzione finale:")
for i, (cnt, e) in enumerate(zip(entry_sample_count, precomputed_entries)):
    pct = 100.0 * cnt / max_train_steps
    is_pres = (e["target_str"] == e["positive_str"]
               == e["unconditional_str"] == e["neutral_str"])
    tag = "PRES" if is_pres else "ENH "
    print(f"  [{i:2d}] {tag} count={cnt:4d} ({pct:5.1f}%)  "
          f"target={e['target_str'][:50]}")

# -----------------------------------------------------------------------------
# STEP G — Save finale
# -----------------------------------------------------------------------------
save_name = f"flux-{slider_name}"
save_path = Path(output_dir) / save_name
save_path.mkdir(parents=True, exist_ok=True)
print(f"\n=== [G] Saving LoRA weights -> {save_path} ===")
for i in range(num_sliders):
    out_file = save_path / f"slider_{i}.pt"
    networks[i].save_weights(str(out_file), dtype=weight_dtype)
    print(f"  saved {out_file}")

# salva anche loss history per plot offline
loss_file = Path(output_dir) / f"{slider_name}_losses.npy"
np.save(loss_file, np.array(losses["concept"]))
print(f"  saved losses -> {loss_file}")

flush()
mem_report("fine training")
print("\n=== DONE ===")
