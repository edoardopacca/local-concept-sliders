# Modified training script for Concept Sliders with an explicit
# preservation term on a second prompt set.
#
# Relative to scripts/train.py (byte-identical to upstream
# trainscripts/textsliders/train_lora_xl.py apart from sys.path), this
# script:
#
#   1. Loads TWO prompt YAMLs:
#        - prompts_file         → man entries, standard "enhance" loss
#        - prompts_file_woman   → woman entries, preservation loss only
#   2. At every iteration, samples one entry from each list.
#   3. Runs the upstream enhance pass on the man entry exactly as before
#      (denoise trajectory + positive/neutral/unconditional + target_with_LoRA).
#   4. ADDITIONALLY runs a preservation pass on the woman entry:
#        a. denoise trajectory with network ON, using woman target prompt
#           (no-grad — the trajectory only establishes a sensible
#            intermediate latent at which to compare eps with/without LoRA)
#        b. eps_with_LoRA    = UNet + LoRA on this woman state (WITH grad)
#        c. eps_without_LoRA = UNet with LoRA off on the same state (no grad)
#        d. loss_preservation = MSE(eps_with_LoRA, eps_without_LoRA.detach())
#   5. Combines the two losses:
#        loss_total = loss_enhance_on_man + lambda_pres * loss_preservation_on_woman
#      and backprops through loss_total.
#
# The intent is to force the LoRA to leave "woman"-activated UNet
# features unchanged at every timestep, while still moving the "man"
# activations along the smile direction. This addresses the CLIP
# attribute-leak failure mode observed on v2_guidance4 (smile leaks onto
# the woman at scales ≥ 2) by penalizing the exact quantity that causes
# the leak — nonzero ΔW on woman activations — instead of relying on
# prompt engineering (which failed empirically) or spatial masks
# (rejected by design).
#
# Cost per iteration: ~2× vs upstream train.py. The preservation branch
# adds one extra denoise trajectory, one extra no-grad forward pass, and
# one extra with-grad forward pass on the UNet. lambda_pres is a CLI
# knob (--lambda_pres, default 1.0) so we can sweep it without touching
# YAML.

from typing import List, Optional
import argparse
import ast
import os
import sys
from pathlib import Path
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sdxl.core.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from sdxl.core import train_util
from sdxl.core import model_util
from sdxl.core import prompt_util
from sdxl.core.prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
from sdxl.core import debug_util
from sdxl.core import config_util
from sdxl.core.config_util import RootConfig

import wandb

NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def build_prompt_pairs(prompts, tokenizers, text_encoders, criteria, cache):
    """Encode a list of PromptSettings and build PromptEmbedsPair objects.

    The cache is shared across calls so that repeated prompts (e.g. the
    same empty unconditional "") are not re-encoded.
    """
    pairs: list[PromptEmbedsPair] = []
    with torch.no_grad():
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] is None:
                    tex_embs, pool_embs = train_util.encode_prompts_xl(
                        tokenizers,
                        text_encoders,
                        [prompt],
                        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                    )
                    cache[prompt] = PromptEmbedsXL(tex_embs, pool_embs)

            pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )
    return pairs


def train(
    config: RootConfig,
    prompts_man: list[PromptSettings],
    prompts_woman: list[PromptSettings],
    lambda_pres: float,
    device,
):
    metadata = {
        "prompts_man": ",".join([p.json() for p in prompts_man]),
        "prompts_woman": ",".join([p.json() for p in prompts_woman]),
        "lambda_pres": lambda_pres,
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    optimizer = optimizer_module(
        network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs
    )
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Man prompts (enhance):")
    for settings in prompts_man:
        print(settings)
    print("Woman prompts (preservation):")
    for settings in prompts_woman:
        print(settings)

    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    man_prompt_pairs = build_prompt_pairs(
        prompts_man, tokenizers, text_encoders, criteria, cache
    )
    woman_prompt_pairs = build_prompt_pairs(
        prompts_woman, tokenizers, text_encoders, criteria, cache
    )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    loss = None

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )

            optimizer.zero_grad()

            man_pair: PromptEmbedsPair = man_prompt_pairs[
                torch.randint(0, len(man_prompt_pairs), (1,)).item()
            ]
            woman_pair: PromptEmbedsPair = woman_prompt_pairs[
                torch.randint(0, len(woman_prompt_pairs), (1,)).item()
            ]

            # shared timesteps: same t sampled for both branches
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()

            # ================= MAN BRANCH (upstream enhance loss) =================
            height_m, width_m = man_pair.resolution, man_pair.resolution
            if man_pair.dynamic_resolution:
                height_m, width_m = train_util.get_random_resolution_in_bucket(
                    man_pair.resolution
                )

            latents_m = train_util.get_initial_latents(
                noise_scheduler, man_pair.batch_size, height_m, width_m, 1
            ).to(device, dtype=weight_dtype)

            add_time_ids_m = train_util.get_add_time_ids(
                height_m,
                width_m,
                dynamic_crops=man_pair.dynamic_crops,
                dtype=weight_dtype,
            ).to(device, dtype=weight_dtype)

            with network:
                denoised_latents_m = train_util.diffusion_xl(
                    unet,
                    noise_scheduler,
                    latents_m,
                    text_embeddings=train_util.concat_embeddings(
                        man_pair.unconditional.text_embeds,
                        man_pair.target.text_embeds,
                        man_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        man_pair.unconditional.pooled_embeds,
                        man_pair.target.pooled_embeds,
                        man_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids_m, add_time_ids_m, man_pair.batch_size
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            positive_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_m,
                text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.text_embeds,
                    man_pair.positive.text_embeds,
                    man_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.pooled_embeds,
                    man_pair.positive.pooled_embeds,
                    man_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids_m, add_time_ids_m, man_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
            neutral_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_m,
                text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.text_embeds,
                    man_pair.neutral.text_embeds,
                    man_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.pooled_embeds,
                    man_pair.neutral.pooled_embeds,
                    man_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids_m, add_time_ids_m, man_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
            unconditional_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_m,
                text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.text_embeds,
                    man_pair.unconditional.text_embeds,
                    man_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.pooled_embeds,
                    man_pair.unconditional.pooled_embeds,
                    man_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids_m, add_time_ids_m, man_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            # ================= WOMAN BRANCH (preservation prep, no-grad) =================
            height_w, width_w = woman_pair.resolution, woman_pair.resolution
            if woman_pair.dynamic_resolution:
                height_w, width_w = train_util.get_random_resolution_in_bucket(
                    woman_pair.resolution
                )

            # fresh noise for woman trajectory (independent of man)
            latents_w = train_util.get_initial_latents(
                noise_scheduler, woman_pair.batch_size, height_w, width_w, 1
            ).to(device, dtype=weight_dtype)

            add_time_ids_w = train_util.get_add_time_ids(
                height_w,
                width_w,
                dynamic_crops=woman_pair.dynamic_crops,
                dtype=weight_dtype,
            ).to(device, dtype=weight_dtype)

            # denoise trajectory uses LoRA (ON) — same setup as upstream
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )
            with network:
                denoised_latents_w = train_util.diffusion_xl(
                    unet,
                    noise_scheduler,
                    latents_w,
                    text_embeddings=train_util.concat_embeddings(
                        woman_pair.unconditional.text_embeds,
                        woman_pair.target.text_embeds,
                        woman_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        woman_pair.unconditional.pooled_embeds,
                        woman_pair.target.pooled_embeds,
                        woman_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids_w, add_time_ids_w, woman_pair.batch_size
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            noise_scheduler.set_timesteps(1000)

            # teacher: woman prediction WITHOUT LoRA (no grad, detached)
            woman_eps_nolora = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_w,
                text_embeddings=train_util.concat_embeddings(
                    woman_pair.unconditional.text_embeds,
                    woman_pair.target.text_embeds,
                    woman_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    woman_pair.unconditional.pooled_embeds,
                    woman_pair.target.pooled_embeds,
                    woman_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids_w, add_time_ids_w, woman_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

        # ================= WITH-GRAD FORWARDS =================

        # man: target prediction WITH LoRA (gradient flows here)
        with network:
            target_latents_m = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_m,
                text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.text_embeds,
                    man_pair.target.text_embeds,
                    man_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    man_pair.unconditional.pooled_embeds,
                    man_pair.target.pooled_embeds,
                    man_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids_m, add_time_ids_m, man_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

        # woman: target prediction WITH LoRA (gradient flows here)
        with network:
            woman_eps_lora = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_w,
                text_embeddings=train_util.concat_embeddings(
                    woman_pair.unconditional.text_embeds,
                    woman_pair.target.text_embeds,
                    woman_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    woman_pair.unconditional.pooled_embeds,
                    woman_pair.target.pooled_embeds,
                    woman_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids_w, add_time_ids_w, woman_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False
        woman_eps_nolora.requires_grad = False

        # ================= LOSS =================
        loss_man = man_pair.loss(
            target_latents=target_latents_m,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            unconditional_latents=unconditional_latents,
        )
        loss_woman = F.mse_loss(woman_eps_lora, woman_eps_nolora.detach())
        loss = loss_man + lambda_pres * loss_woman

        pbar.set_description(
            f"L*1k: {loss.item()*1000:.3f} "
            f"(man: {loss_man.item()*1000:.3f}, "
            f"pres: {loss_woman.item()*1000:.3f}, "
            f"lam: {lambda_pres})"
        )
        if config.logging.use_wandb:
            wandb.log(
                {
                    "loss": loss,
                    "loss_man": loss_man,
                    "loss_pres": loss_woman,
                    "iteration": i,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents_m,
            woman_eps_lora,
            woman_eps_nolora,
            latents_m,
            latents_w,
            denoised_latents_m,
            denoised_latents_w,
        )
        flush()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()
    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(",")
        attributes = [a.strip() for a in attributes]

    if args.prompts_file is not None:
        config.prompts_file = args.prompts_file
    if args.alpha is not None:
        config.network.alpha = args.alpha
    if args.rank is not None:
        config.network.rank = args.rank
    config.save.name += f"_alpha{config.network.alpha}"
    config.save.name += f"_rank{config.network.rank}"
    config.save.name += f"_{config.network.training_method}"
    config.save.name += f"_lam{args.lambda_pres}"
    config.save.path += f"/{config.save.name}"

    # man (enhance) prompts — from config.prompts_file (config_util's
    # RootConfig recognizes this field).
    prompts_man = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)

    # woman (preservation) prompts — may come from CLI or from a custom
    # field in the YAML. We read the raw YAML to grab `prompts_file_woman`
    # if the CLI flag is not provided.
    if args.prompts_file_woman is not None:
        woman_path = args.prompts_file_woman
    else:
        import yaml as _yaml
        with open(config_file, "r") as f:
            raw_cfg = _yaml.safe_load(f)
        woman_path = raw_cfg.get("prompts_file_woman")
        if woman_path is None:
            raise ValueError(
                "prompts_file_woman must be provided either via CLI "
                "(--prompts_file_woman) or as a top-level key in the config YAML."
            )
    prompts_woman = prompt_util.load_prompts_from_yaml(woman_path, attributes)

    print("Lambda_pres:", args.lambda_pres)
    print("Man prompts:", len(prompts_man))
    print("Woman prompts:", len(prompts_woman))

    device = torch.device(f"cuda:{args.device}")
    train(config, prompts_man, prompts_woman, args.lambda_pres, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Config file for training.")
    parser.add_argument(
        "--prompts_file",
        required=False,
        default=None,
        help="Override man (enhance) prompts file.",
    )
    parser.add_argument(
        "--prompts_file_woman",
        required=False,
        default=None,
        help="Woman (preservation) prompts file. If omitted, read from "
        "`prompts_file_woman` key in the config YAML.",
    )
    parser.add_argument(
        "--lambda_pres",
        type=float,
        required=False,
        default=1.0,
        help="Weight on the preservation loss term.",
    )
    parser.add_argument("--alpha", type=float, required=False, default=None)
    parser.add_argument("--rank", type=int, required=False, default=None)
    parser.add_argument("--device", type=int, required=False, default=0)
    parser.add_argument("--name", type=str, required=False, default=None)
    parser.add_argument("--attributes", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)
