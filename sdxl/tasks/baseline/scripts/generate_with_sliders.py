#!/usr/bin/env python
"""
Text-to-Image Generation with Concept Sliders (SDXL).

Self-contained copy of exp_generation/generate_with_sliders.py.

Modifications vs upstream (all minimal, none change the inference math):

    1. sys.path insertion — points at the sibling scripts/ folder so this
       script imports the local copies of `lora.py` / `train_util.py` rather
       than `trainscripts/textsliders/lora.py`. Keeps this training folder
       self-contained.

    2. Weight loader — upstream uses `torch.load(args.slider)` which only
       understands pickle format. Our training script writes
       `.safetensors` by default, so we sniff the file extension and
       dispatch to `safetensors.torch.load_file` when appropriate. Pickle
       files (.pt, .pth) still work.

    3. Default `--save_path` — defaults to a path inside
       `training_local_concept_sliders/SDXL_train/outputs/` instead of
       `exp_generation/outputs/`.

The monkey-patched `StableDiffusionXLPipeline.__call__` (see `call` below)
and the main generation loop are byte-identical to upstream. In particular:
  - `start_noise=700` means the slider only fires at timesteps <= 700 out
    of 1000 (upstream default), i.e. after the image structure has already
    been laid down — the slider edits appearance, not layout.
  - `scale=0` means the LoRA has no effect (identical to the base model's
    output for the same seed). `scale > 0` amplifies the learned direction;
    `scale < 0` reverses it.
"""

import sys
import os

import torch
from PIL import Image
import argparse
import random
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines import StableDiffusionXLPipeline

from sdxl.core.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def load_slider_state_dict(path: str):
    """Load a LoRA slider state dict, auto-detecting .safetensors vs pickle.

    Upstream training (network.save_weights) writes .safetensors when the
    file extension is .safetensors, and falls back to torch.save otherwise.
    Our training config uses .safetensors, so we need the safetensors
    reader here; older .pt files still work for backwards compatibility.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path)


@torch.no_grad()
def call(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    network=None,
    start_noise=None,
    scale=None,
    unet=None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    self.check_inputs(
        prompt, prompt_2, height, width, callback_steps,
        negative_prompt, negative_prompt_2,
        prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds,
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt, prompt_2=prompt_2, device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    num_channels_latents = unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents, height, width,
        prompt_embeds.dtype, device, generator, latents,
    )

    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    add_text_embeds = pooled_prompt_embeds
    # diffusers >=0.21 requires `text_encoder_projection_dim` to be passed
    # explicitly to `_get_add_time_ids`; the upstream inference script was
    # written against an older signature where this was inferred.
    text_encoder_projection_dim = (
        int(pooled_prompt_embeds.shape[-1])
        if getattr(self, "text_encoder_2", None) is None
        else self.text_encoder_2.config.projection_dim
    )
    add_time_ids = self._get_add_time_ids(
        original_size, crops_coords_top_left, target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size, negative_crops_coords_top_left,
            negative_target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(
        batch_size * num_images_per_prompt, 1
    )

    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )

    if (
        denoising_end is not None
        and isinstance(denoising_end, float)
        and 0 < denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                self.scheduler.config.num_train_timesteps
                - (denoising_end * self.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(
            list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
        )
        timesteps = timesteps[:num_inference_steps]

    latents = latents.to(unet.dtype)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if t > start_noise:
                network.set_lora_slider(scale=0)
            else:
                network.set_lora_slider(scale=scale)

            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )

            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            with network:
                noise_pred = unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps
                and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type)

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for image generation.")
    parser.add_argument("--slider", type=str, required=True,
                        help="Path to the trained LoRA slider (.safetensors or .pt).")
    parser.add_argument("--save_path", type=str,
                        default="sdxl/tasks/baseline/outputs/_generations",
                        help="Where to save the per-scale images and the comparison grid.")
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 1, 2, 3],
                        help="Slider scales to sweep over. scale=0 is the baseline "
                             "(LoRA off), positive amplifies the learned direction, "
                             "negative reverses it. Upstream default: [0, 1, 2, 3]. "
                             "For bidirectional tests, pass e.g. -2 -1 0 1 2.")
    parser.add_argument("--start_noise", type=int, default=700,
                        help="Slider fires only at timesteps <= start_noise "
                             "(out of 1000). Default 700 = slider edits the last "
                             "~70%% of denoising, preserving image layout.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducibility. Same seed across all scales "
                             "= identical layout, only slider effect changes.")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank used at training time. Must match.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LoRA alpha used at training time. Must match.")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of DDIM inference steps.")
    args = parser.parse_args()

    device = "cuda"
    StableDiffusionXLPipeline.__call__ = call

    seed = args.seed if args.seed is not None else random.randint(0, 2**15)
    print(f"[INFO] Prompt: {args.prompt}")
    print(f"[INFO] Slider: {args.slider}")
    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Scales: {args.scales}")
    print(f"[INFO] start_noise: {args.start_noise}")

    # --- Load pipeline ---
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to(device)
    unet = pipe.unet

    # --- Load LoRA slider ---
    # Extend target modules to include conv layers (byte-identical to upstream).
    # NOTE: `modules` is computed but not actually consumed by LoRANetwork,
    # which hardcodes DEFAULT_TARGET_REPLACE. Kept for fidelity.
    modules = DEFAULT_TARGET_REPLACE
    modules += UNET_TARGET_REPLACE_MODULE_CONV

    network = LoRANetwork(
        unet,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.alpha,
        train_method="noxattn",
    ).to(device, dtype=torch.float16)
    network.load_state_dict(load_slider_state_dict(args.slider))

    # --- Generate at each scale ---
    os.makedirs(args.save_path, exist_ok=True)
    image_list = []
    for scale in args.scales:
        generator = torch.manual_seed(seed)
        img = pipe(
            args.prompt,
            num_images_per_prompt=1,
            num_inference_steps=args.steps,
            generator=generator,
            network=network,
            start_noise=args.start_noise,
            scale=scale,
            unet=unet,
        ).images[0]
        image_list.append(img)
        img.save(os.path.join(args.save_path, f"scale_{scale}.png"))

    # --- Save comparison grid ---
    fig, ax = plt.subplots(1, len(image_list), figsize=(20, 4))
    if len(image_list) == 1:
        ax = [ax]
    for i, a in enumerate(ax):
        a.imshow(image_list[i])
        a.set_title(f"{args.scales[i]}", fontsize=15)
        a.axis("off")
    slider_name = os.path.splitext(os.path.basename(args.slider))[0]
    plt.suptitle(f"{slider_name}  |  seed={seed}  |  {args.prompt}", fontsize=14)
    fig.savefig(os.path.join(args.save_path, "grid.png"), bbox_inches="tight")
    plt.close(fig)

    del unet, network, pipe
    flush()
    print(f"[INFO] Done. Images saved to {args.save_path}")
