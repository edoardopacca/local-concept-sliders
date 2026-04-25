# shop_concept/flux_real_pipeline.py
# ---------------------------------------------------------------------------
# Derivato da LoRAShop-main/flux_real_pipeline.py.
#
# DIFF RISPETTO ALL'ORIGINALE (vedi CHANGES.md per dettagli):
#   * Import riscritti in relativi (`from .utils`, `from .flux_blocks`).
#   * `__call__` accetta un nuovo kwarg `target_lora_scales` che viene
#     iniettato in `joint_attention_kwargs["target_lora_scales"]`, letto
#     da flux_blocks.TransformerBlock.set_adapter per applicare la scale
#     continua del concept slider per-target.
# ---------------------------------------------------------------------------
# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dataclasses import dataclass
from PIL import Image

from diffusers.utils import BaseOutput

from .utils import get_attr, set_attr_raw
from .flux_blocks import TransformerBlock, SingleTransformerBlock

import matplotlib.pyplot as plt
import math

from scipy.stats import gaussian_kde

import torch.nn.functional as F

import cv2

@dataclass
class PipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray]

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16):

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs):

    if timesteps is not None: # Init from timesteps directly
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None: # Init from sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else: # Init from number of timesteps
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class RealGenerationPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin): # Not all of them may be needed
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, 
                scheduler: FlowMatchEulerDiscreteScheduler,
                vae: AutoencoderKL,
                text_encoder: CLIPTextModel,
                tokenizer: CLIPTokenizer,
                text_encoder_2: T5EncoderModel,
                tokenizer_2: T5TokenizerFast,
                transformer: FluxTransformer2DModel):

        super().__init__()
        
        # Register Modules
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )

        self.default_sample_size = 128

    def get_t5_context(self,
                       prompt: Union[str, List[str]] = None,
                       num_images_per_prompt: int = 1,
                       max_sequence_length: int = 512,
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None):
        
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_input_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt"
        ).input_ids

        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            print(f"WARNING T5 - Input truncated for prompt: {prompt}")

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0].to(dtype=self.text_encoder_2.dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # Duplicate prompt embeds and attention mask
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, torch.argmin(text_input_ids)

    def get_clip_context(self,
                         prompt: Union[str, List[str]],
                         num_images_per_prompt: int = 1,
                         device: Optional[torch.device] = None):
        
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt"
        ).input_ids

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            print(f"WARNING CLIP - Input truncated for prompt: {prompt}")

        # Pooled Embedding for CLIP
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=self.text_encoder.dtype, device=device)

        # Duplicate prompt embeds and attention mask
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(self,
                      prompt: Union[str, List[str]],
                      prompt_2: Union[str, List[str]],
                      device: Optional[torch.device] = None,
                      num_images_per_prompt: int = 1,
                      prompt_embeds: Optional[torch.FloatTensor] = None,
                      pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                      max_sequence_length: int = 512,
                      lora_scale: Optional[float] = None,):
    
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # Pooled Embeddings from CLIP
            pooled_prompt_embeds = self.get_clip_context(
                prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt
            )

            # Sequenced Embeddings from T5
            prompt_embeds, prompt_length = self.get_t5_context(
                prompt=prompt_2, 
                num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
                device=device
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids, prompt_length

    @staticmethod
    def prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, height // 2 * width // 2, num_channels_latents * 4)
        return latents

    @staticmethod
    def unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width //  vae_scale_factor
        
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    def prepare_latents(self,
                        batch_size: int,
                        num_channels_latents: int,
                        height: int,
                        width: int,
                        dtype: torch.dtype,
                        device: torch.device,
                        generator: torch.Generator,
                        latents=None):
        
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self.prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self.prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    
    def register_transformer_blocks(self):
        weight_keys = self.transformer.state_dict().keys()
        transformer_modules = []
        single_transformer_modules = []

        for weight_key in weight_keys:
            module_name = ".".join(weight_key.split(sep=".")[:2])
            if weight_key.startswith("single_transformer_blocks"):
                if module_name not in single_transformer_modules:
                    single_transformer_modules.append(module_name)
            elif weight_key.startswith("transformer_blocks"):
                if module_name not in transformer_modules:
                    transformer_modules.append(module_name)
        
        
        for single_transformer_module in single_transformer_modules:
            orig_module = get_attr(self.transformer, single_transformer_module)
            unit = SingleTransformerBlock(orig_module, single_transformer_module)
            set_attr_raw(self.transformer, single_transformer_module, unit)

        for transformer_module in transformer_modules:
            orig_module = get_attr(self.transformer, transformer_module)
            unit = TransformerBlock(orig_module, transformer_module)
            set_attr_raw(self.transformer, transformer_module, unit)

    def get_temb_for_projections(self, timestep, pooled_projections, guidance=None):
        temb = (
            self.transformer.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.transformer.time_text_embed(timestep, guidance, pooled_projections)
        )

        return temb

    def get_substring_tokens(self, prompt, target_prompt):
            # Tokenize with offsets
            encoding = self.tokenizer_2(prompt, return_offsets_mapping=True)
            tokens = self.tokenizer_2.convert_ids_to_tokens(encoding["input_ids"])
            token_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            # Find character span of the target prompt
            try:
                start_char = prompt.index(target_prompt)
            except ValueError:
                raise ValueError(f"Target substring '{target_prompt}' not found in prompt.")
    
            end_char = start_char + len(target_prompt)

            # Find matching token indices
            token_indices = [
                i for i, (start, end) in enumerate(offsets)
                if end > start_char and start < end_char
            ]

            matched_tokens = [tokens[i] for i in token_indices]
            matched_token_ids = [token_ids[i] for i in token_indices]

            return {
                "token_indices": token_indices,
                "tokens": matched_tokens,
                "token_ids": matched_token_ids
            }
    
    def visualize_tensor_pdf(self, tensors, save_path="tensor_histograms.png"):
        num_images = len(tensors)
        grid_cols = 6  # Adjust if needed
        grid_rows = math.ceil(num_images / grid_cols)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 2.5))

        for idx, tensor in enumerate(tensors):
            values = tensor.flatten().float().cpu().numpy()

            # Estimate the PDF using kernel density estimation
            kde = gaussian_kde(values)
            x_vals = np.linspace(values.min(), values.max(), 200)
            pdf_vals = kde(x_vals)

            row, col = divmod(idx, grid_cols)
            ax = axes[row][col] if grid_rows > 1 else axes[col]
            ax.plot(x_vals, pdf_vals, color='blue')
            ax.tick_params(labelsize=6)

        # Hide unused subplots
        for idx in range(num_images, grid_rows * grid_cols):
            row, col = divmod(idx, grid_cols)
            ax = axes[row][col] if grid_rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    def visualize_tensor_grid(self, tensors, save_path="tensor_grid.png", map_height=64, map_width=64):
        num_images = len(tensors)
        grid_cols = 4  # You can adjust this
        grid_rows = math.ceil(num_images / grid_cols)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))

        for idx, tensor in enumerate(tensors):
            image = tensor.view(map_height, map_width).float().cpu().numpy()
            row, col = divmod(idx, grid_cols)
            ax = axes[row][col] if grid_rows > 1 else axes[col]
            ax.imshow(image, cmap='jet')
            ax.axis('off')

        # Hide unused subplots
        for idx in range(num_images, grid_rows * grid_cols):
            row, col = divmod(idx, grid_cols)
            ax = axes[row][col] if grid_rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def time_shift(self, mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


    def get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    
    def get_schedule(
        self, num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # estimate mu based on linear estimation between two points
            mu = self.get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = self.time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()        

    @torch.no_grad()
    def get_img_latents(self, img_path, num_channels_latents):
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        img_width = int(img_width // (self.vae_scale_factor * 2) * (self.vae_scale_factor * 2))
        img_height = int(img_height // (self.vae_scale_factor * 2) * (self.vae_scale_factor * 2))
        img = img.resize((img_width, img_height))
        img = np.array(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
        img = img.unsqueeze(0)
        img = img.to(self._execution_device).to(self.vae.dtype)
        img_latent = self.vae.encode(img).latent_dist.mode()
        batch_size, channels_latents, height, width = img_latent.shape
        img_latent = (img_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        img_latent = self.pack_latents(img_latent, 1, num_channels_latents, height, width) 

        latent_image_ids = self.prepare_latent_image_ids(1, height // 2, width // 2, self._execution_device, self.transformer.dtype)
        
        return img_latent, latent_image_ids, height * self.vae_scale_factor, width * self.vae_scale_factor
    
    def encode_concepts(self, concepts, max_sequence_length, num_images_per_prompt, device):
        if type(concepts) == str:
            concepts = [concepts]

        if concepts[-1] != "background":
            concepts.append("background")

        concept_embeds = []
        for concept in concepts:
            concept_prompt_embeds, concept_pooled_prompt_embeds, concept_text_ids, concept_prompt_length = self.encode_prompt(
                prompt=concept,
                prompt_2=concept,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                device=device
            )

            concept_embeds.append(concept_prompt_embeds.mean(dim=1, keepdim=True))
        
        concept_embeds = torch.cat(concept_embeds, dim=1)
        concept_ids = torch.zeros(concept_embeds.shape[1], 3).to(device)

        return concept_embeds, concept_ids
    
    def get_target_token_idxs(self, prompt, target_prompt):
        if type(target_prompt) == str: # There is a single instance
            return self.get_substring_tokens(prompt, target_prompt)["token_indices"]
        elif type(target_prompt) == list:
            targets_token_idxs = []
            for target in target_prompt:
                assert type(target) == str, "The input 'target_prompt' should either be a string or a list of strings"
                target_token_idxs = self.get_substring_tokens(prompt, target)["token_indices"]
                targets_token_idxs.append(target_token_idxs)
            return targets_token_idxs
        else:
            raise ValueError("The input 'target_prompt' should either be a string or a list of strings")
    
    # ------------- helpers -----------------------------------------------------
    def gauss_kernel(self, k=3, sigma=1.0, device=None):
        ax = torch.arange(k, device=device) - k // 2
        g  = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k2 = (g[:, None] * g[None, :]) / g.sum() ** 2
        return k2[None, None]                       # [1,1,k,k]

    def renorm(self, x):
        return (x - x.amin((-1,-2),True)) / (x.amax((-1,-2),True) -
                                            x.amin((-1,-2),True) + 1e-8)

    def morph_reconstruct(self, seed, mask, iters=32):
        """Grayscale reconstruction by dilation (GPU)."""
        for _ in range(iters):
            seed = torch.min(mask, F.max_pool2d(seed, 3, 1, 1))
        return seed
    # ---------------------------------------------------------------------------

    def one_homogeneous_blob(self, mask_flat, H, W,
                            k=3, sigma=2.0, thr=0.5,
                            max_passes=20,
                            flatten='reconstruct',   # 'binary', 'reconstruct', 'distance'
                            lmb=6.0):
        """
        mask_flat : [B, H*W, 1]
        returns   : same shape; exactly one quasi-flat blob
        """
        B, N, _ = mask_flat.shape
        assert N == H * W

        mask   = mask_flat.view(B, 1, H, W)
        kernel = self.gauss_kernel(k, sigma).to(mask_flat)

        # 1. blur until 1 blob
        mask = self.renorm(mask)
        for _ in range(max_passes):
            mask = self.renorm(F.conv2d(mask, kernel, padding=k//2))
            # check CCs
            cc_ok = []
            for b in range(B):
                nb = (mask[b,0] > thr).byte().cpu().numpy()
                cc_ok.append(cv2.connectedComponents(nb)[0] - 1 <= 1)
            if all(cc_ok):
                break

        # 2. homogenise
        if flatten == 'binary':
            mask = (mask > thr).float()                         # flat 1/0

        elif flatten == 'reconstruct':
            # seed = peak mask, mask = original mask
            peak   = (mask == mask.amax((-1,-2),keepdim=True)).float()
            mask   = self.morph_reconstruct(peak, mask)              # flat plateau
            mask   = self.renorm(mask)                               # 0-1 again

        elif flatten == 'distance':
            # binary → distance-transform taper
            bmask  = (mask > thr).float()
            dist   = torch.from_numpy(
                    cv2.distanceTransform(bmask[0,0].cpu().numpy().astype(np.uint8),
                                            cv2.DIST_L2, 5)).to(mask.device)
            dist   = dist.unsqueeze(0).unsqueeze(0)             # [1,1,H,W]
            mask   = torch.exp(-dist / lmb) * bmask

        return mask.view(B, N, 1)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        target_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 1,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        invert: bool = False,
        latent_image_ids: Optional[torch.FloatTensor] = None,
        edit_start_step: int = 8,
        # shop_concept: scale per-slider del Concept Slider (continuo).
        # Lunghezza attesa == numero di slider caricati (== num_targets se
        # slider_to_target=None, == arbitrario altrimenti).
        # Se None, default a tutti 1.0 (equivalente a LoRAShop originale).
        target_lora_scales: Optional[List[float]] = None,
        # shop_concept multi-LoRA: mappa slider_idx -> target_idx.
        # `slider_to_target[i] = j` significa "slider i va applicato sulla
        # regione mascherata dal target_prompt[j]". Se None, default a
        # identita' (1 slider per target). Se piu' slider mappano allo
        # stesso target, le loro delta vengono SOMMATE additivamente
        # (composizionalita' Concept Sliders Metodo 2 / ExitStack via
        # multi-adapter PEFT) dentro la stessa regione mascherata.
        slider_to_target: Optional[List[int]] = None,
        # shop_concept: se settato (es. "out/seed42_scale1.0_mask"), salva
        # PNG delle maschere (soft + binary) usate per il blend, una per
        # ogni target_prompt. Suffissi: _target{i}_soft.png, _target{i}_seg.png.
        mask_dump_path: Optional[str] = None,
    ):

        target_token_idxs = self.get_target_token_idxs(prompt, target_prompt)

        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}

        joint_attention_kwargs["target_token_idxs"] = target_token_idxs

        # shop_concept: propaga target_lora_scales + mapping slider->target.
        num_targets = len(target_prompt) if isinstance(target_prompt, list) else 1

        if slider_to_target is None:
            # Backward-compat: 1 slider per target (identita').
            num_sliders = num_targets
            target_to_sliders = [[i] for i in range(num_targets)]
        else:
            num_sliders = len(slider_to_target)
            for s_idx, t_idx in enumerate(slider_to_target):
                if not (0 <= t_idx < num_targets):
                    raise ValueError(
                        f"slider_to_target[{s_idx}]={t_idx} fuori range "
                        f"[0, {num_targets}). Ci sono {num_targets} target_prompt."
                    )
            target_to_sliders = [[] for _ in range(num_targets)]
            for s_idx, t_idx in enumerate(slider_to_target):
                target_to_sliders[t_idx].append(s_idx)
            for t_idx, slist in enumerate(target_to_sliders):
                if len(slist) == 0:
                    raise ValueError(
                        f"target {t_idx} ('{target_prompt[t_idx]}') non ha alcuno "
                        f"slider associato. Aggiungi almeno uno slider che mappi "
                        f"a questo target o rimuovi il target."
                    )
            joint_attention_kwargs["target_to_sliders"] = target_to_sliders

        if target_lora_scales is not None:
            if len(target_lora_scales) != num_sliders:
                raise ValueError(
                    f"target_lora_scales ha len={len(target_lora_scales)} ma ci "
                    f"sono {num_sliders} slider caricati. Serve una scale per "
                    f"ogni concept slider."
                )
            joint_attention_kwargs["target_lora_scales"] = [float(s) for s in target_lora_scales]

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Prepare context
        prompt_embeds, pooled_prompt_embeds, text_ids, prompt_length_no_pad = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        if invert:
            assert latents is not None, "If editing real image, latents should not be None"
            assert latent_image_ids is not None, "If editing real image, latent_image_ids should not be None"
        else:
            latents, latent_image_ids = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=latents
            )

        downsample_factor = self.vae_scale_factor * 2
        map_height = height // downsample_factor
        map_width = width // downsample_factor

        # Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_sequence_length = latents.shape[1]
        mu = calculate_shift(
            image_seq_len=image_sequence_length,
            base_seq_len=self.scheduler.config.base_image_seq_len,
            max_seq_len=self.scheduler.config.max_image_seq_len,
            base_shift=self.scheduler.config.base_shift,
            max_shift=self.scheduler.config.max_shift
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
            sigmas=sigmas,
            mu=mu
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        joint_attention_kwargs["prompt_length"] = prompt_embeds.size(1)
        joint_attention_kwargs["prompt_length_no_pad"] = prompt_length_no_pad

        joint_attention_kwargs["size"] = (height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2))

        # Rotary Emb for image
        joint_attention_kwargs["img_only_rotary_emb"] = self.transformer.pos_embed(latent_image_ids)

        # Rotary Emb for concepts + image
        joint_attention_kwargs["concept_img_rotary_emb"] = self.transformer.pos_embed(torch.cat((text_ids, latent_image_ids), dim=0))

        # Rotary Emb for concepts
        joint_attention_kwargs["concept_rotary_emb"] = self.transformer.pos_embed(text_ids)

        subject_prior = None

        masks = [[] for _ in range(len(target_prompt))]

        # shop_concept mask-dump: catture dall'ultima iter di prior extraction.
        _mask_dump_soft = None  # list of soft sigmoid masks [1, N, 1], pre-binarize
        _mask_dump_raw = None   # list of raw normalized attention [B, N] pre-blob

        # pass for prior
        prior_latents = latents.clone()

        prior_sigmas = self.scheduler.sigmas

        joint_attention_kwargs["edit_start_step"] = edit_start_step
        # Extract the prior mask

        for i, t in enumerate(timesteps[:5]):
            if self.interrupt:
                    continue

            joint_attention_kwargs["current_iter"] = i

            timestep = torch.Tensor([t]).expand(latents.shape[0]).to(latents)
            sigma_curr = prior_sigmas[i]
            sigma_prev = prior_sigmas[i+1]

            joint_attention_kwargs["encoder_hidden_states"] = []
                
            joint_attention_kwargs["double_block_maps"] = []
            joint_attention_kwargs["single_block_maps"] = []

            joint_attention_kwargs["double_subject_priors"] = []
            joint_attention_kwargs["single_subject_priors"] = []

            joint_attention_kwargs["mode"] = "prior"

            noise_pred = self.transformer(
                hidden_states=prior_latents,
                timestep=sigma_curr.expand(prior_latents.shape[0]).to(prior_latents),
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False
            )[0]

            target_masks = joint_attention_kwargs["double_subject_priors"][-1]

            prior_latents_mid = prior_latents + (sigma_prev - sigma_curr) / 2 * noise_pred
            sigma_mid = sigma_curr + (sigma_prev - sigma_curr) / 2

            joint_attention_kwargs["encoder_hidden_states"] = []
                
            joint_attention_kwargs["double_block_maps"] = []
            joint_attention_kwargs["single_block_maps"] = []

            joint_attention_kwargs["double_subject_priors"] = []
            joint_attention_kwargs["single_subject_priors"] = []

            joint_attention_kwargs["mode"] = "prior"

            noise_pred_mid = self.transformer(
                hidden_states=prior_latents_mid,
                timestep=sigma_mid.expand(prior_latents_mid.shape[0]).to(prior_latents),
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False
            )[0]

            target_masks_mid = joint_attention_kwargs["double_subject_priors"][-1]
            
            first_order = (noise_pred_mid - noise_pred) / ((sigma_prev - sigma_curr) / 2)
            prior_latents = prior_latents + (sigma_prev - sigma_curr) * noise_pred + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order

            raw_masks = []
            test_masks = []
            avg_masks = []
            for idx, (target_mask, target_mask_mid) in enumerate(zip(target_masks, target_masks_mid)):
                target_mask = (target_mask + target_mask_mid) / 2
                target_mask = (target_mask - target_mask.min()) / (target_mask.max() - target_mask.min())
                target_mask = target_mask.squeeze(1)
                target_mask = self.one_homogeneous_blob(target_mask, map_height, map_width, sigma=1.0, thr=0.8)
                mask_mean = target_mask.mean()
                target_mask = F.sigmoid(10 * (target_mask - target_mask.mean()))
                target_mask = target_mask.view(1, -1, 1)
                masks[idx].append(target_mask)
                avg_mask = target_mask #torch.mean(torch.cat(masks[idx], dim=0), dim=0)
                mean_mask = avg_mask
                avg_mask = (mean_mask > torch.quantile(mean_mask.float(), 0.6)).to(avg_mask)#torch.quantile(mean_mask.float(), 0.6)).to(avg_mask)
                raw_masks.append ((mean_mask * avg_mask).view(1, -1, 1))
                avg_mask = avg_mask.view(1, -1, 1)
                avg_masks.append(avg_mask)
          
            values, labels = torch.max(torch.cat(raw_masks, dim=0), dim=0)
            confidence_threshold = 0.1  # adjust
            labels[values < confidence_threshold] = -1
            
            seg_masks = []
            for idx in range(len(target_masks)):
                seg_masks.append((labels == idx).to(latents))

            # shop_concept mask-dump: cattura le soft masks dall'ULTIMA iter
            # di prior-extraction (i corrisponde al timestep idx del for-loop).
            if mask_dump_path is not None and i == len(timesteps[:5]) - 1:
                # masks[idx][-1] e' la sigmoid-soft aggiunta a riga ~840
                _mask_dump_soft = [masks[idx][-1].detach().clone() for idx in range(len(target_masks))]

        avg_masks = seg_masks

        # shop_concept mask-dump: scrivi PNG per ciascun target.
        if mask_dump_path is not None:
            _dump_dir = os.path.dirname(mask_dump_path)
            if _dump_dir:
                os.makedirs(_dump_dir, exist_ok=True)

            def _mask_to_png(m, path, Hm, Wm):
                # m: tensor with N = Hm*Wm elements (any shape)
                arr = m.detach().float().cpu().numpy().reshape(Hm, Wm)
                arr = arr - arr.min()
                rng = arr.max()
                if rng > 0:
                    arr = arr / rng
                img_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_u8).save(path)

            coverages = []
            for _idx, _seg in enumerate(seg_masks):
                _p_seg = f"{mask_dump_path}_target{_idx}_seg.png"
                _mask_to_png(_seg, _p_seg, map_height, map_width)
                coverages.append(float(_seg.float().mean().item()))

            if _mask_dump_soft is not None:
                for _idx, _soft in enumerate(_mask_dump_soft):
                    _p_soft = f"{mask_dump_path}_target{_idx}_soft.png"
                    _mask_to_png(_soft, _p_soft, map_height, map_width)

            print(f"[mask_dump] wrote {mask_dump_path}_target*_{{seg,soft}}.png "
                  f"| coverage(seg)={coverages} | map={map_height}x{map_width}")

        # Reset the timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu
        )

        sigmas = self.scheduler.sigmas

        joint_attention_kwargs["is_inverted_latent"] = invert
        joint_attention_kwargs["cached_index"] = 3

        # Denoising Loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                joint_attention_kwargs["current_iter"] = i

                timestep = torch.Tensor([t]).expand(latents.shape[0]).to(latents)

                sigma_curr = sigmas[i]
                sigma_prev = sigmas[i+1]

                joint_attention_kwargs["encoder_hidden_states"] = []
                
                joint_attention_kwargs["double_block_maps"] = []
                joint_attention_kwargs["single_block_maps"] = []
                    
                joint_attention_kwargs["target_mask"] = avg_masks
                joint_attention_kwargs["mode"] = "blend"
                joint_attention_kwargs["order"] = "first"
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=sigma_curr.expand(latents.shape[0]).to(latents),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False
                )[0]

                latents_mid = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
                sigma_mid = sigma_curr + (sigma_prev - sigma_curr) / 2

                joint_attention_kwargs["encoder_hidden_states"] = []
                
                joint_attention_kwargs["double_block_maps"] = []
                joint_attention_kwargs["single_block_maps"] = []

                joint_attention_kwargs["target_mask"] = avg_masks
                joint_attention_kwargs["mode"] = "blend"
                joint_attention_kwargs["order"] = "second"
                noise_pred_mid = self.transformer(
                    hidden_states=latents_mid,
                    timestep=sigma_mid.expand(latents.shape[0]).to(latents),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False
                )[0]

                first_order = (noise_pred_mid - noise_pred) / ((sigma_prev - sigma_curr) / 2)
                latents = latents + (sigma_prev - sigma_curr) * noise_pred + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = self.unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PipelineOutput(images=image)

    @torch.no_grad()
    def invert(
        self,
        prompt: str = None,
        prompt_2: str = None,
        num_inversion_steps: int = 30,
        guidance_scale: float = 1.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 256,
        img_path: str = None
    ):

        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}
        
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Prepare context
        prompt_embeds, pooled_prompt_embeds, text_ids, prompt_length_no_pad = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=1,
            device=device,
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4

        latents, latent_image_ids, height, width = self.get_img_latents(img_path, num_channels_latents)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inversion_steps, num_inversion_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inversion_steps,
            device,
            None,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inversion_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        sigmas = self.scheduler.sigmas

        sigmas = sigmas.flip(0) # Flip for inversion

        # Handle Guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        save_start_iter = len(timesteps) - 3
        joint_attention_kwargs["save_start_iter"] = save_start_iter

        reversed_timesteps = list(timesteps)[::-1]

        # Inversion Loop
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                latents_dtype = latents.dtype

                sigma_curr = sigmas[i]
                sigma_prev = sigmas[i+1]

                joint_attention_kwargs["current_iter"] = i
                joint_attention_kwargs["timestep_index"] = reversed_timesteps.index(t)

                joint_attention_kwargs["mode"] = "invert"
                joint_attention_kwargs["order"] = "first"
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=sigma_curr.expand(latents.shape[0]).to(latents.dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False
                )[0]

                latents_mid = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
                sigma_mid = sigma_curr + (sigma_prev - sigma_curr) / 2

                joint_attention_kwargs["mode"] = "invert"
                joint_attention_kwargs["order"] = "second"
                noise_pred_mid = self.transformer(
                    hidden_states=latents_mid,
                    timestep=sigma_mid.expand(latents.shape[0]).to(latents.dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False
                )[0]

                first_order = (noise_pred_mid - noise_pred) / ((sigma_prev - sigma_curr) / 2)
                latents = latents + (sigma_prev - sigma_curr) * noise_pred + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self.maybe_free_model_hooks()

        return latents, latent_image_ids, height, width
            




