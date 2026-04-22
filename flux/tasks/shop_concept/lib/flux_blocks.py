# shop_concept/flux_blocks.py
# ---------------------------------------------------------------------------
# Derivato da LoRAShop-main/flux_blocks.py.
#
# DIFF RISPETTO ALL'ORIGINALE (vedi anche CHANGES.md):
#   1. set_adapter(module, target_idx) e' stato esteso in set_adapter(
#      module, target_idx, target_lora_scales=None): oltre a selezionare
#      l'adapter PEFT `default_{target_idx}`, imposta esplicitamente
#      `module.scaling[adapter] = target_lora_scales[target_idx]` se la
#      lista e' passata. Questo rende supportato il lora_scale CONTINUO
#      dei Concept Sliders (non solo 0.0 / 1.0).
#   2. In tutte le chiamate interne a set_adapter(...) passiamo
#      `joint_attention_kwargs.get("target_lora_scales")`.
#   3. enable_lora_all / disable_lora_all NON toccati: mantengono il
#      comportamento LoRAShop (scaling=1.0 come "ON", 0.0 come "OFF").
#      Nota che dopo la conversione degli slider a PEFT con fold_alpha=True
#      (vedi convert_slider_to_peft.py), "scaling=1.0" coincide con
#      "slider applicato a piena forza di training"; eventuali valori di
#      slider diversi da 1.0 vengono applicati via set_adapter nel loop
#      per-target.
# ---------------------------------------------------------------------------

import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np

from diffusers.utils import scale_lora_layers, unscale_lora_layers

from peft.tuners.tuners_utils import BaseTunerLayer

MASK_MIN_VAL = 1e-3


def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbounded_dim=-1):
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None].to(x.device)
        sin = sin[None, None].to(x.device)

        if use_real_unbounded_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbounded_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Helper condiviso (vs LoRAShop originale dove era copiato identico nelle
# due classi, senza il parametro target_lora_scales).
# ---------------------------------------------------------------------------
def _set_adapter_with_scale(module_to_set: nn.Module,
                            target_idx: int,
                            target_lora_scales=None):
    """Seleziona l'adapter PEFT `default_{target_idx}` su tutti i tuner layers
    del modulo. Se `target_lora_scales` e' una sequenza, imposta anche la
    scaling del layer a `target_lora_scales[target_idx]` (es. 0.5 / 1.5 per
    il concept slider continuo). Altrimenti la scaling non viene toccata
    (PEFT default, che nel nostro caso -> 1.0 grazie a fold_alpha).
    """
    adapter_name = f"default_{target_idx}"
    if target_lora_scales is not None and 0 <= target_idx < len(target_lora_scales):
        scale_val = float(target_lora_scales[target_idx])
    else:
        scale_val = None

    for m in module_to_set.modules():
        if isinstance(m, BaseTunerLayer):
            m.set_adapter(adapter_name)
            if scale_val is not None:
                m.scaling[adapter_name] = scale_val


class TransformerBlock(nn.Module):
    def __init__(self, orig_module, module_name):
        super().__init__()

        self.module_name = module_name
        self.orig_module = orig_module

        ## Normalization modules - Latent
        self.norm1 = orig_module.norm1
        self.norm2 = orig_module.norm2

        ## Normalization modules - Context
        self.norm1_context = orig_module.norm1_context
        self.norm2_context = orig_module.norm2_context

        ## Attention module
        self.attn = orig_module.attn

        ## Feed-Forward
        self.ff = orig_module.ff
        self.ff_context = orig_module.ff_context

    def disable_lora(self, module_to_disable: nn.Module):
        for module in module_to_disable.modules():
            if isinstance(module, BaseTunerLayer):
                for active_adapter in module.active_adapters:
                    module.scaling[active_adapter] = 0

    def disable_lora_all(self):
        self.disable_lora(self.norm1)
        self.disable_lora(self.norm2)
        self.disable_lora(self.norm1_context)
        self.disable_lora(self.norm2_context)
        self.disable_lora(self.attn)
        self.disable_lora(self.ff)
        self.disable_lora(self.ff_context)

    def enable_lora(self, module_to_disable: nn.Module):
        for module in module_to_disable.modules():
            if isinstance(module, BaseTunerLayer):
                for active_adapter in module.active_adapters:
                    module.scaling[active_adapter] = 1.0

    def enable_lora_all(self):
        self.enable_lora(self.norm1)
        self.enable_lora(self.norm2)
        self.enable_lora(self.norm1_context)
        self.enable_lora(self.norm2_context)
        self.enable_lora(self.attn)
        self.enable_lora(self.ff)
        self.enable_lora(self.ff_context)

    # MODIFICATO (shop_concept): accetta target_lora_scales opzionale.
    def set_adapter(self, module_to_set: nn.Module, adapter_idx: int,
                    target_lora_scales=None):
        _set_adapter_with_scale(module_to_set, adapter_idx, target_lora_scales)

    def calc_attention_mask(self, img_features, text_features, target_token_indices, joint_attention_kwargs):
        k_text = self.attn.add_k_proj(text_features)

        batch_size, _, _ = k_text.shape
        inner_dim = k_text.shape[-1]
        head_dim = inner_dim // self.attn.heads

        k_text = k_text.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_added_k:
            k_text = self.attn.norm_added_k(k_text)

        num_text_tokens = k_text.shape[1]

        k_img = self.attn.to_k(img_features)
        k_img = k_img.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        q_text = self.attn.add_q_proj(text_features)
        q_text = q_text.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_added_q:
            q_text = self.attn.norm_added_q(q_text)

        q_img = self.attn.to_q(img_features)
        q_img = q_img.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q:
            q_img = self.attn.norm_q(q_img)

        query = q_img
        key = k_text

        query = apply_rotary_emb(query, joint_attention_kwargs["img_only_rotary_emb"])
        key = apply_rotary_emb(key, joint_attention_kwargs["concept_rotary_emb"])

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = math.sqrt(q_text.size(-1))
        attention_scores = attention_scores / scale_factor
        attention_scores = torch.softmax(attention_scores, dim=-1)

        attn_maps = []
        for target in target_token_indices:
            attention_map = attention_scores.mean(dim=1)
            attention_map = attention_map[:, :, target].mean(dim=-1)
            attention_map = attention_map.unsqueeze(-1)
            attention_map = attention_map - attention_map.min()
            attention_map = attention_map / attention_map.sum()
            attn_maps.append(attention_map)

        return attn_maps

    def calc_attention(self, hidden_states, encoder_hidden_states, image_rotary_emb, cached_value=None):
        batch_size, _, _ = encoder_hidden_states.shape

        # Image Q, K, V features
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        if cached_value is not None:
            value = cached_value
        else:
            value = self.attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        # Normalize Q, K
        if self.attn.norm_q is not None:
            query = self.attn.norm_q(query)
        if self.attn.norm_k is not None:
            key = self.attn.norm_k(key)

        # Text Q, K, V features
        text_query_proj = self.attn.add_q_proj(encoder_hidden_states)
        text_key_proj = self.attn.add_k_proj(encoder_hidden_states)
        text_value_proj = self.attn.add_v_proj(encoder_hidden_states)

        text_query_proj = text_query_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        text_key_proj = text_key_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        text_value_proj = text_value_proj.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_added_q is not None:
            text_query_proj = self.attn.norm_added_q(text_query_proj)
        if self.attn.norm_added_k is not None:
            text_key_proj = self.attn.norm_added_k(text_key_proj)

        # Calculate Attention
        query = torch.cat([text_query_proj, query], dim=2)
        key = torch.cat([text_key_proj, key], dim=2)
        value = torch.cat([text_value_proj, value], dim=2)

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :encoder_hidden_states.shape[1], :],
            hidden_states[:, encoder_hidden_states.shape[1]:, :]
        )

        # Linear Projection
        hidden_states = self.attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = self.attn.to_out[1](hidden_states)

        # Text Linear Projection
        encoder_hidden_states = self.attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


    def forward_blend_block(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        target_token_idxs = joint_attention_kwargs["target_token_idxs"]
        interest_token_idxs = target_token_idxs

        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}
        target_lora_scales = joint_attention_kwargs.get("target_lora_scales")  # shop_concept

        # Constructing the attn_output

        # LoRA pass for attn_output
        lora_outputs = []
        lora_shift_mlps = []
        lora_scale_mlps = []
        lora_gate_mlps = []
        for target_idx in range(len(interest_token_idxs)):
            # Normalization Pass
            self.set_adapter(self.norm1, target_idx, target_lora_scales)
            lora_norm_hidden_states, lora_gate_msa, lora_shift_mlp, lora_scale_mlp, lora_gate_mlp = self.norm1(hidden_states, emb=temb)
            self.set_adapter(self.norm1_context, target_idx, target_lora_scales)
            lora_norm_encoder_hidden_states, lora_c_gate_msa, lora_c_shift_mlp, lora_c_scale_mlp, lora_c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

            # Attention pass
            self.set_adapter(self.attn, target_idx, target_lora_scales)
            lora_attn_output, lora_context_attn_output = self.calc_attention(
                hidden_states=lora_norm_hidden_states,
                encoder_hidden_states=lora_norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb
            )

            lora_attn_output = lora_gate_msa.unsqueeze(1) * lora_attn_output
            lora_outputs.append(lora_attn_output)
            lora_shift_mlps.append(lora_shift_mlp)
            lora_scale_mlps.append(lora_scale_mlp)
            lora_gate_mlps.append(lora_gate_mlp)

        # Base model pass for attn_output
        self.disable_lora_all()
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        attn_output, context_attn_output = self.calc_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        # Blending the attn_outputs
        lora_attn_masks = joint_attention_kwargs["target_mask"]

        if joint_attention_kwargs["current_iter"] >= joint_attention_kwargs["edit_start_step"]:
            # Blending step
            blending_numerator = torch.zeros_like(attn_output)
            blending_denumerator = torch.zeros_like(attn_output)

            for target_idx in range(len(interest_token_idxs)):
                blending_numerator += lora_attn_masks[target_idx] * lora_outputs[target_idx]
                blending_denumerator += lora_attn_masks[target_idx]

            raw_denumerator = blending_denumerator.clone()
            blending_denumerator = blending_denumerator.clamp(min=MASK_MIN_VAL)
            blending_output = blending_numerator / blending_denumerator

            no_mask_pixels = (raw_denumerator < MASK_MIN_VAL)
            blending_output[no_mask_pixels] = attn_output[no_mask_pixels]
            attn_output = blending_output

        hidden_states = hidden_states + attn_output
        self.enable_lora_all()

        # Constructing the ff_output
        lora_ff_outputs = []
        for target_idx in range(len(interest_token_idxs)):
            # Normalization pass
            self.set_adapter(self.norm2, target_idx, target_lora_scales)
            lora_norm_hidden_states = self.norm2(hidden_states)
            lora_norm_hidden_states = lora_norm_hidden_states * (1 + lora_scale_mlps[target_idx][:, None]) + lora_shift_mlps[target_idx][:, None]
            self.set_adapter(self.ff, target_idx, target_lora_scales)
            lora_ff_output = self.ff(lora_norm_hidden_states)
            lora_ff_output = lora_gate_mlps[target_idx].unsqueeze(1) * lora_ff_output
            lora_ff_outputs.append(lora_ff_output)

        # Base model pass for ff_output
        self.disable_lora_all()
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp[:, None] * ff_output

        # Blending the ff_output
        if joint_attention_kwargs["current_iter"] >= 3:
            # Blending step
            blending_numerator = torch.zeros_like(ff_output)
            blending_denumerator = torch.zeros_like(ff_output)

            for target_idx in range(len(interest_token_idxs)):
                blending_numerator += lora_attn_masks[target_idx] * lora_ff_outputs[target_idx]
                blending_denumerator += lora_attn_masks[target_idx]

            raw_denumerator = blending_denumerator.clone()
            blending_denumerator = blending_denumerator.clamp(min=MASK_MIN_VAL)
            blending_output = blending_numerator / blending_denumerator

            no_mask_pixels = (raw_denumerator < MASK_MIN_VAL)
            blending_output[no_mask_pixels] = ff_output[no_mask_pixels]
            ff_output = blending_output

        hidden_states = hidden_states + ff_output
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        self.enable_lora_all()

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_prior_extract(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        # Normalize Latents
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize Context
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention
        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        attn_map = self.calc_attention_mask(
            img_features=norm_hidden_states,
            text_features=norm_encoder_hidden_states,
            target_token_indices=joint_attention_kwargs["target_token_idxs"],
            joint_attention_kwargs=joint_attention_kwargs
        )

        joint_attention_kwargs["double_subject_priors"].append(attn_map)

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_block(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        # Normalize Latents
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # Normalize Context
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention
        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None, **kwargs):
        # **kwargs: compat shim per diffusers >= 0.36 che puo' passare
        # kwargs addizionali (es. attention_mask, controlnet_*) che LoRAShop
        # 0.31 non conosceva. Li assorbiamo senza usarli, il comportamento
        # semantico e' identico.
        if joint_attention_kwargs["mode"] == "blend":
            self.enable_lora_all()
            return self.forward_blend_block(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        elif joint_attention_kwargs["mode"] == "pass":
            self.disable_lora_all()
            return self.forward_block(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        elif joint_attention_kwargs["mode"] == "prior":
            self.disable_lora_all()
            return self.forward_prior_extract(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        elif joint_attention_kwargs["mode"] == "invert":
            self.disable_lora_all()
            return self.forward_block(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        else:
            raise ValueError("Invalid value in joint_attention_kwargs['mode'], it should be one of ('blend', 'pass', 'prior', 'invert')")


class SingleTransformerBlock(nn.Module):
    def __init__(self, orig_module, module_name):
        super().__init__()

        self.module_name = module_name

        # Registering Modules
        self.orig_module = orig_module

        ## Normalization Module
        self.norm = orig_module.norm

        ## Projection Modules
        self.mlp_hidden_dim = orig_module.mlp_hidden_dim
        self.proj_mlp = orig_module.proj_mlp
        self.proj_out = orig_module.proj_out
        self.act_mlp = orig_module.act_mlp

        ## Attention Module
        self.attn = orig_module.attn

        self.cached_values = {}

    def disable_lora(self, module_to_disable: nn.Module):
        for module in module_to_disable.modules():
            if isinstance(module, BaseTunerLayer):
                for active_adapter in module.active_adapters:
                    module.scaling[active_adapter] = 0

    def disable_lora_all(self):
        self.disable_lora(self.norm)
        self.disable_lora(self.proj_mlp)
        self.disable_lora(self.proj_out)
        self.disable_lora(self.act_mlp)
        self.disable_lora(self.attn)

    def enable_lora_all(self):
        self.enable_lora(self.norm)
        self.enable_lora(self.proj_mlp)
        self.enable_lora(self.proj_out)
        self.enable_lora(self.act_mlp)
        self.enable_lora(self.attn)

    def enable_lora(self, module_to_disable: nn.Module):
        for module in module_to_disable.modules():
            if isinstance(module, BaseTunerLayer):
                for active_adapter in module.active_adapters:
                    module.scaling[active_adapter] = 1.0

    # MODIFICATO (shop_concept): accetta target_lora_scales opzionale.
    def set_adapter(self, module_to_set: nn.Module, adapter_idx: int,
                    target_lora_scales=None):
        _set_adapter_with_scale(module_to_set, adapter_idx, target_lora_scales)

    def calc_attention_mask(self, img_features, text_features, target_token_indices, joint_attention_kwargs):
        k_img = self.attn.to_k(img_features)

        batch_size, _, _ = k_img.shape
        inner_dim = k_img.shape[-1]
        head_dim = inner_dim // self.attn.heads

        k_img = k_img.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        q_img = self.attn.to_q(img_features)

        q_img = q_img.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        q_text = self.attn.to_q(text_features)

        q_text = q_text.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        k_text = self.attn.to_k(text_features)
        k_text = k_text.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q:
            q_img = self.attn.norm_q(q_img)
        if self.attn.norm_k:
            k_text = self.attn.norm_k(k_text)

        query = q_img
        key = k_text

        query = apply_rotary_emb(query, joint_attention_kwargs["img_only_rotary_emb"])
        key = apply_rotary_emb(key, joint_attention_kwargs["concept_rotary_emb"])

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = math.sqrt(q_text.size(-1))
        attention_scores = attention_scores / scale_factor

        attention_scores = torch.softmax(attention_scores, dim=-1)

        attn_maps = []
        for target in target_token_indices:
            attention_map = attention_scores.mean(dim=1)
            attention_map = attention_map[:, :, target].mean(dim=-1)
            attention_map = attention_map.unsqueeze(-1)
            attention_map = attention_map - attention_map.min()
            attention_map = attention_map / attention_map.sum()
            attn_maps.append(attention_map)

        return attn_maps

    def calc_attention(self, hidden_states, image_rotary_emb, cached_value=None, return_value=False):
        batch_size, _, _ = hidden_states.shape

        # Q, K, V features
        query = self.attn.to_q(hidden_states)
        key = self.attn.to_k(hidden_states)
        if cached_value is not None:
            value = cached_value
        else:
            value = self.attn.to_v(hidden_states)

        if return_value:
            value_to_return = value.clone()

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        if self.attn.norm_q:
            query = self.attn.norm_q(query)
        if self.attn.norm_k:
            key = self.attn.norm_k(key)

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if return_value:
            return hidden_states, value_to_return

        return hidden_states


    def forward_blend_block(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        residual = hidden_states

        target_token_idxs = joint_attention_kwargs["target_token_idxs"]
        interest_token_idxs = target_token_idxs
        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}
        target_lora_scales = joint_attention_kwargs.get("target_lora_scales")  # shop_concept

        # LoRA pass for all targets
        lora_outputs = []
        for target_idx in range(len(interest_token_idxs)):
            # Normalization Pass
            self.set_adapter(self.norm, target_idx, target_lora_scales)
            lora_norm_hidden_states, lora_gate = self.norm(hidden_states, emb=temb)
            # Projection Pass
            self.set_adapter(self.proj_mlp, target_idx, target_lora_scales)
            self.set_adapter(self.act_mlp, target_idx, target_lora_scales)
            lora_mlp_hidden_states = self.act_mlp(self.proj_mlp(lora_norm_hidden_states))
            # Attention pass
            self.set_adapter(self.attn, target_idx, target_lora_scales)
            lora_attn_output = self.calc_attention(
                hidden_states=lora_norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **{}
            )

            lora_hidden_states = torch.cat([lora_attn_output, lora_mlp_hidden_states], dim=2)
            lora_gate = lora_gate.unsqueeze(1)
            # Output Projection pass
            self.set_adapter(self.proj_out, target_idx, target_lora_scales)
            lora_hidden_states = lora_gate * self.proj_out(lora_hidden_states)
            lora_outputs.append(lora_hidden_states)

        # Original model pass
        self.disable_lora_all()
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        if joint_attention_kwargs["is_inverted_latent"] and joint_attention_kwargs["current_iter"] < joint_attention_kwargs["cached_index"]:
            value = self.cached_values[f"inv_iter_{joint_attention_kwargs['current_iter']}_order_{joint_attention_kwargs['order']}_V"]
            value = value.to(hidden_states)
            attn_output = self.calc_attention(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                cached_value=value
            )
        else:
            attn_output = self.calc_attention(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)

        lora_attn_masks = joint_attention_kwargs["target_mask"]

        self.enable_lora_all()
        # Begin Blending
        if joint_attention_kwargs["current_iter"] >= joint_attention_kwargs["edit_start_step"]:
            blended_numerator = torch.zeros_like(hidden_states[:, joint_attention_kwargs["prompt_length"]:, :])
            blended_denominator = torch.zeros_like(hidden_states[:, joint_attention_kwargs["prompt_length"]:, :])

            for target_idx in range(len(interest_token_idxs)):
                blended_numerator += lora_attn_masks[target_idx] * lora_outputs[target_idx][:, joint_attention_kwargs["prompt_length"]:, :]
                blended_denominator += lora_attn_masks[target_idx]

            raw_denumerator = blended_denominator.clone()
            blended_denominator = blended_denominator.clamp(min=MASK_MIN_VAL)
            blended_output = blended_numerator / blended_denominator

            no_mask_pixels = (raw_denumerator < MASK_MIN_VAL)
            blended_output[no_mask_pixels] = hidden_states[:, joint_attention_kwargs["prompt_length"]:, :][no_mask_pixels]
            hidden_states[:, joint_attention_kwargs["prompt_length"]:, :] = blended_output

        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def forward_block(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}

        attn_output = self.calc_attention(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def forward_block_invert(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}

        if joint_attention_kwargs["current_iter"] >= joint_attention_kwargs["save_start_iter"]:
            attn_output, value = self.calc_attention(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                return_value=True
            )

            # Register the value
            value = value.cpu()
            self.cached_values[f"inv_iter_{joint_attention_kwargs['timestep_index']}_order_{joint_attention_kwargs['order']}_V"] = value
        else:
            attn_output = self.calc_attention(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                return_value=False
            )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def forward_prior_extract(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb
        )

        norm_img_hidden_states = norm_hidden_states[:, joint_attention_kwargs["prompt_length"]:, :]
        norm_encoder_hidden_states = norm_hidden_states[:, :joint_attention_kwargs["prompt_length"], :]

        attn_map = self.calc_attention_mask(
            img_features=norm_img_hidden_states,
            text_features=norm_encoder_hidden_states,
            target_token_indices=joint_attention_kwargs["target_token_idxs"],
            joint_attention_kwargs=joint_attention_kwargs
        )

        joint_attention_kwargs["single_subject_priors"].append(attn_map)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        **kwargs,
    ):
        # Compat diffusers>=0.36: i SingleTransformerBlock ricevono
        # `encoder_hidden_states` separato e devono restituire la tupla
        # (encoder_hidden_states_out, hidden_states_out), come gia' fanno i
        # double blocks. In 0.31 (e nelle forward_* interne) il single block
        # lavora invece su una singola `hidden_states` gia' concatenata
        # [encoder; image]. Quindi:
        #   - se riceviamo encoder_hidden_states != None: concat all'inizio,
        #     chiamiamo la logica interna invariata, poi split finale e
        #     return tuple.
        #   - se riceviamo encoder_hidden_states is None (uso legacy o
        #     prior-extract dalla nostra pipeline): comportamento originale
        #     LoRAShop, return single tensor concatenato.
        split_return = encoder_hidden_states is not None
        if split_return:
            enc_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        if joint_attention_kwargs["mode"] == "blend":
            self.enable_lora_all()
            out = self.forward_blend_block(hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        elif joint_attention_kwargs["mode"] == "pass":
            self.disable_lora_all()
            out = self.forward_block(hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        elif joint_attention_kwargs["mode"] == "prior":
            self.disable_lora_all()
            out = self.forward_prior_extract(hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        elif joint_attention_kwargs["mode"] == "invert":
            self.disable_lora_all()
            out = self.forward_block_invert(hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        else:
            raise ValueError("Invalid value in joint_attention_kwargs['mode'], it should be one of ('blend', 'pass', 'prior', 'invert')")

        if split_return:
            # diffusers>=0.36 si aspetta tuple (enc_out, img_out)
            return out[:, :enc_len], out[:, enc_len:]
        return out
