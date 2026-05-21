# `flux/tasks/shop_concept/`

LoRAShop-based variant of the mask-guided application, used as the
**internal-mask alternative** discussed in Appendix D of the paper.
Instead of taking the mask from a separate SAM segmentation pass, the
mask is derived internally from the model's own cross-attention during
the first few denoising steps, then used to gate one or more Concept
Sliders inside the regions matched by their `target_prompt`.

In the paper we report this pipeline as an alternative we explored. The
attention-derived masks turned out to be too coarse for the kind of
fine-grained, attribute-level edits that we ultimately wanted, and that
finding motivated the external SAM-based design in
`flux/tasks/masked_lora/`. The code is still useful both as the implementation
of the alternative (so the comparison can be reproduced) and as a working
example of how a Concept Slider can be plugged into a mask-aware Flux
pipeline. The qualitative example used in the appendix of the paper
(`fig:appendix:internal-masks`) is produced here.

The pipeline is Flux-only: SDXL is not supported.

## Origin

The transformer-level mask-aware machinery is adapted from
[LoRAShop](https://github.com/gemlab-vt/LoRAShop) (Dalva et al., 2025, MIT
licensed). LoRAShop assumes one LoRA per region and applies each adapter at
fixed strength (`scaling = 1`); for Concept Sliders we need a continuous
per-slider scale `s_i` and, optionally, several sliders inside the same
region. The two key modifications on top of LoRAShop are therefore:

- a `target_lora_scales` argument that propagates the continuous slider scale
  all the way into PEFT's `scaling` table, replacing LoRAShop's
  toggle-on/toggle-off behaviour with the per-slider intensity that the paper
  refers to as `s_i`;
- a `slider_to_target` mapping that lets multiple PEFT adapters be active in
  the same region; PEFT then sums their deltas additively
  (`W·x + Σ_i s_i · B_i A_i · x`), which is the Concept-Sliders compositional
  recipe applied inside a masked region.

Two smaller compatibility fixes are also part of this directory because they
are needed to run the LoRAShop-style code on the diffusers / torch versions we
used:

- the `forward` methods of the patched transformer blocks accept the
  additional keyword arguments that diffusers ≥ 0.36 passes to attention
  blocks (e.g. `attention_mask`, `controlnet_*`), and the single-block
  variant returns the `(encoder_hidden_states, hidden_states)` tuple expected
  by the newer pipeline;
- `__init__.py` installs a small shim on `torch.nn.functional.scaled_dot_product_attention`
  that drops the `enable_gqa` kwarg, which diffusers ≥ 0.36 passes
  unconditionally but which only exists on torch ≥ 2.5. Flux does not use
  grouped-query attention, so dropping the kwarg is semantically a no-op.

Everything else outside of `lib/` — the CLI entrypoints, the slider
converter, the sweep wrapper, and the localization-evaluation harness — was
written for this project.

## Layout

```
flux/tasks/shop_concept/
├── __init__.py                       # SDPA compatibility shim
├── lib/
│   ├── utils.py                      # vendored from LoRAShop, unchanged
│   ├── flux_blocks.py                # adapted from LoRAShop (see "Origin")
│   └── flux_real_pipeline.py         # adapted from LoRAShop (see "Origin")
├── scripts/
│   ├── generate.py                   # main CLI entrypoint
│   ├── sweep.py                      # single-slider seed × scale sweep
│   └── convert_slider_to_peft.py     # convert Concept Slider .pt → PEFT .safetensors
├── jobs/
│   └── new_slurm/                    # SLURM templates used for the paper results
├── outputs/                          # one-shot generation outputs (populated locally)
└── README.md                         # this file
```

All entrypoints are designed to be launched from the repository root, either
as a module (`python -m flux.tasks.shop_concept.scripts.generate ...`) or as a
script (`python flux/tasks/shop_concept/scripts/generate.py ...`).

## How it works in one paragraph

The pipeline runs a Flux denoising trajectory in two phases. In the first
~5 timesteps it runs the unmodified transformer (LoRA disabled) and
accumulates per-target attention maps from one of the late transformer
blocks; those maps are post-processed into one binary mask per
`target_prompt`. From the timestep specified by `--edit_start_step`
onwards, the pipeline runs two forward passes per step — one with LoRA off
and one with the requested sliders active — and blends their hidden states
inside the masked regions using the per-target masks. The base prediction is
kept outside any mask, so regions that do not match any target are not
touched. The `target_prompt` strings must be **exact substrings** of
`--prompt`, because the matching to attention tokens is done by literal
character span (`prompt.index(target_prompt)`); we keep this constraint
unchanged from the upstream design.

## Slider format

Concept Sliders are trained with the LoRA framework under
`flux/trained_sliders/training/` and saved as `.pt` files in
kohya-ss / `LoRANetwork` layout. LoRAShop instead expects PEFT
`.safetensors`. The converter in `scripts/convert_slider_to_peft.py` bridges
the two formats: it enumerates the 266 attention linear modules of Flux 1.0
trained with `train_method='xattn'` (19 double blocks × 8 + 38 single blocks
× 3), renames each `lora_unet_<flat_path>.lora_down/up.weight` into the
corresponding `transformer.<dotted_path>.lora_A/B.weight`, and folds
`alpha / rank` into `lora_B` so that PEFT's default `scaling = 1` matches
"slider applied at the strength it was trained with". A `.pt` passed to any
of the entrypoints is converted on the fly and cached under
`flux/tasks/shop_concept/_peft_cache/`; `.safetensors` files are loaded as is.

## Usage

The three entrypoints below cover the main use cases. All of them use
Flux.1-dev as the base model and assume the repository root is the working
directory.

### Single edit

`scripts/generate.py` is the main CLI: load Flux once, apply one or more
sliders to one or more regions of a single image.

```bash
# One slider on one region (the man's smile)
python -m flux.tasks.shop_concept.scripts.generate \
       --prompt "a man and a woman facing the camera" \
       --target_prompt "man" \
       --slider_paths flux/trained_sliders/sliders/general/smile/slider_0.pt \
       --lora_scales 1.5 \
       --output_path flux/tasks/shop_concept/outputs/smile_on_man.png \
       --seed 42 --height 1024 --width 1024

# Two sliders on two different regions (1:1 mapping, no flag needed)
python -m flux.tasks.shop_concept.scripts.generate \
       --prompt "a man standing in front of a landscape" \
       --target_prompt "landscape" "man" \
       --slider_paths flux/trained_sliders/sliders/general/painterly/slider_0.pt \
                      flux/trained_sliders/sliders/general/age/slider_0.pt \
       --lora_scales  1.0 1.0 \
       --output_path flux/tasks/shop_concept/outputs/compose.png \
       --seed 42

# Several sliders inside the same region: composition is additive (Appendix D)
python -m flux.tasks.shop_concept.scripts.generate \
       --prompt "a man and a woman facing the camera" \
       --target_prompt "man" "woman" \
       --slider_paths flux/trained_sliders/sliders/general/smile/slider_0.pt \
                      flux/trained_sliders/sliders/general/age/slider_0.pt \
                      flux/trained_sliders/sliders/general/smile/slider_0.pt \
                      flux/trained_sliders/sliders/general/age/slider_0.pt \
       --slider_to_target 0 0 1 1 \
       --lora_scales       1.0 1.0 -1.0 -1.0 \
       --output_path flux/tasks/shop_concept/outputs/compose_both.png \
       --seed 42
```

### Sweep on a single slider

`scripts/sweep.py` is a thin wrapper around `generate.py` that loads Flux
once and iterates over `(seed × scale)` pairs internally. Useful when one
wants a dose-response grid for a single slider on a single region; running
`generate.py` in a bash loop instead would pay the Flux load time on every
image.

```bash
python -m flux.tasks.shop_concept.scripts.sweep \
       --slider_path flux/trained_sliders/sliders/general/smile/slider_0.pt \
       --prompt "a man and a woman facing the camera" \
       --target_prompt "man" \
       --seeds 42 123 \
       --scales 0.0 0.5 1.0 1.5 2.0 \
       --output_dir flux/tasks/shop_concept/outputs/sweep_smile_man/
```

### Note on evaluation

This pipeline was used **qualitatively** to produce the figure in
Appendix D of the paper (`paper_figures/lorashop.png`), which
illustrates that the internal cross-attention masks are too coarse for
fine-grained attribute edits. We did not run a quantitative evaluation
of this variant — that conclusion motivated the external mask-guided
design implemented under [`flux/tasks/masked_lora/`](../masked_lora/).

## Useful parameters

| Flag | Default | Meaning |
|---|---|---|
| `--num_inference_steps` | 30 | Total flow-matching steps |
| `--edit_start_step` | 8 | First step at which the mask-guided blend is active; before this the model runs without LoRA and only collects the prior masks |
| `--guidance_scale` | 3.5 | Flux.1-dev distilled guidance |
| `--height`, `--width` | 1024 each | Native Flux resolution |
| `--max_sequence_length` | 256 | T5 prompt length cap |
| `--cache_dir` | `flux/tasks/shop_concept/_peft_cache` | Where converted PEFT `.safetensors` are cached |
| `--lora_fill_rank` | 16 | Rank used to pad missing keys when several sliders are aligned together |
| `--seed` | none | Random seed; required for reproducible runs |

## Constraint on `target_prompt`

Every `target_prompt` must be a **literal substring** of `--prompt`. The
matching to attention tokens uses `prompt.index(target_prompt)`; if the
substring is not found, the entrypoint raises an error.

For example, with `--prompt "a woman in a red dress and a man smiling"`:

- `"woman in a red dress"` is a valid `target_prompt`
- `"the woman"` is not — the article is not in the prompt
- `"red dress"` is valid but selects only the dress tokens, which is what
  attention will localize on

## Notes

- The compute cost per image is roughly twice the cost of an unconditioned
  Flux generation, because the active denoising steps run one base forward
  plus one forward per region (the latter performed once with all relevant
  adapters active).
- Disjoint masks are assumed when several `target_prompt`s are passed. If
  the cross-attention assigns the same image tokens to more than one target,
  the blend mixes their contributions in those tokens; the entrypoint does
  not enforce non-overlap, it only logs the overlap area when it detects one.
- Setting `target_lora_scales = [0.0, 0.0, ...]` for every slider produces
  a clean baseline equivalent to a plain Flux generation: useful when the
  prompt is fixed and one wants the unedited image for reference.
- The SLURM templates under `jobs/new_slurm/` reflect the configuration
  used for the paper results.
