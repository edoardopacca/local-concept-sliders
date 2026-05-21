# `sdxl/tasks/real_editing/`

Real-image editing pipeline used for the verification mentioned in §1 of
the paper: the same masked-edit operator of §4 is applied to a real
photograph after a preliminary inversion. The inversion quality is an
orthogonal variable to slider localisation, so this task is kept
outside the main experimental loop; the SDXL pipeline here is what we
used to confirm that the edit operator transfers from generated to real
images.

## Pipeline

```
real image
   |
   v
[Stage 1] Tight inversion (HPC)
   |   scripts/invert_real_image.py
   |   Writes outputs/<RUN_ID>/{original.png, reconstruction.png,
   |     x_t.pt, metadata.json, ...}
   v
[Stage 2] SAM mask (local, interactive)
   |   mask_SAM/segment_with_sam.py on reconstruction.png
   |   Writes outputs/<RUN_ID>/mask_target.png
   v
[Stage 3] Masked LoRA edit (HPC)
       scripts/edit_real_image_masked.py
       Writes outputs/<RUN_ID>/{edited_<name>.png,
         composite_edited_<name>.png}
```

The crop in Stage 2 must be taken on `reconstruction.png`, not on
`original.png`: SDXL works on the 1024x1024 crop and a mask drawn on
the full-resolution original would not align with the latents.

The final pixel-space composite (`composite_edited_<name>.png`) pastes
the edited 1024x1024 crop back onto the full-resolution original, so
the part of the image outside the mask is preserved at the original
resolution.

## Origin

The inversion backend is adapted from **Tight Inversion** (Kadosh et al.,
2025, [HuggingFace Space](https://huggingface.co/spaces/tight-inversion/tight-inversion),
[arXiv:2502.20376](https://arxiv.org/abs/2502.20376)). The Space ships
without an explicit license file; we use the adapted excerpts as
research-only reference and credit the authors accordingly. The
file-level mapping is:

| File here | Adapted from (Tight Inversion HF Space) | What was extracted |
|---|---|---|
| `lib/inversion/tight_inversion.py` | `src/exact_inversion.py`, `src/pipes/sdxl_inversion_pipeline.py`, `app.py` | the `inversion_step()` GD optimisation, the `unet_pass()` helper, the SDXL inversion loop with `added_cond_kwargs`, the IP-Adapter integration pattern |
| `lib/inversion/ddim.py` | `src/schedulers/ddim_scheduler.py` | the `inv_step()` reverse DDIM math |

Runtime weights downloaded on first use: IP-Adapter (plus, SDXL, ViT-H)
from [`h94/IP-Adapter`](https://huggingface.co/h94/IP-Adapter) (Apache-2.0)
for the image conditioning, and SDXL base
(`stabilityai/stable-diffusion-xl-base-1.0`, OpenRAIL++) as the backbone.

Everything else in this directory — the masked-edit operator
(`lib/editing/`), the IO helpers (`lib/io/`), the SDXL `ModelContext`
wrapper (`lib/models/sdxl.py`), and the two CLI entrypoints — is
original to this work. The files in `lib/archive/` (`null_text.py`,
`provenance.py`, `sd1x.py`) are early SD-1.x prototypes kept for
reference; they are not used by the pipeline reported in the paper.

## Layout

```
sdxl/tasks/real_editing/
├── scripts/
│   ├── invert_real_image.py            # Stage 1: invert a real image
│   └── edit_real_image_masked.py       # Stage 3: masked LoRA edit
├── lib/
│   ├── models/                         # ModelContext (SDXL) + loader factory
│   ├── inversion/                      # backends: tight_inversion, ddim, registry
│   ├── editing/                        # MaskedLoRAEditor, blending, slider loader
│   ├── io/                             # artefact (de)serialisation, metrics
│   └── archive/                        # legacy SD1.x / null_text / EDICT (not used in the paper)
├── jobs/
│   └── new_slurm/                      # SLURM templates for Stage 1 and Stage 3 (user-provided)
└── README.md
```

## Stage 1 — Inversion (HPC)

```bash
python sdxl/tasks/real_editing/scripts/invert_real_image.py \
    --image path/to/photo.jpg \
    --prompt "a portrait of a person" \
    --run_id my_run_001
```

Useful CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `--run_id` | `try_tight_001` | Output folder under `outputs/` |
| `--image` | (required) | Path to the input image |
| `--prompt` | `two people standing together` | Textual description of the image |
| `--num_gd_steps` | `3` | Gradient-descent steps per inversion step |
| `--ipa_scale` | `0.4` | IP-Adapter conditioning strength |

By default the script runs Tight Inversion + IP-Adapter. Pass
`--no_ipa` to disable IP-Adapter (only useful for the pure-DDIM
baseline).

## Stage 2 — SAM mask (local)

```bash
# Interactive: click on the target subject in the matplotlib window
python mask_SAM/segment_with_sam.py \
    --run_dir sdxl/tasks/real_editing/outputs/<RUN_ID> \
    --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
    --image_name reconstruction.png \
    --mode interactive

# Or point/box mode for batch operation; see mask_SAM/segment_with_sam.py --help
```

The mask is saved as `mask_target.png` directly in the run directory.

## Stage 3 — Masked edit (HPC)

```bash
python sdxl/tasks/real_editing/scripts/edit_real_image_masked.py \
    --run_dir sdxl/tasks/real_editing/outputs/my_run_001 \
    --slider_path sdxl/trained_sliders/sliders/general/smile_person/smile_person.safetensors \
    --slider_scale 2.0 \
    --start_noise 800 \
    --output_name edited_smile_v1.png
```

Useful CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `--run_dir` | (required) | Run to edit (must contain `x_t.pt` and `mask_target.png`) |
| `--slider_path` | `sdxl/trained_sliders/sliders/smiling.pt` | Slider LoRA weights |
| `--slider_scale` | `0.4` | Edit strength (typical 1.0-3.0) |
| `--start_noise` | `350` | Apply the slider only on timesteps <= this threshold |
| `--guidance_scale` | `2.0` | CFG scale (do not push above ~3.0) |
| `--feather_radius` | `16` | Gaussian blur radius on the mask edges |
| `--output_name` | `edited_tight_001_v2.png` | Output filename |

Outputs:

- `edited_<name>.png` — 1024x1024 crop decoded directly from the VAE.
- `composite_edited_<name>.png` — the edit pasted into the original
  full-resolution image (the file actually used downstream).

## Implementation notes

- **VAE in float32.** The SDXL VAE is numerically unstable in float16
  and produces black / NaN outputs. The loader (`lib/models/sdxl.py`)
  permanently keeps the VAE in float32 even when the rest of the model
  runs in float16; this avoids the fragile `upcast_vae()` partial path
  used by some versions of diffusers.
- **IP-Adapter is mandatory at edit time.** The inversion is performed
  WITH IP-Adapter as visual conditioning. The editing pass must use the
  same IP-Adapter with the same parameters, otherwise the denoising
  trajectory drifts from the inversion trajectory and the output
  collapses to black.
- **`start_noise`.** Low values (~350) produce subtle, detail-only
  edits. High values (~800) produce more structural edits. Pushing too
  high causes artefacts.
- **`slider_scale`.** Above ~3.0, artefacts appear outside the mask
  even with feathering. Stay between 1.0 and 2.5 for natural-looking
  results.
- **GPU memory.** SDXL UNet (float16) + VAE (float32) + IP-Adapter +
  ViT-H image encoder need ~14-16 GB. The SLURM template asks for 24 GB.

## Pretrained weights

The pipeline downloads the following weights on first use (HuggingFace
cache):

- `stabilityai/stable-diffusion-xl-base-1.0`
- `h94/IP-Adapter`, with
  `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` and
  `models/image_encoder/`.

If running offline, make sure both are present in the local cache
before submitting the job.

## References

- Tight Inversion: Kadosh et al., 2025, arXiv:2502.20376
- IP-Adapter: Ye et al., 2023, arXiv:2308.06721
