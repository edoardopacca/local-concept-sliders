# `sdxl/tasks/masked_lora/`

SDXL implementation of the **external mask-guided application** described
in §4 of the paper. This is the SDXL counterpart of
[`flux/tasks/masked_lora/`](../../../flux/tasks/masked_lora/); the
methodology is identical, only the backbone-specific details differ
(latent layout, LoRA wiring, default hyperparameters).

Given a generated image and a user-supplied segmentation mask, the
pipeline re-generates the same scene while applying one or more Concept
Sliders only inside the masked region, using the per-step blend in
eq. (1) of the paper.

## Pipeline

```
Phase 1 (HPC)              Phase 2 (local, interactive)        Phase 3 (HPC)
generate base + meta   ->  SAM mask on base.png             -> masked re-generation
                           (also init_latents.pt for SDXL)
```

- **Phase 1** runs an unedited SDXL pass on a fixed seed and saves the
  base image, the initial latents and the metadata (seed, scheduler
  config, prompt) needed to reproduce the trajectory in phase 3.
- **Phase 2** runs SAM locally to produce one or more binary masks; the
  scripts live under [`mask_SAM/`](../../../mask_SAM/) at the repo root.
- **Phase 3** reruns the denoising from the same seed and applies the
  slider(s) only inside the masked region(s) via the per-step blend
  ``eps_pred = (1 - sum_i mask_i) * eps_base + sum_i (mask_i * eps_styled_i)``.

The single-mask form is the operator of eq. (1); the multi-mask form
extends it to several disjoint masks (cost ``1 + N`` UNet forwards per
step) and to several sliders inside the same mask (additive composition
of the LoRA deltas, no extra forward).

## Origin

The masked-edit operator and its multi-mask / multi-slider extension are
original to this work. The slider LoRA stack (`LoRANetwork`,
`apply_to`, `set_lora_slider`) lives in [`sdxl/core/`](../../core/) and
is adapted from the upstream Concept Sliders codebase
([rohitgandikota/sliders](https://github.com/rohitgandikota/sliders),
MIT licensed); the per-step blending and the ExitStack-based multi-LoRA
composition are added here on top.

## Layout

```
sdxl/tasks/masked_lora/
├── scripts/
│   ├── 01_generate_base.py        # Phase 1: base image + init_latents + metadata
│   └── 03_masked_edit.py          # Phase 3: multi-mask multi-LoRA masked edit
├── jobs/
│   └── new_slurm/                 # SLURM templates used for the paper
├── runs/                          # per-evaluation run directories (git-ignored)
├── requirements-local-sam.txt     # deps for the local SAM stage
└── README.md                      # this file
```

## Usage

All commands are launched from the repository root. The SDXL backbone is
`stabilityai/stable-diffusion-xl-base-1.0` and the slider format
accepted is either `.pt` (Concept Sliders native) or `.safetensors`.

### Phase 1: base image

```bash
python sdxl/tasks/masked_lora/scripts/01_generate_base.py \
       --prompt "a couple at a Parisian cafe" \
       --seed 1001 \
       --run_id eval_smile_person_01 \
       --output_root sdxl/tasks/masked_lora/runs
```

Writes `runs/eval_smile_person_01/base.png`, `init_latents.pt`
(reused by phase 3 for bit-identical reproduction) and `metadata.json`.

### Phase 2: SAM mask

Run interactively on a machine with a display (typically a local
workstation):

```bash
python mask_SAM/segment_with_sam.py \
       --run_dir sdxl/tasks/masked_lora/runs/eval_smile_person_01 \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --image_name base.png --mode interactive
```

For batch operation see `mask_SAM/choose_masks.py` (one mask per run).

### Phase 3: masked edit

Single mask, single slider:

```bash
python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
       --run_dir sdxl/tasks/masked_lora/runs/eval_smile_person_01 \
       --slider_path sdxl/trained_sliders/sliders/general/smile_person/smile_person.safetensors \
       --slider_scale 2.0 \
       --start_noise 700
```

**Multi-mask multi-slider** (paper §4):

```bash
python sdxl/tasks/masked_lora/scripts/03_masked_edit.py \
       --run_dir sdxl/tasks/masked_lora/runs/two_subjects_<id> \
       --mask_names    mask_man.png mask_woman.png \
       --slider_paths  sdxl/trained_sliders/sliders/general/smile_person/smile_person.safetensors \
                       sdxl/trained_sliders/sliders/general/age_person/age_person.safetensors   \
                       sdxl/trained_sliders/sliders/general/smile_person/smile_person.safetensors \
                       sdxl/trained_sliders/sliders/general/age_person/age_person.safetensors   \
       --slider_to_mask 0 0 1 1 \
       --slider_scales  1.0 1.0 -1.0 -1.0
```

In multi-mode the script loads N distinct `LoRANetwork` instances (one
per slider), all attached to the same UNet. Inside the per-mask
denoising step only the relevant subset of networks is activated via
`ExitStack`; PEFT-style additive composition is replicated by the
nested LoRA wrappers.

## Evaluation

The six-concept eval reported in the paper appendix
(`tab:appendix:results-masked-sdxl`) is launched in three steps, with
matching SLURM templates under `jobs/new_slurm/`:

```bash
# Phase 1: generate 20 base images per concept.
sbatch sdxl/tasks/masked_lora/jobs/new_slurm/eval_smile_person_phase1.slurm
# (same for age_person, curlyhair, daynight, furlength, painterly)

# Phase 2: SAM masks on the 20 base images of each concept (run locally).
python mask_SAM/choose_masks.py \
       --runs_root sdxl/tasks/masked_lora/runs \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --run_ids $(ls sdxl/tasks/masked_lora/runs/ | grep '^eval_')

# Phase 3: masked edit at three scales per concept.
sbatch sdxl/tasks/masked_lora/jobs/new_slurm/eval_smile_person_phase3.slurm

# Aggregate metrics over the 20 runs of each concept.
sbatch sdxl/tasks/masked_lora/jobs/new_slurm/run_eval_metrics.slurm
```

Outputs are written to `runs/eval_<concept>_<seed>/`
(`base.png`, `mask_target.png`, `edited_<concept>_s{1,2,3}.png`,
`metadata.json`, `eval_metrics_s{1,2,3}.json`) and aggregated by
`metrics/eval_masked.py` under `metrics/results_sdxl_masked/<concept>/`.

The exact evaluation prompts and seeds are not duplicated in the
paper appendix; they are committed in this repository as part of the
SLURM templates under `jobs/new_slurm/` and the eval scripts under
`scripts/`.

## Useful parameters

| Flag | Default | Meaning |
|---|---|---|
| `--steps` (phase 1) | 50 | DDIM denoising steps |
| `--guidance_scale` (phase 1) | 5.0 | Classifier-free guidance scale |
| `--height`, `--width` | 1024 each | Native SDXL resolution |
| `--dtype` | float16 | UNet dtype; VAE is forced to float32 by the loader |
| `--start_noise` (phase 3) | 700 | Timestep threshold: for ``t > start_noise`` the styled forward is skipped and the output equals the base. Matches §4 (the blend activates after roughly the first 25% of the trajectory). |
| `--rank` (phase 3) | 4 | LoRA rank used by `LoRANetwork` |
| `--skip_metrics` | flag | Skip the per-run in-script metrics in favour of the dedicated `metrics/eval_masked.py` aggregator (used by the SLURM eval). |

## Notes

- The cost per image is roughly ``(1 + N) x`` the cost of a plain SDXL
  generation, where N is the number of disjoint masks.
- The pipeline assumes disjoint masks in multi-mask mode. If two masks
  share latent pixels, the script emits a warning during phase 3 and the
  overlapping pixels accumulate the deltas from both regions.
- Phase 1 saves `init_latents.pt` so phase 3 can rebuild the exact same
  starting latents, which makes the ``all sliders off`` run in phase 3
  bit-identical to the base image. The flux pipeline regenerates the
  initial latents from the seed instead, because the packed/unpacked
  layouts make on-disk caching more fragile on Flux.
- The `runs/` directory is git-ignored and is populated as soon as the
  three-stage pipeline is launched. The qualitative samples used in the
  paper figures are produced by re-running the same pipeline with the
  prompts and seeds documented in `jobs/new_slurm/`.
