# training_local_concept_sliders

Self-contained training folder for **spatial Concept Sliders** — LoRA sliders
anchored to a specific subject (e.g. `"man"`) rather than a generic one
(e.g. `"person"`), so that at inference time the slider's effect localizes
naturally to that subject, without any change to the loss, architecture or
inference procedure.

This folder is completely self-contained: it owns its own copy of the training
script and utility modules, its own configs, prompt files, SLURM jobs, logs
and outputs. Nothing outside this folder is modified.

## Methodology

The training pipeline is **byte-identical** to Gandikota et al. (ECCV 2024) —
`trainscripts/textsliders/train_lora_xl.py` in this repo, which itself is the
authors' official SDXL training script. The only code modification is a
`sys.path` insertion at the top of `scripts/train.py` so that the script can
be run from any working directory.

Our contribution is **entirely in the prompt YAMLs**. The standard
"smile" slider uses `target: "person"`; we anchor the same concept direction
to a specific subject by using `target: "man"` (or `"woman"`, etc.). The
subject choice propagates through the LoRA conditioning and yields a slider
whose effect is naturally spatially localized to that subject.

See the top-level `CONTEXT_FOR_AI_TRAINING_SETUP.md` in this folder for the
full theoretical motivation.

## Folder layout

```
training_local_concept_sliders/
├── README.md                     — this file
├── CONTEXT_FOR_AI_TRAINING_SETUP.md — project brief / motivation
├── scripts/
│   ├── train.py                  — copy of trainscripts/textsliders/train_lora_xl.py
│   │                               (only sys.path header added)
│   ├── generate_with_sliders.py  — copy of exp_generation/generate_with_sliders.py
│   │                               (sys.path + .safetensors loader + default save path)
│   ├── lora.py                   — byte-identical copy of upstream
│   ├── prompt_util.py            — byte-identical copy of upstream
│   ├── config_util.py            — byte-identical copy of upstream
│   ├── model_util.py             — byte-identical copy of upstream
│   ├── train_util.py             — byte-identical copy of upstream
│   └── debug_util.py             — byte-identical copy of upstream
├── configs/
│   └── <exp_name>.yaml           — training config (paths, optimizer, iters, ...)
├── prompts/
│   └── <exp_name>.yaml           — 4-prompt slider definition
├── jobs/
│   ├── train_<exp_name>.slurm    — SLURM submission script for training
│   └── generate_<exp_name>.slurm — SLURM submission script for inference
├── logs/                         — SLURM stdout/stderr (%x_%j.out / .err)
└── outputs/                      — trained LoRA weights (*.safetensors) +
                                    generated test images (generations_*/)
```

## The 4-prompt YAML

Each experiment is defined by a single YAML under `prompts/`. Field semantics
(from upstream `prompt_util.py`):

| field           | meaning                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `target`        | subject the LoRA is conditioned on; LoRA is *active* on this prompt     |
| `positive`      | positive pole of the concept direction                                  |
| `unconditional` | negative pole of the concept direction (NOT the empty CFG unconditional)|
| `neutral`       | base point used by the loss; conventionally equal to `target`           |
| `action`        | `"enhance"` or `"erase"` — must be `"enhance"` for our experiments      |
| `guidance_scale`| scale for the learned direction in the loss (NOT CFG guidance)          |
| `resolution`    | training resolution (we use 512)                                        |
| `batch_size`    | per-step batch size                                                     |

The semantic direction learned by the LoRA is `(positive - unconditional)`,
applied as a shift from `neutral`, and active only when conditioned on
`target`.

### A note on `guidance_scale` vs `guidance`

`PromptSettings` in `prompt_util.py` expects the field `guidance_scale:`
(with underscore). Some upstream example YAMLs use the shorter `guidance:` —
this is silently ignored by pydantic and falls back to the default. We
always use the correct name.

## Current experiments

| Experiment  | target | positive        | unconditional        | iterations | rank | alpha |
|-------------|--------|-----------------|----------------------|------------|------|-------|
| `smile_man` | `man`  | `man, smiling`  | `man, not smiling`   | 1000       | 4    | 1.0   |

The `smile_man` run is our baseline: minimal single-entry YAML,
single-subject anchor, no ethnicity/age variants. Its purpose is to validate
the pipeline end-to-end and to produce a first slider we can compare against
the classic `smile_person` baseline.

## How to run an existing experiment on the HPC

From the repo root on the HPC (`/home/<your-username>/FERT_PROJECT/local-concept-sliders`):

```bash
sbatch training_local_concept_sliders/SDXL_train/jobs/train_smile_man.slurm
```

The SLURM script:

1. Activates the project venv (`~/Linux4HPC/venvs/sliders`).
2. Sets `HF_HOME` / `HF_HUB_CACHE` to the shared offline HuggingFace cache.
3. Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so the job never
   attempts network access on compute nodes.
4. Invokes `scripts/train.py` with the experiment config.

Logs appear in `training_local_concept_sliders/SDXL_train/logs/<jobname>_<jobid>.out|.err`.

The trained LoRA is saved under
`outputs/<name>_alpha<alpha>_rank<rank>_<training_method>/<name>_alpha<alpha>_rank<rank>_<training_method>_last.safetensors`
(plus intermediate `*_{500,1000,...}steps.safetensors` checkpoints every
`save.per_steps` steps). The `_alpha..._rank..._<method>` suffix is
appended by `train.py` so that different hyperparameter runs never
overwrite each other. For `smile_man` with rank=4 / alpha=1.0 /
noxattn that resolves to:

```
outputs/smile_man_alpha1.0_rank4_noxattn/smile_man_alpha1.0_rank4_noxattn_last.safetensors
```

## How to try an existing slider (inference)

After the training job finishes, the slider can be swept over a range
of scales on a fixed seed — this is the central qualitative test of
whether the slider works:

```bash
sbatch training_local_concept_sliders/SDXL_train/jobs/generate_smile_man.slurm
```

The inference script (`scripts/generate_with_sliders.py`) monkey-patches
`StableDiffusionXLPipeline.__call__` to inject the LoRA only at
timesteps `<= start_noise` (default 700 out of 1000), so the image
layout is frozen from the base-model denoising and only the later
steps see the slider. Output is saved under
`outputs/<slider_name>/generations_<tag>/`:

  - `scale_<s>.png` for each scale
  - `grid.png`      — side-by-side comparison strip

The `generate_smile_man.slurm` example sweeps over `scales = [-2, -1,
0, 1, 2]` so you see the slider in both directions (amplify / reverse)
plus the exact `scale=0` baseline. Negative scales are the "anti"
direction of the learned concept.

**Localization test.** To check that the `smile_man` slider is
anchored to "man" and not to generic subjects, rerun the inference
job on prompts that contain *other* subjects (a woman, a mixed scene)
and compare: the slider should leave those other subjects unchanged.
See the commented section at the bottom of `jobs/generate_smile_man.slurm`
for ready-to-paste variants.

## How to add a new experiment

For an experiment called `<exp>`:

1. Create `prompts/<exp>.yaml` with the 4-prompt definition (one or more
   YAML list entries — the training loop picks from the list uniformly at
   random each step, which approximates the preservation set of Eq. 8 in
   SGD form).
2. Create `configs/<exp>.yaml` by copying `configs/smile_man.yaml` and
   editing only `prompts_file`, `save.name`, and `save.path` (the rest is
   shared across all experiments to keep results comparable).
3. Create `jobs/train_<exp>.slurm` by copying `jobs/train_smile_man.slurm`
   and changing `--job-name`, `--config_file`, `--name` accordingly.
4. Commit, push, `sbatch`.

That's it. No code changes required for a new experiment.

## Hyperparameters (shared across experiments)

Kept identical to upstream `trainscripts/textsliders/data/config-xl.yaml`:

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- LoRA: type `c3lier`, rank `4`, alpha `1.0`, training method `noxattn`
- Optimizer: `AdamW`, lr `2e-4`, constant schedule
- Iterations: `1000`
- Noise scheduler: `ddim`, `max_denoising_steps=50`
- Precision: `bfloat16`
- `use_xformers: true`

These are deliberately **not** exposed per-experiment: any hyperparameter
change would invalidate the comparison with the upstream baseline, which is
what we're measuring localization against.

## Compute budget

Indicative on Bocconi HPC `stud` partition, 1 GPU, SDXL, rank 4, 1000 iters,
resolution 512:

- Wall time: ~45–75 minutes
- GPU memory: ~18–22 GB
- The SLURM script requests 1h30 wall time, 24 GB RAM, 4 CPUs, 1 GPU.

## Known upstream behaviors (unchanged)

Kept as-is per project rule "same training pipeline as upstream":

- `lora.py` uses a hardcoded `DEFAULT_TARGET_REPLACE = ["Attention"]`, so the
  `modules` list computed for the `c3lier` branch of `config_util` is never
  actually passed through. Conv layers are therefore NOT LoRA-adapted.
- `train_util.predict_noise_xl` computes `rescale_noise_cfg(...)` but
  returns the un-rescaled `guided_target`; the rescale is dead code.

Both behaviors are present in the upstream training script. We do not modify
them.

## Reproducibility

Each experiment is fully defined by its `prompts/*.yaml`, `configs/*.yaml`,
`jobs/*.slurm` triple plus the exact repo commit. The training script and
utility modules are self-contained copies under `scripts/`, so a future
repo-wide refactor of `trainscripts/textsliders/` cannot silently change the
training behavior of our experiments.
