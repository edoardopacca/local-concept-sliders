# `sdxl/trained_sliders/training/`

Training framework for SDXL Concept Sliders. The sliders trained here
are the inputs to every SDXL evaluation in the paper:

- The six general (un-anchored) sliders evaluated under the mask-guided
  pipeline of §4 — ``age_person``, ``smile_person``, ``curlyhair_person``,
  ``furlength_pet``, ``daynight``, ``painterly`` — reported in the
  appendix table `tab:appendix:results-masked-sdxl`.

The training stage is **not** a contribution of this work. We use the
upstream Concept Sliders recipe (Gandikota et al., 2023) unchanged; the
paper's contribution is the inference-time mask-guided localisation
built on top of these adapters (§4).

## Origin

The training script is adapted from the upstream Concept Sliders
codebase ([rohitgandikota/sliders](https://github.com/rohitgandikota/sliders),
MIT licensed). The only structural changes are:

- the LoRA stack lives in [`sdxl/core/`](../../core/) (so it can be
  shared by the editing tasks downstream);
- a `sys.path` insertion at the top of each training script.

The training configurations (`configs/`) and prompt sets (`prompts/`)
are original to this work but use the upstream Concept Sliders prompt
schema unchanged.

The file `scripts/train_with_preservation.py` is **obsolete**: it was
an alternative training route developed during an earlier (subject-
anchored) iteration of the project that has been removed from the
paper. It is kept in-tree for archival reasons only; do not invoke it
for new runs.

## Layout

```
sdxl/trained_sliders/training/
├── scripts/
│   ├── train.py                        # upstream Concept Sliders training
│   └── train_with_preservation.py      # OBSOLETE — see Origin
├── configs/                            # one YAML per slider
├── prompts/
│   ├── new_prompt/                     # prompt sets used for the paper runs
│   └── old_prompt/                     # earlier prompt iterations + upstream copies
├── jobs/
│   └── new_slurm/                      # SLURM templates used for the paper
├── requirements-sdxl.lock              # pinned dependencies snapshot
└── README.md                           # this file
```

The slider checkpoints produced here are saved under
[`sdxl/trained_sliders/sliders/`](../sliders/) (`.pt` /
`.safetensors`). The final sliders used by the paper runs are committed
under `general/` via a whitelist in `.gitignore`; any other checkpoint
produced by training is git-ignored and consumed by the downstream
tasks locally.

## How a slider is defined

Each slider is fully described by three files:

1. A **prompt YAML** under `prompts/new_prompt/<slider>.yaml`. Every YAML
   entry has four fields used by the Concept Sliders objective:

   | field | meaning |
   |---|---|
   | `target` | subject the LoRA is conditioned on; the LoRA is active on this prompt. |
   | `positive` | positive pole of the concept direction. |
   | `unconditional` | negative pole of the concept direction (not the empty CFG unconditional). |
   | `neutral` | base point used by the loss; conventionally equal to `target`. |
   | `action` | `enhance` for every entry used in the paper. |
   | `guidance_scale` | scale of the learned direction inside the loss (not CFG guidance). |
   | `resolution` / `batch_size` | training resolution / per-step batch size. |

   The semantic direction learned is `(positive - unconditional)`,
   applied as a shift from `neutral`, and active only when conditioned
   on `target`. For the un-anchored sliders used in the paper the
   `target` field is a generic subject ("a person", "a pet", or a
   scene) so the slider fires on any matching subject.

2. A **config YAML** under `configs/<slider>.yaml` that points to the
   prompt YAML and sets the model id, the LoRA hyperparameters (rank,
   alpha, training method) and the optimisation settings. The six
   sliders reported in the paper all use the same hyperparameters
   (rank 4, alpha 1.0, training method `noxattn`, AdamW with lr 2e-4
   constant, 1000 iterations, bfloat16) — the same defaults as the
   upstream Concept Sliders recipe.

3. A **SLURM template** under `jobs/new_slurm/train_<slider>.slurm` that
   activates the training environment and invokes
   `scripts/train.py --config_file <slider>.yaml`.

The prompt-vs-config split keeps the slider definition decoupled from
the optimisation knobs (which are held constant across runs).

## Usage

From the repository root:

```bash
# Train one slider directly:
python sdxl/trained_sliders/training/scripts/train.py \
       --config_file sdxl/trained_sliders/training/configs/age_person_sdxl_v1.yaml

# Or via the bundled SLURM template:
sbatch sdxl/trained_sliders/training/jobs/new_slurm/train_age_person_sdxl_v1.slurm
```

The output is saved under
`outputs/<slider>_alpha<alpha>_rank<rank>_<method>/`
(local to the training folder), with intermediate checkpoints written
every `save.per_steps` steps. The trained slider then needs to be moved
to `sdxl/trained_sliders/sliders/` to be picked up by the evaluation
pipelines.

A typical training run on a single A100-class GPU takes ~45-75 min for
1000 iterations at 512x512.

## Adding a new slider

For a new slider called `<exp>`:

1. Create `prompts/new_prompt/<exp>.yaml` with the four-prompt
   definition. Several entries can be listed and the training loop
   samples one of them uniformly at random per step.
2. Create `configs/<exp>.yaml` by copying an existing config and
   editing only `prompts_file`, `save.name` and `save.path`.
3. Create `jobs/new_slurm/train_<exp>.slurm` by copying an existing
   SLURM template and editing the `--job-name` and the `--config_file`.

No code changes are required for a new slider.

## Reproducibility

Each slider is fully defined by the triple
``prompts/new_prompt/<exp>.yaml`` + ``configs/<exp>.yaml`` +
``jobs/new_slurm/train_<exp>.slurm`` plus the repo commit. The
`scripts/` directory is a self-contained copy of the upstream
Concept Sliders training files, so future refactors of `sdxl/core/`
will not silently change the behaviour of past runs.
