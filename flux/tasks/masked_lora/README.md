# `flux/tasks/masked_lora/`

Flux.1-dev implementation of the **external mask-guided application**
described in §4 of the paper. This is the main contribution of the
project on the Flux backbone, and it mirrors the SDXL pipeline under
`sdxl/tasks/masked_lora/`.

Given a generated image and a user-supplied segmentation mask, the
pipeline re-generates the same scene while applying one or more Concept
Sliders only inside the masked region, using the per-step blend in
eq. (1) of the paper.

## Pipeline

```
Phase 1 (HPC)           Phase 2 (local, interactive)        Phase 3 (HPC)
generate base + meta -> SAM mask on base.png             -> masked re-generation
```

- **Phase 1** runs an unedited Flux pass on a fixed seed and saves the
  base image together with the metadata (seed, scheduler config, prompt)
  needed to reproduce the trajectory in phase 3.
- **Phase 2** runs SAM locally to produce one or more binary masks; the
  scripts live under `mask_SAM/` at the repo root.
- **Phase 3** reruns the denoising from the same seed and applies the
  slider(s) only inside the masked region(s) via the per-step blend
  ``v_pred = (1 - sum_i mask_i) * v_base + sum_i (mask_i * v_styled_i)``.
  The base and styled passes share the same latent at every step, so
  there is no seam at the mask boundary. See
  `scripts/03_masked_edit.py` for the implementation.

The single-mask form is the operator of eq. (1); the multi-mask form
extends it to several disjoint masks (cost ``1 + N`` forwards per step)
and to several sliders inside the same mask (additive composition of the
LoRA deltas, no extra forward).

## Origin

The masked-edit operator and the multi-mask / multi-slider extension are
original to this work. The Flux base pipeline is the upstream diffusers
implementation; the mask packing into Flux's 2x2 token grid is the only
backbone-specific adaptation needed.

The slider conversion (kohya-ss `.pt` -> PEFT `.safetensors`) and the
SDPA compatibility shim for torch 2.4 / diffusers >= 0.36 are reused
from [`flux/tasks/shop_concept/`](../shop_concept/) without duplication.

## Layout

```
flux/tasks/masked_lora/
├── scripts/
│   ├── 01_generate_base.py     # Phase 1: base image + metadata
│   └── 03_masked_edit.py       # Phase 3: multi-path mask-guided blend
├── jobs/
│   └── new_slurm/              # SLURM templates used for the paper
├── outputs/                    # one-shot generation outputs (populated locally)
├── runs/                       # per-evaluation run directories (git-ignored)
└── README.md                   # this file
```

Phase 2 (SAM) is shared with the SDXL pipeline and lives under
`mask_SAM/` at the repository root.

## Usage

All commands are launched from the repository root. The Flux backbone is
loaded as `black-forest-labs/FLUX.1-dev`; the slider format accepted is
either `.pt` (auto-converted to PEFT safetensors on the fly and cached in
`_peft_cache/`) or `.safetensors` (loaded as is).

### Phase 1: base image

```bash
python flux/tasks/masked_lora/scripts/01_generate_base.py \
       --prompt "a woman and a man sitting at a Parisian cafe" \
       --seed 1001 \
       --run_id eval_smile_person_01 \
       --output_root flux/tasks/masked_lora/runs
```

Writes `runs/eval_smile_person_01/base.png` and `metadata.json`.

### Phase 2: SAM mask

Run interactively on a machine with a display (typically a local
workstation):

```bash
python mask_SAM/segment_with_sam.py \
       --run_dir flux/tasks/masked_lora/runs/eval_smile_person_01 \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --image_name base.png --mode interactive
```

For batch operation over many runs see `mask_SAM/choose_masks.py`. The
masks are written into the same run directory.

### Phase 3: masked edit

Single mask, single slider, with an explicit scale:

```bash
python flux/tasks/masked_lora/scripts/03_masked_edit.py \
       --run_dir flux/tasks/masked_lora/runs/eval_smile_person_01 \
       --slider_path flux/trained_sliders/sliders/general/smile/slider_0.pt \
       --slider_scale 2.0
```

Single mask, single slider, multi-scale **sweep** (one output per scale,
Flux is loaded only once):

```bash
python flux/tasks/masked_lora/scripts/03_masked_edit.py \
       --run_dir flux/tasks/masked_lora/runs/eval_smile_person_01 \
       --slider_path flux/trained_sliders/sliders/general/smile/slider_0.pt \
       --slider_scales 1.0 2.0 3.0
```

**Multi-mask multi-slider** (paper §4):

```bash
python flux/tasks/masked_lora/scripts/03_masked_edit.py \
       --run_dir flux/tasks/masked_lora/runs/two_subjects_<id> \
       --mask_names    mask_man.png mask_woman.png \
       --slider_paths  flux/trained_sliders/sliders/general/smile/slider_0.pt \
                       flux/trained_sliders/sliders/general/age/slider_0.pt   \
                       flux/trained_sliders/sliders/general/smile/slider_0.pt \
                       flux/trained_sliders/sliders/general/age/slider_0.pt   \
       --slider_to_mask 0 0 1 1 \
       --slider_scales  1.0 1.0 -1.0 -1.0
```

In this example the man gets ``+smile +age`` and the woman gets
``-smile -age``; the same slider file is loaded several times as
distinct PEFT adapters so the same concept can carry a different sign on
each region.

## Evaluation

The 20-image per-concept evaluation reported in the paper appendix is
launched in three steps, all with matching SLURM templates under
`jobs/new_slurm/`:

```bash
# Phase 1: generate 20 base images for each concept (Flux pipeline).
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_smile_person_phase1.slurm
# (same for age_person, curlyhair, daynight, furlength, painterly)

# Phase 2: produce the SAM masks for each run (run locally).
python mask_SAM/choose_masks.py \
       --runs_root flux/tasks/masked_lora/runs \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --run_ids $(ls flux/tasks/masked_lora/runs/ | grep '^eval_')

# Phase 3: masked edit at 3 scales per concept.
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_smile_person_phase3.slurm

# Localization metrics over the 20 runs of each concept.
sbatch flux/tasks/masked_lora/jobs/new_slurm/run_eval_metrics.slurm
```

Outputs are written to `runs/eval_<concept>_<seed>/`
(`base.png`, `mask_target.png`, `edited_<concept>_s{1,2,3}.png`,
`metadata.json`, `eval_metrics_s{1,2,3}.json`) and aggregated by
`metrics/eval_masked.py` under
`metrics/results_flux_masked/<concept>/`.

The exact evaluation prompts and seeds are not duplicated in the
paper appendix; they are committed in this repository as part of the
SLURM templates under `jobs/new_slurm/` and the eval scripts under
`scripts/`.

## Useful parameters

| Flag | Default | Meaning |
|---|---|---|
| `--steps` (phase 1) | 30 | Flow-matching steps |
| `--guidance_scale` (phase 1) | 3.5 | Flux distilled guidance |
| `--height`, `--width` | 1024 each | Native Flux resolution |
| `--max_sequence_length` | 256 | T5 prompt length cap |
| `--edit_start_step` | 8 | First step at which the masked blend becomes active (matches §4: roughly the first 25% of the trajectory runs without LoRA) |
| `--lora_fill_rank` | 16 | Rank used when several sliders need to be aligned |
| `--cache_dir` | `flux/tasks/masked_lora/_peft_cache` | PEFT-converted slider cache |

## Notes

- The cost per image is roughly ``(1 + N) x`` the cost of an unconditioned
  Flux generation, where N is the number of disjoint masks.
- The pipeline assumes disjoint masks in multi-mask mode. If two masks
  share tokens, the script emits a warning during phase 3 and the
  overlapping tokens accumulate the deltas from both regions.
- Phase 1 does not save the initial latents to disk: on Flux they are
  regenerated from the seed inside phase 3, which avoids shape
  mismatches between packed and unpacked latent layouts.
- The masks committed under `runs/eval_*/mask_target.png` are the same
  ones used at inference and for the localization metrics; the
  evaluation is end-to-end reproducible given the trained sliders.
