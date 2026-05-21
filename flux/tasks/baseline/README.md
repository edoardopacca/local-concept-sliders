# `flux/tasks/baseline/`

Exploratory task: generate images with Flux.1-dev and apply a trained
slider globally over a sweep of scales. Used during development as a
sanity check on each newly trained slider (dose-response, baseline
behaviour, ethnicity / age coverage). Not directly reported in the
paper; the `outputs/` directory is populated locally by each run.

The generation script (`scripts/generate_flux_slider.py`) loads the Flux
pipeline, applies one or more LoRA adapters and iterates over a list of
(prompt, seed, scale) combinations.

## Origin

The script is a thin wrapper around the Flux pipeline implementation in
[`flux/core/custom_flux_pipeline.py`](../../core/custom_flux_pipeline.py)
(adapted from diffusers) and the LoRA stack in
[`flux/core/lora.py`](../../core/lora.py) (adapted from the upstream
Concept Sliders codebase).

## Layout

```
flux/tasks/baseline/
├── scripts/
│   ├── generate_flux.py            # unedited Flux baseline (no slider)
│   └── generate_flux_slider.py     # Flux + one or more sliders, scale sweep
├── jobs/
│   └── new_slurm/                  # SLURM templates for the baseline sweeps
└── outputs/                        # per-run subdirectories (populated locally)
```

## Usage

```bash
sbatch flux/tasks/baseline/jobs/new_slurm/<your-template>.slurm
```

Customise inside the SLURM template:

- `LORA_DIR` — path to the slider directory under `flux/trained_sliders/sliders/`
- `PROMPT` — generation prompt
- `SCALES` — sweep range
- `SEEDS` — seed list
- `SAVE_DIR` — output directory

Output: one PNG per (prompt, seed, scale) combination under `SAVE_DIR`.
