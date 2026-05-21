# `sdxl/tasks/baseline/`

Exploratory task: generate images with SDXL and apply a trained slider
globally over a sweep of scales. Used during development as a sanity
check on each newly trained slider (does the direction work? what is
the dose-response curve?). Not directly reported in the paper; the
`outputs/` directory is populated locally by each run.

The main entrypoint reuses
[`sdxl/trained_sliders/training/scripts/generate_with_sliders.py`](../../trained_sliders/training/scripts/generate_with_sliders.py)
via `scripts/generate_with_sliders.py` (a thin re-export). Each job
produces one PNG per scale plus a side-by-side grid.

## Origin

The generation script is adapted from the upstream Concept Sliders
codebase ([rohitgandikota/sliders](https://github.com/rohitgandikota/sliders),
MIT licensed).

## Layout

```
sdxl/tasks/baseline/
├── scripts/
│   └── generate_with_sliders.py    # thin entrypoint re-export
├── jobs/
│   └── new_slurm/                  # SLURM templates for baseline sweeps
└── outputs/                        # one subdir per (slider, scene); populated locally
```

## Usage

```bash
sbatch sdxl/tasks/baseline/jobs/new_slurm/<your-template>.slurm
```

Customise inside the SLURM template:

- `SLIDER` — path to the trained slider (`.pt` or `.safetensors`)
- `PROMPT` — generation prompt
- `SCALES` — sweep range
- `SAVE_PATH` — output directory

Output: `outputs/<run>/scale_<s>.png` per scale plus `grid.png` for the
side-by-side comparison.
