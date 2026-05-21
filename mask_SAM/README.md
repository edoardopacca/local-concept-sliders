# `mask_SAM/`

SAM (Segment Anything) wrappers used at **Phase 2** of the mask-guided
pipeline: produce a binary mask (`mask_target.png`) on the base image
generated at Phase 1, then drop it into the relevant run directory for
Phase 3.

The three scripts here are thin wrappers around the upstream
[Segment Anything](https://github.com/facebookresearch/segment-anything)
package (Apache-2.0, Kirillov et al., ICCV 2023). The SAM checkpoint
(`sam_vit_h_4b8939.pth`, ~2.4 GB) is downloaded once into
`mask_SAM/checkpoints/` and git-ignored.

## Layout

```
mask_SAM/
├── segment_with_sam.py        # single-image SAM CLI (point / box / json / interactive)
├── choose_masks.py            # batch mode: two clicks + preview + choose
├── place_masks.py             # copy masks from masks_output/ into the matching run dirs
├── checkpoints/               # SAM weights (downloaded on first use, git-ignored)
└── README.md                  # this file
```

## When to use which script

- **`segment_with_sam.py`** — single image, single mask. Useful for the
  real-editing pipeline (`sdxl/tasks/real_editing/`) where each run is
  individual.
- **`choose_masks.py`** — batch over several run directories at once.
  Loads SAM once and asks for two clicks per image; previews the two
  candidate masks side by side and lets you pick the better one (or
  retry). Used for the per-concept masked-edit evaluation
  (`sdxl/tasks/masked_lora/`, `flux/tasks/masked_lora/`).
- **`place_masks.py`** — administrative helper. If the masks were
  produced separately (e.g. on a different machine) and shipped as
  `masks_output/eval_*/mask_target.png`, this script copies each one
  into the matching run directory.

## Installation

```bash
python3 -m venv .venv-sam && source .venv-sam/bin/activate
pip install torch torchvision numpy pillow matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git

mkdir -p mask_SAM/checkpoints
curl -L -o mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

For the masked_lora tasks under SDXL the dependency list is also
mirrored in [`sdxl/tasks/masked_lora/requirements-local-sam.txt`](../sdxl/tasks/masked_lora/requirements-local-sam.txt).

## Usage examples

### Single image, single mask

```bash
# Interactive: click on the target in the matplotlib window, then close it.
python mask_SAM/segment_with_sam.py \
       --run_dir path/to/run \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --image_name base.png \
       --mode interactive

# Point mode: provide the click coordinates directly.
python mask_SAM/segment_with_sam.py \
       --run_dir path/to/run \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --image_name base.png \
       --mode point --point_x 256 --point_y 80

# Box mode: provide an axis-aligned bounding box [x1 y1 x2 y2].
python mask_SAM/segment_with_sam.py \
       --run_dir path/to/run \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --image_name base.png \
       --mode box --box 20 10 492 240
```

Output: `path/to/run/mask_target.png` (filename overridable via
`--output_name`) plus `mask_meta.json` recording the prompt mode.

### Batch over many runs

```bash
python mask_SAM/choose_masks.py \
       --runs_root flux/tasks/masked_lora/runs \
       --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
       --run_ids $(ls flux/tasks/masked_lora/runs/ | grep '^eval_')
```

For each run the script asks for two clicks, opens the two candidate
masks side by side in the system image viewer, and reads a single
character on stdin (`1`, `2`, or `r` to retry). The chosen mask is
saved as `mask_target.png` in the run directory.

### Place pre-existing masks into run directories

```bash
python mask_SAM/place_masks.py \
       --masks_dir masks_output \
       --runs_root flux/tasks/masked_lora/runs
```

Use this when the masks were already produced separately and need to
be moved into the matching run directories before the next phase.

## Notes

- The masks produced here are always **pixel-resolution binary PNGs**.
  Downstream tasks downsample them to the latent grid of their backbone
  (8x for SDXL, 16x for Flux). Drawing the mask at the full image
  resolution keeps the masking pipeline independent of the backbone.
- The interactive scripts use `matplotlib.pyplot.ginput` and the OS
  image viewer (`open` / `xdg-open` / `start`); a display server must
  be available.
- `choose_masks.py` saves temporary preview files
  (`_*_preview.png`) inside the run directory and removes them at the
  end of each iteration.
