# `sdxl/tasks/masked_lora_editing/`

**Legacy real-image editing prototype, superseded.** The first
real-image masked-edit pipeline implemented in the project: an SD-1.4
version followed by an SDXL port, both using DDIM inversion on the
input photograph and then the masked LoRA blend at edit time.

Superseded by [`sdxl/tasks/real_editing/`](../real_editing/), which
uses **Tight Inversion + IP-Adapter** for the inversion and the same
masked-edit operator on the SDXL side. The Tight-Inversion pipeline is
the one mentioned in §1 of the paper; this folder is kept only for
reference (early ablation results) and is not exercised by any of the
paper experiments.

## Pipeline (legacy)

```
Phase 1 (HPC)              Phase 2 (local)                Phase 3 (HPC)
DDIM inversion of the      SAM mask on reconstruction.png masked LoRA edit
real image
```

Two backbones are implemented through the same scripts:
`scripts/01_invert_real.py` (DDIM inversion of the input image) and
`scripts/03_masked_edit_real.py` (masked LoRA blend at edit time).
The SDXL variant requires SDXL sliders (the SD-1.4 sliders are not
compatible with the SDXL UNet).

## Why it was superseded

The DDIM inversion used here is unconditional: at higher slider scales
the reconstruction drifts and the masked-edit can produce ghosting
outside the mask. The Tight Inversion backend in
`sdxl/tasks/real_editing/` couples each inversion step with a gradient
descent step on the noise prediction and uses IP-Adapter as visual
conditioning, which keeps the edit trajectory aligned with the
inversion trajectory and removes the drift on every image we tested.
That implementation is what the paper refers to in §1.

If you need a quick DDIM-only baseline on SD-1.4, this folder is still
runnable. Otherwise, use `sdxl/tasks/real_editing/`.
