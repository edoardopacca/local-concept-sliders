# External Sources

This file documents the provenance of all external code used in
the `real_editing/` pipeline.

| Component | Status | Source | URL | Commit/Tag | License | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| tight_inversion | implemented | HF Space tight-inversion/tight-inversion | https://huggingface.co/spaces/tight-inversion/tight-inversion | b9919c2 | Public HF Space (no explicit license file) | arXiv:2502.20376 |
| ddim | implemented | inv_step math from HF Space tight-inversion/tight-inversion + diffusers DDIMScheduler | https://huggingface.co/spaces/tight-inversion/tight-inversion | b9919c2 | Public HF Space (no explicit license) | -- |
| null_text | implemented (SD1.4), experimental (SDXL) | This repo: exp_editing/edit_with_sliders.py NullInversion class | local | -- | -- | arXiv:2211.09794 |
| edict | stub | salesforce/EDICT | https://github.com/salesforce/EDICT | -- | BSD-3-Clause | arXiv:2211.12446 |

## Tight Inversion — Files Adapted

The following files from the HuggingFace Space `tight-inversion/tight-inversion`
(commit `b9919c2`) were adapted for use in `real_editing/inversion/`:

| Original file | Adapted into | What was extracted |
| --- | --- | --- |
| `src/exact_inversion.py` | `tight_inversion.py` | `inversion_step()` GD optimization logic, `unet_pass()` helper |
| `src/schedulers/ddim_scheduler.py` | `ddim.py` | `MyDDIMScheduler.inv_step()` reverse DDIM step math |
| `src/pipes/sdxl_inversion_pipeline.py` | `tight_inversion.py` | SDXL inversion loop structure with `added_cond_kwargs` |
| `app.py` | `tight_inversion.py` | IP-Adapter integration pattern |

## Runtime Dependencies

| Dependency | Source | License | Notes |
| --- | --- | --- | --- |
| IP-Adapter weights | h94/IP-Adapter (HuggingFace) | Apache-2.0 | Downloaded on first use via `pipe.load_ip_adapter()` |
| CLIP Vision encoder | h94/IP-Adapter models/image_encoder | Apache-2.0 | Used by Tight Inversion for image conditioning |
| LoRANetwork | trainscripts/textsliders/lora.py (this repo) | -- | Architecture-agnostic LoRA implementation |
