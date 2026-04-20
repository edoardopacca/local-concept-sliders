# masked_Lora_FLUX/jobs

Slurm entrypoints per la pipeline MaskedLoRA su Flux.1-dev.

La pipeline è **split HPC + locale**:

```
  HPC          local Mac         HPC
 phase1   →   phase2 (SAM)   →  phase3
(Flux)      (interactive)     (dual-path)
```

Su HPC non c'è display, quindi SAM interactive non gira. La base la
generiamo su GPU su Bocconi, la mask la disegniamo a click sul Mac (più
veloce e più preciso di `--mode box` a coordinate cieche), poi il risultato
finale lo rimandiamo su HPC per il dual-path.

## Slurm disponibili (HPC)

- `run_phase1.slurm` — genera `base.png` + `metadata.json` in
  `masked_Lora_FLUX/runs/<RUN_ID>/`.
- `run_phase3.slurm` — dual-path velocity-blend con LoRA van Gogh.
  Output: `edited.png` + `edit_meta.json`.

`RUN_ID` deve combaciare fra i due.

## Workflow completo

```bash
# --- Step 1 (HPC) ---
sbatch masked_Lora_FLUX/jobs/run_phase1.slurm
# aspetta, poi:
scp bocconi:/home/<your-username>/FERT_PROJECT/local-concept-sliders/masked_Lora_FLUX/runs/<RUN_ID>/base.png ./

# --- Step 2 (locale, sul Mac) ---
# Usa lo script 02_segment_with_sam.py localmente, vedi
# masked_Lora_FLUX/scripts/local_sam.md per i dettagli.
# Output: mask.png

# --- Step 3 (upload della mask su HPC) ---
scp mask.png bocconi:/home/<your-username>/FERT_PROJECT/local-concept-sliders/masked_Lora_FLUX/runs/<RUN_ID>/

# --- Step 4 (HPC) ---
sbatch masked_Lora_FLUX/jobs/run_phase3.slurm
scp bocconi:/home/<your-username>/FERT_PROJECT/local-concept-sliders/masked_Lora_FLUX/runs/<RUN_ID>/edited.png ./
```

## Iterare sulla sola mask

Il bello dello split è che phase1 lo fai una volta sola. Se la mask non ti
piace, riavvii SAM sul Mac, ricarichi `mask.png`, rilanci solo `run_phase3`.
Niente rigenerazione di Flux.

## Confronto diretto LoRAShop vs MaskedLoRA

Per avere A/B pulito:

1. `run_phase1.slurm` + mask sul Mac + `run_phase3.slurm` con
   `vangogh_flux_v1`, seed 42, `--slider_scale 2.0`.
2. `shop_concept/jobs/sweep_vangogh_sky.slurm` con stesse seed e prompt,
   `--target_prompt "sky"`.
3. Confronta `masked_Lora_FLUX/runs/<RUN_ID>/edited.png` vs
   `shop_concept/outputs/sweep_vangogh_sky/seed42_scale*.png`.

Lettura della differenza:

- Se MaskedLoRA mantiene il cielo stilizzato e le montagne photorealistic
  mentre LoRAShop stilizza anche le montagne → thesis finding confermato
  (blend esterno elimina il leak via self-attention).
- Se MaskedLoRA lascia un bordo visibile fra cielo e montagne → mask
  soft/feathering. Si modifica `03_masked_edit.py::pack_mask_for_flux`
  sostituendo la soglia `> 0.5` con `mask_soft = mask_down` diretto.

## Naming degli output

- `runs/<RUN_ID>/base.png` — Flux base (phase 1, HPC)
- `runs/<RUN_ID>/mask.png` — mask SAM (phase 2, locale)
- `runs/<RUN_ID>/mask_meta.json` — mode/prompt SAM usato (locale)
- `runs/<RUN_ID>/edited.png` — risultato dual-path (phase 3, HPC)
- `runs/<RUN_ID>/edit_meta.json` — slider, scale, edit_start_step
