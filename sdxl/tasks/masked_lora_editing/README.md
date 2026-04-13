# sdxl/tasks/masked_lora_editing (Real Image Version)

Pipeline parallela a `sdxl/tasks/masked_lora`, ma per editing di immagini reali.

## Obiettivo

Applicare slider LoRA solo in una regione spaziale (maschera SAM) su un'immagine reale:

`eps_blend = M * eps_lora + (1 - M) * eps_base`

## Fasi (pipeline SD1.4 originale)

1. **Phase 1 (HPC): inversione immagine reale**
   - Script: `01_invert_real.py`
   - Output: `original.png`, `reconstruction.png`, `x_t.pt`, `uncond_embeddings.pt`, `metadata.json`
2. **Phase 2 (Locale o HPC): segmentazione SAM**
   - Script: `02_segment_with_sam.py`
   - Output: `mask_target.png`, `mask_meta.json`
3. **Phase 3 (HPC): masked edit LoRA**
   - Script: `03_masked_edit_real.py`
   - Output: `edited_target_only.png`, `edit_meta.json`, `metrics.json`

## Job Slurm pronti (SD1.4)

- `jobs/run_phase1.slurm`
- `jobs/run_phase3.slurm`

Entrambi sono configurati con le stesse convenzioni cluster di `sdxl/tasks/masked_lora`:
`account=3226571`, `partition=stud`, `qos=stud`.

## Esempio rapido

### Phase 1

```bash
sbatch sdxl/tasks/masked_lora_editing/jobs/old_slurm/run_phase1.slurm
```

### Phase 2 (interattiva locale)

```bash
python mask_SAM/segment_with_sam.py \
  --run_dir sdxl/tasks/masked_lora_editing/outputs/real_edit_001 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --mode interactive \
  --output_name mask_target.png
```

### Phase 3

```bash
sbatch sdxl/tasks/masked_lora_editing/jobs/old_slurm/run_phase3.slurm
```

## Note

- Questa pipeline usa SD1.4 come `sdxl/tasks/real_editing` (esiste invece il vecchio `exp_editing` rimosso).
- In modalità HPC offline, lasciare `--skip_metrics` in phase 3.
- Per metriche complete, servono pesi LPIPS/CLIP in cache accessibile dal compute node.

## Variante SDXL (più potente)

Per testare un backbone più forte, è disponibile una pipeline parallela SDXL:

1. **Phase 1 SDXL (HPC): preparazione latenti da immagine reale**
   - Script: `01_prepare_real_sdxl.py`
   - Job: `jobs/run_phase1_sdxl.slurm`
   - Output: `original.png`, `source_latents.pt`, `metadata.json`
2. **Phase 2: segmentazione SAM**
   - Script: `02_segment_with_sam.py` (stesso script della pipeline SD1.4)
3. **Phase 3 SDXL (HPC): masked edit LoRA**
   - Script: `03_masked_edit_real_sdxl.py`
   - Job: `jobs/run_phase3_sdxl.slurm`
   - Output: `edited_target_only_sdxl.png`, `edit_meta.json`, `metrics.json`

### Nota importante su slider compatibili

- La variante SDXL richiede slider SDXL (es. `sdxl/trained_sliders/sliders/*.pt`).
- Gli slider SD1.4 non sono compatibili con U-Net SDXL.
