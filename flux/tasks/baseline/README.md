# flux/tasks/baseline

**Generazione + applicazione base** di un concept slider Flux.1-dev.

Genera un set di immagini con Flux e applica uno slider trainato a varie `lora_scales` per visualizzare l'effetto. Niente segmentation, niente masking, niente real image editing — solo "prompt + slider + sweep di scale".

## Cosa fa

Sotto il cofano riusa `flux/trained_sliders/training/scripts/generate_flux_slider.py`:
- Carica Flux.1-dev base
- Carica uno o più LoRA da `--lora_dirs`
- Genera N immagini per ogni combinazione (prompt × seed × scale)
- Salva in `--save_dir`

## Quando usarlo

- Verificare visivamente l'effetto di uno slider Flux appena trainato
- Sweep di scale per vedere la dose-response
- Confronto multi-prompt

## Lancio

```bash
sbatch flux/tasks/baseline/jobs/run_baseline.slurm
```

Customizza dentro lo SLURM:
- `LORA_DIR` — path alla cartella dello slider trainato (`flux/trained_sliders/sliders/<nome>/`)
- `PROMPT` — testo di generazione
- `SCALES` — lista di scale
- `SEEDS` — seed
- `SAVE_DIR` — output dir (default sotto `flux/tasks/baseline/outputs/`)

## Output

`flux/tasks/baseline/outputs/<run_name>/` contiene un PNG per ogni combinazione (scale, seed, prompt).
