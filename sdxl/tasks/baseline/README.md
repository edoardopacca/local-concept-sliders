# sdxl/tasks/baseline

**Generazione + applicazione base** di un concept slider SDXL.

Genera un set di immagini con SDXL e applica uno slider trainato a varie `scale` per visualizzare l'effetto. Niente segmentation, niente masking, niente real image editing — solo "prompt + slider + sweep di scale".

## Cosa fa

Sotto il cofano riusa `sdxl/trained_sliders/training/scripts/generate_with_sliders.py`:
- Carica SDXL base
- Carica lo slider LoRA da `--slider`
- Genera N immagini per ogni `scale` in `--scales`
- Salva in `--save_path` una cartella con `scale_X.png` per ogni scala + un grid di confronto

## Quando usarlo

- Verificare visivamente l'effetto di uno slider appena trainato
- Generare baseline (`scale=0`) vs intervento (`scale=2`) per confronti
- Sweep di scale per vedere la dose-response

## Lancio

```bash
sbatch sdxl/tasks/baseline/jobs/run_baseline.slurm
```

Customizza dentro lo SLURM:
- `SLIDER` — path allo slider (`sdxl/trained_sliders/sliders/<nome>.pt` o `.safetensors`)
- `PROMPT` — testo di generazione
- `SCALES` — lista di scale da provare
- `SAVE_PATH` — output dir (default sotto `sdxl/tasks/baseline/outputs/`)

## Output

`sdxl/tasks/baseline/outputs/<run_name>/` contiene:
- `scale_0.png`, `scale_1.png`, ... — un PNG per ogni scale
- `grid.png` — confronto affiancato di tutti gli scale
