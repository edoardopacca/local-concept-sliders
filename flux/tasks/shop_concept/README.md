# shop_concept

Applica Concept Sliders a Flux.1-dev con il metodo di composizione di
**LoRAShop**: mask estratte dall'attenzione del block 19 nei primi 5
timestep, poi blending per-token nel denoising. Piu' slider
contemporaneamente, ciascuno applicato alla regione corrispondente al
proprio `target_prompt`.

Self-contained: tutto quello che serve e' in questa cartella. Nessun
import da `LoRAShop-main/` o altri pacchetti del repo.

Per la lista dettagliata delle modifiche rispetto a `LoRAShop-main/`
vedi `CHANGES.md`.

## Layout

```
shop_concept/
  __init__.py
  utils.py                    # helper (copia identica di LoRAShop)
  flux_blocks.py              # TransformerBlock + SingleTransformerBlock patched
  flux_real_pipeline.py       # RealGenerationPipeline patched
  convert_slider_to_peft.py   # converter .pt (LoRANetwork) -> .safetensors (PEFT)
  generate.py                 # entrypoint CLI
  jobs/
    generate_shop_concept.slurm
  CHANGES.md
  README.md
```

## Workflow tipico

1. **Addestra uno o piu' Concept Sliders** con
   `training_local_concept_sliders/FLUX_train/scripts/train_flux_slider.py`.
   L'output e' una cartella con `slider_0.pt`.

2. **(Opzionale) Pre-converti il .pt a .safetensors PEFT** — utile se
   riutilizzi lo stesso slider spesso:

   ```bash
   python -m shop_concept.convert_slider_to_peft \
       --input  training_local_concept_sliders/.../slider_0.pt \
       --output shop_concept/_peft_cache/vangogh_v1.safetensors
   ```

   Se salti questo step, `generate.py` converte on-the-fly e cacha
   dentro `shop_concept/_peft_cache/`.

3. **Lancia la generazione** via slurm:

   ```bash
   sbatch shop_concept/jobs/generate_shop_concept.slurm
   ```

   Oppure da shell (ambiente gia' attivo):

   ```bash
   python -m shop_concept.generate \
       --slider_paths training_local_concept_sliders/.../vangogh_v1/slider_0.pt \
                      training_local_concept_sliders/.../age_man_v1/slider_0.pt \
       --target_prompt "landscape" "man" \
       --lora_scales 1.0 1.0 \
       --prompt "a man standing in front of a landscape" \
       --output_path shop_concept/outputs/demo.png \
       --height 512 --width 512 \
       --num_inference_steps 30 \
       --edit_start_step 8 \
       --seed 42
   ```

## Vincolo sui prompt

Ogni `--target_prompt` **deve essere sottostringa letterale** di
`--prompt`. LoRAShop estrae gli indici token con
`prompt.index(target_prompt)`; se non matcha, error hard. Es:
  * prompt: `"a woman in a red dress and a man smiling"`
  * target: `"woman in a red dress"` OK
  * target: `"the woman"` KO (non sottostringa)

## Semantica di `--lora_scales`

E' la scale continua del Concept Slider, identica al `--lora_scales`
del generatore standalone in `FLUX_train/scripts/generate_flux_slider.py`:

  * `0.0` = slider off (nessun effetto)
  * `1.0` = slider a piena forza (forza di training)
  * `> 1.0` = extrapolation (piu' forte del training)
  * `< 0.0` = direction reversed (slider all'inverso)

Dopo la conversione con `fold_alpha=True` (default di
`convert_slider_to_peft.py`), la scale 1.0 corrisponde esattamente
a `multiplier * (alpha/rank)` del training. Non ci sono fattori
nascosti di 16x o 1/16.

## Parametri rilevanti

  * `--edit_start_step` (default 8): il blending mask-guidato inizia
    solo da questo timestep in avanti. Nei primi step si fa solo
    estrazione della mask e denoising base. Alzare per effetto piu'
    soft, abbassare per dominanza maggiore degli slider.
  * `--num_inference_steps` (default 30): passi totali di denoising.
    I primi 5 sono fissi per mask extraction.
  * `--height/--width`: 512x512 e' un buon compromesso GPU/tempo per
    sanity check; 1024x1024 e' la risoluzione nativa di Flux.1-dev.

## Fallback single-slider

Anche passando un solo slider + un solo target_prompt, il pipeline
funziona: applica lo slider solo nell'area mascherata e fuori resta
il Flux base. Utile per isolare l'effetto a una regione precisa senza
impattare il resto della scena.

## Dipendenze

Stesso venv del training (`flux-sliders`). Nessun pacchetto extra
richiesto:
  * torch, diffusers (testato con 0.31.0), transformers
  * peft (>= 0.10.0)
  * safetensors
  * opencv-python (per morphological reconstruction delle mask)
  * scipy, matplotlib (usati in utility di visualizzazione dell'ori-
    ginale; preservati per drop-in compat).
