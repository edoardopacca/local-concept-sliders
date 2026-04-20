# flux/tasks/masked_lora

Port di MaskedLoRA (originariamente SDXL) su Flux.1-dev.

## Idea in una riga

Due forward pass del transformer per step di denoising (uno con slider, uno
senza), blend finale nello spazio del rumore usando una mask binaria:

```
eps_blend = mask * eps_slider + (1 - mask) * eps_base
latents   = scheduler.step(eps_blend, t, latents)
```

## Perche' su Flux

Su Flux abbiamo documentato empiricamente che **LoRAShop fallisce sui style
LoRA** (van Gogh propaga su tutta l'immagine anche con mask localizzata
sulle montagne). La causa e' la self-attention che, dopo l'applicazione
localizzata del LoRA, propaga le feature stilizzate verso i token fuori
mask nei blocchi successivi.

MaskedLoRA elimina fisicamente questa leak: i due forward pass sono
indipendenti, non c'e' mai una self-attention che mescola token "styled"
e "clean" insieme. Il blend avviene FUORI dal modello, sullo spazio
epsilon del noise prediction.

## Costo

~2x rispetto a una normale generazione Flux: due forward del transformer
per step. Su A100 512x512, ~60s/immagine invece di 30s.

## Differenza chiave vs SDXL originale

Flux usa **packed latents**: (B, N_tokens, C_packed) invece di (B, C, H, W).
La mask va convertita in mask-per-token con downscale a H/16 * W/16 e
flatten alla sequenza. Il resto della logica e' identico all'SDXL.

## Struttura

```
flux/tasks/masked_lora/
  flux_masked_pipeline.py  # wrapper con logica dual-path
  01_generate_base.py      # genera base image + init_latents + metadata
  02_get_mask.py           # produce mask (SAM o alternative)
  03_masked_edit.py        # core: dual-path blend
  jobs/                    # slurm scripts HPC
  logs/                    # slurm output
  outputs/                 # generated images
```

## Relazione con shop_concept/

Riusa:
- `shop_concept/convert_slider_to_peft.py` per caricare gli slider .pt Flux
- lo stesso cache `_peft_cache/` per i .safetensors convertiti

Non riusa (volutamente):
- l'attention hooking di `shop_concept/flux_blocks.py` (LoRAShop-style):
  qui vogliamo una strada completamente diversa per confrontare
