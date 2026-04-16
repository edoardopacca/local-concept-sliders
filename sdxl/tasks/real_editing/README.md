# real_editing

Pipeline per **real image editing** con concept sliders su SDXL.  
Obiettivo: invertire una fotografia reale nello spazio latente del modello, poi applicare un LoRA slider in modo mascherato su una regione specifica — senza toccare il resto dell'immagine.

---

## Pipeline completa

```
Foto originale
      │
      ▼
[Stage 1] Tight Inversion  ──────────────────────────────────────────────
      │   scripts/invert_real_image.py
      │   Job HPC: real_editing/jobs/run_tight_inversion.slurm
      │
      │   Output: runs/<RUN_ID>/
      │     ├── original.png        ← foto a piena risoluzione (EXIF corretta)
      │     ├── reconstruction.png  ← crop 1024×1024 che SDXL vede
      │     ├── x_t.pt              ← latente invertito
      │     ├── text_condition/     ← embeddings testuali salvati
      │     ├── metadata.json
      │     └── config.json
      │
      ▼
[Stage 2] Segmentazione SAM  ────────────────────────────────────────────
      │   scripts/segment_with_sam.py
      │   (gira in locale sul Mac, NON sull'HPC)
      │
      │   ⚠️  Segmenta SEMPRE reconstruction.png, non original.png.
      │       SDXL lavora nel crop 1024×1024. Se segmenti original.png
      │       le coordinate della maschera non si allineano ai latenti.
      │
      │   Output: runs/<RUN_ID>/mask_target.png
      │
      ▼
[Stage 3] Masked Edit  ──────────────────────────────────────────────────
      │   scripts/edit_real_image_masked.py
      │   Job HPC: real_editing/jobs/run_tight_edit.slurm
      │
      │   Output: runs/<RUN_ID>/
      │     ├── edited_<nome>.png            ← crop 1024×1024 editato (raw)
      │     └── composite_edited_<nome>.png  ← OUTPUT FINALE: edit incollato
      │                                         nella foto originale ad alta res
      ▼
Immagine finale
```

---

## Struttura del codice

```
real_editing/
├── models/
│   ├── base.py        ← interfaccia astratta ModelContext
│   ├── sdxl.py        ← implementazione SDXL (VAE fisso in float32, IP-Adapter)
│   └── loader.py      ← factory: load_model_context(...)
│
├── inversion/
│   ├── base.py              ← ABC InversionBackend + dataclass InversionResult
│   ├── tight_inversion.py   ← backend principale: DDIM + GD + IP-Adapter
│   ├── ddim.py              ← funzioni helper usate da tight_inversion
│   └── registry.py          ← get_backend("tight_inversion")
│
├── editing/
│   ├── masked_editor.py  ← MaskedLoRAEditor: applica slider solo dentro la maschera
│   ├── blending.py       ← noise_blend, feather_mask, pixel_composite, load_mask
│   └── slider_loader.py  ← carica checkpoint LoRA sull'UNet
│
├── io/
│   ├── artifacts.py  ← save/load inversion artifacts + edit artifacts
│   └── metrics.py    ← LPIPS, SSIM, PSNR
│
├── jobs/
│   ├── run_tight_inversion.slurm  ← job SLURM Stage 1
│   └── run_tight_edit.slurm       ← job SLURM Stage 3
│
├── runs/              ← output degli esperimenti (creata a runtime)
├── archive/           ← file non più usati (sd1x.py, null_text.py, provenance.py)
└── README.md
```

---

## Stage 1 — Inversione (HPC)

```bash
RUN_ID=my_run_001 \
IMAGE=real_editing/input/foto.jpg \
PROMPT="una persona in piedi" \
sbatch real_editing/jobs/run_tight_inversion.slurm
```

Parametri configurabili via env var:

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `RUN_ID` | `try_tight_001` | Nome cartella output sotto `runs/` |
| `IMAGE` | *(vedi slurm)* | Path immagine di input |
| `PROMPT` | `two people standing together` | Descrizione testuale dell'immagine |
| `NUM_GD_STEPS` | `3` | Passi di gradient descent per step di inversione |
| `IPA_SCALE` | `0.4` | Forza del conditioning IP-Adapter |

---

## Stage 2 — Maschera SAM (Mac locale)

```bash
# Modalità interattiva: clicca sul soggetto nella finestra
python3 scripts/segment_with_sam.py \
    --run_dir real_editing/runs/<RUN_ID> \
    --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
    --image_name reconstruction.png \
    --mode interactive

# Modalità punto: coordinate x,y dentro la regione target
python3 scripts/segment_with_sam.py \
    --run_dir real_editing/runs/<RUN_ID> \
    --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
    --image_name reconstruction.png \
    --mode point --point_x <x> --point_y <y>

# Modalità box: bounding box attorno alla regione target
python3 scripts/segment_with_sam.py \
    --run_dir real_editing/runs/<RUN_ID> \
    --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
    --image_name reconstruction.png \
    --mode box --box <x1> <y1> <x2> <y2>
```

Salva `mask_target.png` direttamente in `real_editing/runs/<RUN_ID>/`.

---

## Stage 3 — Edit mascherato (HPC)

```bash
RUN_ID=my_run_001 \
MASK_FROM_RUN_ID=my_run_001 \
SLIDER_PATH=sdxl/trained_sliders/sliders/smiling.pt \
SLIDER_SCALE=2.0 \
START_NOISE=800 \
OUTPUT_NAME=edited_smile_v1.png \
sbatch real_editing/jobs/run_tight_edit.slurm
```

Parametri configurabili via env var:

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `RUN_ID` | `try_tight_001` | Run da editare (deve avere `x_t.pt`) |
| `MASK_FROM_RUN_ID` | `try_001` | Run che contiene `mask_target.png` |
| `SLIDER_PATH` | `sdxl/trained_sliders/sliders/smiling.pt` | Pesi LoRA slider |
| `SLIDER_SCALE` | `0.4` | Forza dell'edit (valori tipici: 1.0–3.0) |
| `START_NOISE` | `350` | Applica lo slider solo ai timestep ≤ questo valore |
| `GUIDANCE_SCALE` | `2.0` | CFG scale (non alzare oltre ~3.0) |
| `FEATHER_RADIUS` | `16` | Blur gaussiano sui bordi della maschera |
| `OUTPUT_NAME` | `edited_tight_001_v2.png` | Nome file output |

**Output:**
- `edited_<nome>.png` — crop 1024×1024 denoised direttamente dalla VAE
- `composite_edited_<nome>.png` — **output finale**: la regione editata incollata nella foto originale ad alta risoluzione

---

## Note tecniche importanti

**VAE in float32** — Il VAE di SDXL è instabile in float16 e produce output neri/NaN. Nel codice è tenuto permanentemente in float32 anche quando il resto del modello gira in float16.

**IP-Adapter obbligatorio nell'editing** — L'inversione avviene CON IP-Adapter come conditioning visivo. L'editing deve usare lo stesso IP-Adapter con gli stessi parametri, altrimenti la traiettoria di denoising diverge da quella dell'inversione → immagine nera.

**`start_noise`** — Valori bassi (es. 350) = edit sottile solo nei dettagli. Valori alti (es. 800) = edit più strutturale. Alzare troppo rischia artefatti.

**`slider_scale`** — Sopra ~3.0 si vedono artefatti fuori dalla maschera anche con feathering. Per risultati naturali stare tra 1.0 e 2.5.

**Memoria GPU** — SDXL UNet (float16) + VAE (float32) + IP-Adapter + ViT-H encoder richiedono ~14–16 GB. I job SLURM richiedono 24 GB.

---

## Setup HPC (una-tantum)

```bash
export HF_HOME=/home/<your-username>/FERT_PROJECT/Caches_and_venvs/hf_cache
export HF_HUB_CACHE=/home/<your-username>/FERT_PROJECT/Caches_and_venvs/hf_cache/hub
```

Modelli necessari in cache:
- `models--stabilityai--stable-diffusion-xl-base-1.0`
- `models--h94--IP-Adapter` (con `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` e `models/image_encoder/`)

---

## Riferimento paper

Tight Inversion è adattato da:

> Kadosh et al., *"Tight Inversion: Image-Conditioned Inversion for Real Image Editing"*,  
> arXiv:2502.20376, ICCV 2025 Workshop.
