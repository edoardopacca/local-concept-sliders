# Guida: Eval Masked LoRA su Flux

Questa guida spiega come eseguire il set di valutazione **masked LoRA** (6 concept × 20 immagini) usando i tuoi slider **Flux.1-dev**.

Il workflow è diviso in 3 fasi:
- **Fase 1** — su HPC: genera le 120 immagini base
- **Fase 2** — in locale (delegata): crea le maschere SAM interattive
- **Fase 3** — su HPC: applica i masked LoRA e calcola le metriche

---

## Prerequisiti

### 1. Setup venv Flux su HPC (se non già fatto)

```bash
# Dalla tua home su HPC
python3 -m venv ~/venvs/flux-env
source ~/venvs/flux-env/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors peft
pip install lpips  # per le metriche
```

Il tuo `tools/set_slurms.sh` deve definire la funzione `activate_flux_env` che attiva questo venv.

### 2. Slider Flux

Prima di lanciare la **fase 3**, devi editare i 6 file `eval_*_phase3.slurm` e impostare la variabile `SLIDER_PATH` con il path corretto del tuo slider. I file sono in:

```
flux/tasks/masked_lora/jobs/new_slurm/
```

Ogni file ha un commento `⚠️  IMPOSTA IL PATH DEL TUO SLIDER QUI` con una riga da modificare, ad esempio:

```bash
# Prima (placeholder):
SLIDER_PATH="flux/trained_sliders/sliders/<AGE_SLIDER_DIR>/<AGE_SLIDER_FILE>_last.pt"

# Dopo (esempio reale):
SLIDER_PATH="flux/trained_sliders/sliders/age_person_flux_v1_rank16/age_person_flux_v1_rank16_last.pt"
```

Devi farlo per tutti e 6 i concept: `age_person`, `curlyhair`, `smile_person`, `furlength`, `daynight`, `painterly`.

### 3. CyberDuck

Userai **CyberDuck** per scaricare le immagini dall'HPC e caricare le maschere. 
---

## FASE 1 — Genera le immagini base (HPC)

Lancia i 6 job in parallelo. Ogni job genera 20 immagini base (~2.5h ciascuno):

```bash
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_age_person_phase1.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_curlyhair_phase1.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_smile_person_phase1.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_furlength_phase1.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_daynight_phase1.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_painterly_phase1.slurm
```

Output in: `flux/tasks/masked_lora/runs/eval_{concept}_{01..20}/base.png`

Monitora con `squeue -u $USER`. Quando tutti sono completati, passa alla fase 2.

---

## FASE 2 — Crea le maschere SAM (in locale, delegata)

### Cosa fare tu (prima di passare il lavoro)

Con CyberDuck, scarica **tutte** le cartelle `eval_*` da:
```
flux/tasks/masked_lora/runs/
```
sul tuo Mac, mantenendo la struttura di directory. Avrai 120 cartelle, ciascuna con `base.png`.

> **Nota:** le immagini base non vanno su git (sono file binari grandi). Si trasferiscono solo con CyberDuck.

Poi segui le istruzioni qui sotto per delegare la fase 2 a un'altra persona, oppure eseguila tu stesso.

---

### Istruzioni da mandare a chi fa le maschere

---

**Oggetto: maschere SAM per l'eval masked LoRA**

Ciao! Ti mando le istruzioni per creare le maschere per le nostre immagini.

#### 1. Setup (ma Rebe dovrebbe già aver tutto perchè lo aveva fatto per sdxl)

```bash
# Crea un venv Python per SAM
python3 -m venv ~/venvs/sam-local
source ~/venvs/sam-local/bin/activate

pip install torch torchvision numpy pillow matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git

# Scarica il checkpoint SAM (~2.4 GB)
mkdir -p ~/sam_checkpoints
curl -L -o ~/sam_checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Poi decomprimi la cartella che ti mando (`base_images_for_sam.zip`) e mettila dentro alla cartella `mask_SAM/` della repo, accanto a `checkpoints/` e `segment_with_sam.py`.

#### 2. Segmentazione

Lancia questo script dal terminale. Per ogni immagine si apre una finestra: clicca sul soggetto giusto e chiudi. La maschera viene salvata automaticamente.

```bash
cd mask_SAM

for img in base_images_for_sam/*.png; do
    run_id=$(basename "$img" .png)
    mkdir -p "masks_output/$run_id"
    cp "$img" "masks_output/$run_id/base.png"
    python segment_with_sam.py \
        --run_dir       "masks_output/$run_id" \
        --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --mode          interactive \
        --image_name    base.png \
        --output_name   mask_target.png
done
```

**Cosa cliccare per ogni concept** (il nome del concept è nel nome del file immagine):

| Concept | Cosa cliccare |
|---------|---------------|
| `eval_age_person_*` | viso di una qualsiasi **donna** nell'immagine |
| `eval_smile_person_*` | viso di una qualsiasi **donna** nell'immagine |
| `eval_curlyhair_*` | **capelli** di uno degli uomini (clicca proprio sui capelli, non sul viso) |
| `eval_furlength_*` | **corpo** di un animale (uno solo se ce ne sono più) |
| `eval_daynight_*` | regione del **cielo** |
| `eval_painterly_*` | regione del **cielo** (tutte le immagini hanno cielo visibile) |

Se il click non ti convince (maschera brutta), premi Ctrl+C e rilancia solo quella immagine con un click diverso.

#### 3. Output

Quando hai finito, comprimi e mandami la cartella di output:

```bash
zip -r masks_output.zip masks_output/
```

---

### Dopo aver ricevuto le maschere

1. Decomprimi `masks_output.zip` **nella root della repo** (stessa cartella dove c'è `mask_SAM/`, `flux/`, ecc.). Dopo la decompressione devi avere:
   ```
   local-concept-sliders/
     masks_output/
       eval_age_person_01/
         mask_target.png
       eval_age_person_02/
         mask_target.png
       ...
   ```

2. Lancia lo script che copia automaticamente le maschere nelle run dir corrette:
   ```bash
   # Flux (default)
   python mask_SAM/place_masks.py

   # Verifica prima senza copiare (consigliato):
   python mask_SAM/place_masks.py --dry_run
   ```
   Lo script legge `masks_output/` e copia ogni `mask_target.png` in `flux/tasks/masked_lora/runs/{run_id}/mask_target.png`. Stampa `[OK]` per ogni copia riuscita e `[SKIP]` se una run dir non esiste ancora.

3. Con CyberDuck, carica di nuovo la cartella runs sovrascrivendola.
---

## FASE 3 — Applica i masked LoRA (HPC)

**Prima** di lanciare i job, assicurati di aver impostato `SLIDER_PATH` in ciascun file (vedi Prerequisiti § 2).

Inoltre IMPORTANTE: al momento i job fanno :
Ogni job (~6h) genera 3 immagini editate per run dir:
```
edited_{concept}_s1.5.png
edited_{concept}_s2.0.png
edited_{concept}_s3.0.png
```
(eccetto `age_person` e `smile_person` che usano scale `0.5 1.0 1.5`)

però è importante che MODIFICHI questi 1.5, 1.0, ecc a seconda dei tuoi slider quanto pesanti sono. Ad esempio il mio notte non fa niente finchè non metto slider 3, mentre il mio sorriso già all'1 fa tantissimo e il 3 era troppo. 
modifica quindi adattandolo ai tuoi slider.

```bash
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_age_person_phase3.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_curlyhair_phase3.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_smile_person_phase3.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_furlength_phase3.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_daynight_phase3.slurm
sbatch flux/tasks/masked_lora/jobs/new_slurm/eval_painterly_phase3.slurm
```


---

## EVAL — Calcola le metriche (HPC)

Lancia il job di valutazione. 

```bash
# Tutti i concept in un solo job
sbatch flux/tasks/masked_lora/jobs/new_slurm/run_eval_metrics.slurm

# Oppure un solo concept (utile per test):
sbatch --export=CONCEPT=smile_person \
  flux/tasks/masked_lora/jobs/new_slurm/run_eval_metrics.slurm
```

Output in: `metrics/results/{concept}/eval_results.csv` e `eval_aggregate.json`

Scarica i risultati con CyberDuck da:
```
metrics/results/
```

---

## Note tecniche

### Differenze Flux vs SDXL

| | SDXL | Flux |
|---|---|---|
| Modello | `stabilityai/stable-diffusion-xl-base-1.0` | `black-forest-labs/FLUX.1-dev` |
| Steps | 50 | 30 |
| Guidance scale | 7.5 | 3.5 |
| Risoluzione | 1024×1024 | 1024×1024 |
| dtype | float16 | bfloat16 (hardcoded) |
| RAM GPU consigliata | 24 GB | 48 GB |
| Negative prompt | sì | no |
| LoRA backend | custom LoRANetwork | PEFT adapter |
| Latent space | (B, C, H/8, W/8) | packed tokens (H/16)×(W/16) |

### Struttura della run dir

```
flux/tasks/masked_lora/runs/eval_{concept}_{NN}/
  base.png                   # fase 1
  metadata.json              # fase 1 (seed, prompt, steps...)
  mask_target.png            # fase 2 (SAM)
  edited_{concept}_s1.5.png  # fase 3
  edited_{concept}_s2.0.png  # fase 3
  edited_{concept}_s3.0.png  # fase 3
  eval_metrics_s1.json       # eval
  eval_metrics_s2.json       # eval
  eval_metrics_s3.json       # eval
```

### Metriche calcolate

- **lpips_inside / lpips_outside** — LPIPS grezzo nella regione mascherata vs. fuori (raw)
- **lpips_inside_norm / lpips_outside_norm** — LPIPS normalizzato per l'area della maschera (comparabile tra concetti con maschere di dimensioni diverse)
- **lpips_localization** — `lpips_inside_norm / lpips_outside_norm` (>1 = modifica concentrata dentro la maschera)
- **clip_localization** — ΔCLIPᵢₙ / (ΔCLIPᵢₙ + |ΔCLIPₒᵤₜ|) → 1 = localizzazione perfetta
