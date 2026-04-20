# SAM in locale (sul Mac)

Phase 2 della pipeline `masked_Lora_FLUX` gira in locale, NON su HPC, perché
serve un display per il click interattivo.

## Setup una tantum

```bash
# Crea un venv per SAM sul Mac (fuori dalla repo)
python3 -m venv ~/venvs/sam-local
source ~/venvs/sam-local/bin/activate

pip install torch torchvision numpy pillow matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git

# Scarica il checkpoint (~2.4 GB, vit_h)
mkdir -p ~/sam_checkpoints
curl -L -o ~/sam_checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Se preferisci un checkpoint più leggero (qualità ok per target grossi come
il cielo): `sam_vit_b_01ec64.pth` (~375 MB), poi `--sam_model_type vit_b`.

## Workflow per un nuovo RUN_ID

Assumiamo `RUN_ID=vangogh_sky_seed42`.

### 1. Scarica `base.png` da HPC

```bash
# Prepara la run_dir locale che mima quella HPC
mkdir -p ~/masked_lora_local/runs/vangogh_sky_seed42

# Scarica sia base.png che metadata.json (utile per ricordarsi il prompt)
scp bocconi:/home/<your-username>/FERT_PROJECT/local-concept-sliders/masked_Lora_FLUX/runs/vangogh_sky_seed42/base.png \
    ~/masked_lora_local/runs/vangogh_sky_seed42/
scp bocconi:/home/<your-username>/FERT_PROJECT/local-concept-sliders/masked_Lora_FLUX/runs/vangogh_sky_seed42/metadata.json \
    ~/masked_lora_local/runs/vangogh_sky_seed42/
```

### 2. Lancia SAM interactive

Clona (o mantieni clonata) la repo anche in locale solo per avere
`02_segment_with_sam.py` a portata. Poi:

```bash
cd ~/path/to/local-concept-sliders       # clone locale della repo
source ~/venvs/sam-local/bin/activate

python masked_Lora_FLUX/02_segment_with_sam.py \
    --run_dir ~/masked_lora_local/runs/vangogh_sky_seed42 \
    --sam_checkpoint ~/sam_checkpoints/sam_vit_h_4b8939.pth \
    --sam_model_type vit_h \
    --mode interactive \
    --output_name mask.png
```

Si apre una finestra matplotlib con `base.png`:
- **Click** sul cielo (o sul target che vuoi maskare) — un punto singolo
- **Chiudi la finestra** — SAM calcola la mask best-score, la salva in
  `~/masked_lora_local/runs/vangogh_sky_seed42/mask.png`

Controlla visivamente `mask.png`. Se non ti convince, rilancia.

### 3. Upload della mask su HPC

```bash
scp ~/masked_lora_local/runs/vangogh_sky_seed42/mask.png \
    bocconi:/home/<your-username>/FERT_PROJECT/local-concept-sliders/masked_Lora_FLUX/runs/vangogh_sky_seed42/
```

### 4. Lancia phase3 su HPC

Assicurati che il `RUN_ID` in `run_phase3.slurm` combaci, poi:

```bash
ssh bocconi
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
sbatch masked_Lora_FLUX/jobs/run_phase3.slurm
```

## Modalità alternative (senza click)

Se preferisci lavorare senza GUI anche in locale (es. scripting batch):

```bash
# Box mode: bounding box [x1 y1 x2 y2]
python masked_Lora_FLUX/02_segment_with_sam.py \
    --run_dir ~/masked_lora_local/runs/vangogh_sky_seed42 \
    --sam_checkpoint ~/sam_checkpoints/sam_vit_h_4b8939.pth \
    --mode box --box 20 10 492 240

# Point mode: single seed pixel
python masked_Lora_FLUX/02_segment_with_sam.py \
    --run_dir ~/masked_lora_local/runs/vangogh_sky_seed42 \
    --sam_checkpoint ~/sam_checkpoints/sam_vit_h_4b8939.pth \
    --mode point --point_x 256 --point_y 80
```

## Iterare

Il punto chiave dello split: se la mask non ti piace, basta rilanciare
`02_segment_with_sam.py` in locale con un click diverso, caricare la nuova
mask via `scp`, rilanciare solo `run_phase3.slurm`. Non si rigenera mai la
base.
