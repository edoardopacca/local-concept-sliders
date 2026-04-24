# local-concept-sliders

Repository per **training**, **generazione** e **editing** con concept sliders LoRA su **SDXL** e **Flux.1-dev**, eseguito su HPC via SLURM.

## Struttura

```
local-concept-sliders/
├── sdxl/
│   ├── core/                                # libreria condivisa SDXL
│   │   └── lora.py, train_util.py, prompt_util.py, ...
│   ├── trained_sliders/
│   │   ├── training/                        # framework di training SDXL
│   │   │   ├── scripts/  configs/  prompts/{old,new,test}_prompt/
│   │   │   ├── jobs/{old,new,test}_slurm/   logs/
│   │   └── sliders/                         # .pt/.safetensors (gitignored)
│   └── tasks/
│       ├── baseline/                        # generazione + sweep slider
│       │   ├── scripts/  jobs/{old,new,test}_slurm/  logs/  outputs/
│       ├── masked_lora/                     # generazione + segmentazione + masked edit
│       ├── masked_lora_editing/             # masked editing su immagini reali
│       └── real_editing/                    # tight inversion + masked LoRA
│           ├── scripts/  jobs/{old,new,test}_slurm/  logs/  outputs/
│           └── lib/{models,inversion,editing,io,archive}/
│
├── flux/                                    # stessa struttura di sdxl/
│   ├── core/
│   ├── trained_sliders/{training/, sliders/}
│   └── tasks/
│       ├── baseline/                        # generazione Flux + slider sweep
│       ├── masked_lora/                     # ex masked_Lora_FLUX
│       └── shop_concept/                    # multi-LoRA mask-aware
│           ├── scripts/  jobs/  outputs/
│           └── lib/{flux_blocks,flux_real_pipeline,utils}.py
│
├── mask_SAM/                                # SAM checkpoint + script (cross-arch)
│   ├── checkpoints/sam_vit_h_4b8939.pth     # gitignored, 2.4 GB
│   └── segment_with_sam.py                  # CLI usato dal Mac in modalità interattiva
│
├── metrics/                                 # eval CLIP/LPIPS (cross-arch)
│
├── tools/                                   # config personali + script sync HPC↔Mac
│   ├── set_slurms.sh.example                # template config HPC (per gli SLURM)
│   ├── set_slurms.sh                        # config tua HPC (gitignored)
│   ├── pull_config.sh.example               # template config Mac (per i sync script)
│   ├── pull_config.sh                       # config tua Mac (gitignored)
│   ├── pull_from_hpc.sh                     # HPC → Mac: sliders + outputs (incrementale)
│   └── push_to_hpc.sh                       # Mac → HPC: tutti gli .slurm (incrementale)
│
├── .gitignore   .venv_sam/   __init__.py   README.md
```

## Dove trovare cosa

| Cosa cerchi | Path |
|---|---|
| Libreria LoRA SDXL (`LoRANetwork`, `train_util`, ...) | `sdxl/core/` |
| Libreria LoRA Flux (`LoRANetwork`, `custom_flux_pipeline`, ...) | `flux/core/` |
| Sliders SDXL pre-trained (smiling, age, muscular, ...) | `sdxl/trained_sliders/sliders/` |
| Sliders Flux trainati | `flux/trained_sliders/sliders/` |
| Checkpoint SAM | `mask_SAM/checkpoints/sam_vit_h_4b8939.pth` |
| Script SAM segmentation (cross-arch) | `mask_SAM/segment_with_sam.py` |
| Eval scripts (CLIP/LPIPS) | `metrics/` |
| Script sync HPC ↔ Mac | `tools/pull_from_hpc.sh`, `tools/push_to_hpc.sh` |

## Workflow HPC

La repo è la stessa per tutti (clone Git identico). Ogni utente ha **path personali** su HPC (repo, venv/conda, cache HF). Tutto questo è centralizzato in `tools/set_slurms.sh` (gitignored, ognuno crea il suo dal template).

### First-time setup HPC

```bash
ssh hpc                                        # da Mac (alias SSH configurato in ~/.ssh/config)

cd /home/<your-username>                       # vai nella tua home
git clone https://github.com/edoardopacca/local-concept-sliders.git
cd local-concept-sliders

# 1. Crea il tuo file di config HPC
cp tools/set_slurms.sh.example tools/set_slurms.sh
nano tools/set_slurms.sh
# modifica: FERT_REPO, FERT_HF_CACHE, activate_flux_env(), activate_sdxl_env()

# 2. Crea le cartelle dei pesi (gitignored, non vengono dal clone)
mkdir -p sdxl/trained_sliders/sliders flux/trained_sliders/sliders

# 3. Setup env Python (venv o conda — vedi requirements-*.lock nelle training/)
```

### First-time setup Mac (per usare gli script di sync)

```bash
cd ~/Desktop/local-concept-sliders
cp tools/pull_config.sh.example tools/pull_config.sh
nano tools/pull_config.sh
# modifica: HPC_USER, HPC_HOST, HPC_REPO
```

### Workflow giornaliero

```bash
# 1. Modifichi codice/SLURM sul Mac
git add -A && git commit -m "..." && git push     # sul Mac

# 2. Su HPC, prendi gli aggiornamenti
ssh hpc 'cd /home/<your-username>/FERT_PROJECT/local-concept-sliders && git pull'

# 3. (Opzionale) Sincronizza solo gli .slurm senza passare da git
./tools/push_to_hpc.sh new                        # sul Mac

# 4. Lancia il job (su HPC, dalla repo root)
ssh hpc
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
sbatch sdxl/tasks/baseline/jobs/new_slurm/myjob.slurm

# 5. Quando il job finisce, scarica risultati sul Mac
./tools/pull_from_hpc.sh                          # sul Mac (sliders + outputs)
```

> **Nota**: gli SLURM hanno `source $SLURM_SUBMIT_DIR/tools/set_slurms.sh` quindi `sbatch` va lanciato sempre **dalla root della repo**.
>
> SLURM userà l'**account default** dell'utente loggato (verifica con `sacctmgr show user $USER format=user,defaultaccount`).

## Entry-point principali

I job SLURM esistenti sono in `<task>/jobs/old_slurm/`. I nuovi vanno in `new_slurm/`, gli sperimentali in `test_slurm/`.

### Training SDXL slider

```bash
sbatch sdxl/trained_sliders/training/jobs/old_slurm/<train_*.slurm>
# es: train_smile_man_strong_v2_guidance4.slurm
```
Output: `.safetensors` in `sdxl/trained_sliders/sliders/`.

### Training Flux slider

```bash
sbatch flux/trained_sliders/training/jobs/old_slurm/<train_*.slurm>
# es: train_smile_flux_v5_symmetric.slurm
```
Output: `.safetensors` in `flux/trained_sliders/sliders/`.

### Baseline (generazione + sweep slider)

```bash
sbatch sdxl/tasks/baseline/jobs/old_slurm/<generate_*.slurm>
sbatch flux/tasks/baseline/jobs/old_slurm/<generate_*.slurm>
```
Output: PNG in `<arch>/tasks/baseline/outputs/<run_name>/`.

### Real image editing SDXL (3 fasi)

```bash
# Fase 1 — inversione tight (HPC)
sbatch sdxl/tasks/real_editing/jobs/old_slurm/run_tight_inversion.slurm

# Fase 2 — segmentazione SAM (locale Mac, interattiva)
python mask_SAM/segment_with_sam.py \
    --run_dir sdxl/tasks/real_editing/outputs/<RUN_ID> \
    --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
    --image_name reconstruction.png \
    --mode interactive

# Fase 3 — masked LoRA edit (HPC)
sbatch sdxl/tasks/real_editing/jobs/old_slurm/run_tight_edit.slurm
```

### Masked LoRA SDXL / Flux (3 fasi)

```bash
sbatch <arch>/tasks/masked_lora/jobs/old_slurm/run_phase1.slurm   # base generation
python mask_SAM/segment_with_sam.py --run_dir <arch>/tasks/masked_lora/outputs/<RUN_ID> ...
sbatch <arch>/tasks/masked_lora/jobs/old_slurm/run_phase3.slurm   # masked edit
```

### Shop Concept (multi-LoRA Flux mask-aware)

```bash
sbatch flux/tasks/shop_concept/jobs/old_slurm/generate_shop_concept.slurm
sbatch flux/tasks/shop_concept/jobs/old_slurm/sweep_vangogh_sky.slurm
```

## Sync HPC ↔ Mac (script in `tools/`)

```bash
# Mac → HPC: pusha gli .slurm (incrementale)
./tools/push_to_hpc.sh           # tutti
./tools/push_to_hpc.sh new       # solo new_slurm/
./tools/push_to_hpc.sh test      # solo test_slurm/
./tools/push_to_hpc.sh old       # solo old_slurm/

# HPC → Mac: scarica sliders + outputs (incrementale)
./tools/pull_from_hpc.sh                 # default: lascia anche su HPC
REMOVE_REMOTE=1 ./tools/pull_from_hpc.sh # libera HPC dopo download
```

Entrambi i comandi trasferiscono **solo file nuovi o modificati** (rsync). Vedi [`tools/README.md`](tools/README.md) per i dettagli.

## Convenzioni di import

Tutti gli import sono **assoluti** dalla repo root:

```python
from sdxl.core.lora import LoRANetwork
from flux.core.custom_flux_pipeline import FluxPipeline
from sdxl.tasks.real_editing.lib.models.loader import load_model_context
from flux.tasks.shop_concept.lib.flux_real_pipeline import RealGenerationPipeline
```

Lanciare sempre `python` dalla root della repo (è quello che fanno i job SLURM dopo `cd $FERT_REPO`).

## Note operative

- `.pt` e `.safetensors` sono **gitignored** (sliders trainati sono grandi, rigenerabili). Le sottocartelle `sliders/` hanno `.gitkeep` per essere tracciate vuote.
- Le `outputs/*.png` di ogni task **sono tracciate** (sono evidenza sperimentale).
- `**/logs/*` (stdout/stderr SLURM) ignorati con `.gitkeep` per la cartella.
- `tools/set_slurms.sh` e `tools/pull_config.sh` sono **gitignored**: ognuno mantiene il suo localmente.
