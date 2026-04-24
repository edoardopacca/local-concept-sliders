# tools/

Script e config personali per il workflow HPC ↔ Mac.

## File

| File | A cosa serve | Personale? |
|---|---|---|
| `set_slurms.sh.example` | Template config HPC (path repo, cache HF, funzioni activate env) | template tracked |
| `set_slurms.sh` | Config personale HPC, source dagli SLURM | **gitignored** |
| `pull_config.sh.example` | Template config Mac (HPC user/host/path) | template tracked |
| `pull_config.sh` | Config personale Mac, source da `pull_from_hpc.sh` e `push_to_hpc.sh` | **gitignored** |
| **`pull_from_hpc.sh`** | HPC → Mac: scarica sliders + outputs (incrementale) | tracked |
| **`push_to_hpc.sh`** | Mac → HPC: pusha `.slurm` + configs YAML + prompts YAML (incrementale) | tracked |

## First-time setup

**Su HPC** (per i job SLURM):
```bash
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
cp tools/set_slurms.sh.example tools/set_slurms.sh
nano tools/set_slurms.sh
# modifica:
#   FERT_REPO            (path completo della repo HPC, es. /home/3226571/FERT_PROJECT/local-concept-sliders)
#   FERT_HF_CACHE        (path completo cache HuggingFace)
#   activate_flux_env()  (venv | conda | mamba: vedi esempi commentati nel template)
#   activate_sdxl_env()  (idem)
```

**Sul Mac** (per gli script di sync):
```bash
cd ~/Desktop/local-concept-sliders
cp tools/pull_config.sh.example tools/pull_config.sh
nano tools/pull_config.sh
# modifica:
#   HPC_USER  (es. "3226571")
#   HPC_HOST  (alias SSH "hpc" se hai ~/.ssh/config, altrimenti "slogin.hpc.unibocconi.it")
#   HPC_REPO  (path completo HPC, es. "/home/3226571/FERT_PROJECT/local-concept-sliders")
#   LOCAL_REPO (default: ~/Desktop/local-concept-sliders)
```

## Setup SSH alias (consigliato sul Mac)

Per evitare di scrivere `ssh user@hostname` ogni volta, in `~/.ssh/config` sul Mac:

```
Host hpc
    HostName slogin.hpc.unibocconi.it
    User <your-username>
```

Poi puoi usare `ssh hpc`, `scp file hpc:...`, e gli script qui dentro funzionano con `HPC_HOST="hpc"`.

Setup chiave SSH (una volta sola, per evitare password):
```bash
ssh-keygen -t ed25519                          # genera chiave (se non l'hai)
ssh-copy-id <your-username>@slogin.hpc.unibocconi.it  # carica chiave su HPC
ssh hpc                                        # test: deve entrare senza password
```

## I 2 comandi del workflow

### Mac → HPC: `push_to_hpc.sh`

Pusha dal Mac a HPC i file di configurazione del progetto:
- **`.slurm`** in `**/jobs/{old,new,test}_slurm/`
- **`configs/*.yaml`** in `<arch>/trained_sliders/training/configs/`
- **`prompts/*/*.yaml`** in `<arch>/trained_sliders/training/prompts/{old,new,test}_prompt/`

**Incrementale**: trasferisce solo file nuovi/modificati. Mantiene la struttura della repo.

```bash
./tools/push_to_hpc.sh           # default: TUTTO (slurm + configs + prompts)
./tools/push_to_hpc.sh slurm     # solo .slurm
./tools/push_to_hpc.sh new       # solo new_slurm/
./tools/push_to_hpc.sh test      # solo test_slurm/
./tools/push_to_hpc.sh old       # solo old_slurm/
./tools/push_to_hpc.sh configs   # solo configs/*.yaml
./tools/push_to_hpc.sh prompts   # solo prompts/*/*.yaml
./tools/push_to_hpc.sh yaml      # configs + prompts (no slurm)
```

### HPC → Mac: `pull_from_hpc.sh`

Scarica sliders trainati + immagini di output da HPC al Mac. **Incrementale**.

```bash
./tools/pull_from_hpc.sh                 # default: scarica tutto, lascia su HPC
DRY_RUN=1 ./tools/pull_from_hpc.sh       # anteprima (no operazioni)
REMOVE_REMOTE=1 ./tools/pull_from_hpc.sh # scarica E libera HPC
```

## Cosa significa "incrementale"

Sotto il cofano è `rsync` che confronta size + mtime. I file identici vengono **skippati silenziosamente**, vengono trasferiti solo quelli nuovi o modificati. Output:

- `>f+++++++++ x.slurm` → file nuovo (creato su HPC)
- `>f.st...... x.slurm` → file modificato (size/timestamp diversi)
- (riga assente) → file identico, skip

## Workflow tipico

```bash
# 1. Modifichi/crei nuovi .slurm sul Mac
# 2. Pushali su HPC
./tools/push_to_hpc.sh new

# 3. Lanci i job (sul HPC)
ssh hpc
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
sbatch flux/tasks/<task>/jobs/new_slurm/myjob.slurm
exit

# 4. Quando i job finiscono, scarichi tutto sul Mac
./tools/pull_from_hpc.sh
```

## Struttura path Mac ↔ HPC

I path **dentro la repo** sono identici. Cambia solo il **prefisso assoluto**:

```
Mac:  ~/Desktop/local-concept-sliders/                    flux/tasks/baseline/outputs/myrun/img.png
HPC:  /home/<your-username>/FERT_PROJECT/local-concept-sliders/   flux/tasks/baseline/outputs/myrun/img.png
                          ↑ unica differenza (LOCAL_REPO vs HPC_REPO)
```

I prefissi sono settati nei tuoi config personali:
- `LOCAL_REPO` in `tools/pull_config.sh` (Mac)
- `FERT_REPO`  in `tools/set_slurms.sh` (HPC)

Gli script costruiscono dinamicamente i path completi a partire da queste variabili.
