# Come ricreare il venv `sliders` (SDXL)

Questo documento descrive come ricreare il venv Python usato per il training
degli SDXL Concept Sliders (v2/v3), **identico** a quello originale.
È stato scritto prima di cancellare il venv per fare spazio al lavoro su
Flux.1-dev.

## Stato al momento del dump

| Campo | Valore |
|-------|--------|
| Data snapshot | 2026-04-19, ~03:00 |
| Git commit | `127f3c0` ("diag: SDXL baseline with explicit compositional prompts at scale=0") |
| Full hash | `127f3c009c031930efd40f7fede3bb0a18e3f974` |
| Python | 3.9.21 |
| Torch | 2.0.1 (CUDA 11.7, wheel `cu117` via libs nvidia-*-cu11 bundled) |
| Diffusers | 0.31.0 |
| Transformers | 4.44.2 |
| bitsandbytes | 0.41.1 |
| xformers | 0.0.21 |
| Venv path originale | `~/Linux4HPC/venvs/sliders` |
| Repo path | `/home/<your-username>/FERT_PROJECT/local-concept-sliders` |

Nota su CUDA: nessun `module load cuda/...` era necessario perche' torch 2.0.1
viene installato insieme ai pacchetti `nvidia-*-cu11` (runtime bundled). Le
versioni di CUDA 11.7 elencate nel lock file provengono da li'. Se in futuro
HPC rimuove le dependencies bundled, puo' servire `module load cuda/11.7` (o
la 11.x piu' vicina disponibile su HPC).

## Ricreare il venv in 5 comandi

```bash
# 1) Attiva Python 3.9 (se non disponibile di default, prova con miniconda3)
python3 --version    # dovrebbe mostrare 3.9.x
# Se non e' 3.9, in alternativa:
#   module load miniconda3
#   conda create -n sliders-py39 python=3.9 -y
#   conda activate sliders-py39
#   # poi usa `python -m venv` se vuoi isolare, oppure continua nell'env conda

# 2) Crea il venv
python3 -m venv ~/Linux4HPC/venvs/sliders

# 3) Attivalo e aggiorna pip
source ~/Linux4HPC/venvs/sliders/bin/activate
pip install --upgrade pip

# 4) Installa le versioni pinnate dal lock file
pip install -r /home/<your-username>/FERT_PROJECT/local-concept-sliders/training_local_concept_sliders/SDXL_train/requirements-sdxl.lock

# 5) Verifica
python -c "
import torch, diffusers, transformers, bitsandbytes, xformers
print(f'torch {torch.__version__}, cuda avail: {torch.cuda.is_available()}')
print(f'diffusers {diffusers.__version__}')
print(f'transformers {transformers.__version__}')
print(f'bitsandbytes {bitsandbytes.__version__}')
print(f'xformers {xformers.__version__}')
"
```

## Ricreare anche il modello base (SDXL)

Oltre al venv serve anche il modello base SDXL che e' stato cancellato per
fare spazio a Flux. Per riscaricarlo:

```bash
source ~/Linux4HPC/venvs/sliders/bin/activate

export HF_HOME=/home/<your-username>/FERT_PROJECT/Caches_and_venvs/hf_cache
export HF_HUB_CACHE=/home/<your-username>/FERT_PROJECT/Caches_and_venvs/hf_cache/hub

# Serve un token HF (non gated per SDXL, ma serve per aver accesso normale)
huggingface-cli login

# Scarica SDXL base 1.0 (~14 GB)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
```

## Checkout del codice alla versione esatta

```bash
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
git fetch --all
git checkout 127f3c0
```

Da qui il codice in `training_local_concept_sliders/SDXL_train/scripts/` corrisponde
esattamente a quello usato per i training SDXL v2/v3 precedenti.

## Cosa resta anche se cancelli il venv

I risultati del training (i pesi `.safetensors` degli slider addestrati)
**NON vivono nel venv** — sono file persistenti in
`training_local_concept_sliders/SDXL_train/outputs/`. Questi vanno SEMPRE preservati:

```
training_local_concept_sliders/SDXL_train/outputs/
  smile_man_strong_v3_yaml_trick_alpha1.0_rank4_noxattn/
    smile_man_strong_v3_yaml_trick_alpha1.0_rank4_noxattn_500steps.safetensors
    smile_man_strong_v3_yaml_trick_alpha1.0_rank4_noxattn_last.safetensors
  smile_man_strong_v3_preserve_woman_alpha1.0_rank4_noxattn_lam1.0/
    smile_man_strong_v3_preserve_woman_alpha1.0_rank4_noxattn_lam1.0_500steps.safetensors
    smile_man_strong_v3_preserve_woman_alpha1.0_rank4_noxattn_lam1.0_last.safetensors
```

Cancellare il venv non tocca queste cartelle. I risultati della tesi sono al
sicuro.

## Tempi stimati per la ricreazione completa

| Step | Tempo |
|------|-------|
| `python -m venv` + upgrade pip | ~30 s |
| `pip install -r requirements-sdxl.lock` | 8-15 min (torch + xformers sono i piu' lenti) |
| Download SDXL base | 10-15 min |
| **Totale** | **~20-30 min** |

## Quando si potrebbe volerlo ricreare

- Per riprodurre un esperimento SDXL vecchio
- Per fare il fallback "v4 mask-on-loss" su SDXL se Flux non funziona
- Per generare nuove ablation / figure finali dalla tesi usando gli slider
  gia' addestrati (inference, non training)

## Alternative: conda invece di venv

Se in futuro si vuole un ambiente piu' robusto, conda con miniconda3
(modulo `miniconda3` disponibile su HPC) puo' gestire Python 3.9 e le libs
di sistema in modo piu' pulito:

```bash
module load miniconda3
conda create -n sliders-py39 python=3.9 -y
conda activate sliders-py39
pip install -r /home/<your-username>/FERT_PROJECT/local-concept-sliders/training_local_concept_sliders/SDXL_train/requirements-sdxl.lock
```

Il resto dei comandi e' identico.
