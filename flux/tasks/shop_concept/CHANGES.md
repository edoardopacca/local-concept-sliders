# CHANGES — shop_concept vs LoRAShop-main

Tracciamento di ogni differenza tra `shop_concept/` e la repo ufficiale
`LoRAShop-main/`. Il riferimento e' la copia locale presente in
`LoRAShop-main/` (commit che Federico ha scaricato).

Questa cartella e' self-contained: non richiede file al di fuori di se
stessa a parte le dipendenze Python standard (torch, diffusers, peft,
safetensors, opencv-python, scipy, matplotlib, transformers).

## File aggiunti rispetto a LoRAShop-main

Questi file non hanno corrispondente in LoRAShop-main.

### `__init__.py`
Marker di package. Permette di usare import relativi
(`from .flux_blocks import ...`) e di eseguire `python -m shop_concept.generate`.

Contiene anche una **compat shim** per `torch.nn.functional.scaled_dot_product_attention`:
diffusers >= 0.36 passa `enable_gqa=...` al SDPA, ma quel kwarg esiste solo
in torch >= 2.5. Il venv `flux-sliders` su HPC Bocconi ha `torch==2.4.1+cu124`,
quindi senza shim qualsiasi `FluxPipeline.__call__` crasha con
`TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`.
La shim wrappa `F.scaled_dot_product_attention` scartando `enable_gqa` prima
della chiamata. Attiva solo se torch non supporta il kwarg; su torch >= 2.5
e' no-op. `enable_gqa=False` e' il default semantico per Flux (stessa head
count su q/k/v), quindi dropparlo non cambia il risultato numerico
dell'attention.

### `convert_slider_to_peft.py`
Converter standalone: prende in input un `slider_X.pt` (output di
`LoRANetwork.save_weights()` dal training dei Concept Sliders Flux,
formato kohya-ss) e produce un `.safetensors` PEFT caricabile da
`FluxLoraLoaderMixin.load_lora_weights`.

Dettagli chiave:
  * Enumera staticamente i 266 moduli LoRA attesi dall'architettura
    Flux 1.0 col training `train_method='xattn'` (19 double blocks x 8
    linear attn + 38 single blocks x 3 linear attn). Non richiede di
    caricare Flux per costruire la mappa.
  * Rinomina `lora_unet_<flat_path>.lora_down.weight` ->
    `transformer.<dotted_path>.lora_A.weight`.
  * Rinomina `lora_unet_<flat_path>.lora_up.weight` ->
    `transformer.<dotted_path>.lora_B.weight`.
  * Default `fold_alpha=True`: moltiplica `lora_B` per `alpha/rank` e
    scarta il tensor `alpha`. Dopo la conversione, PEFT `scaling=1.0`
    equivale a "slider a piena forza di training" (lora_scale=1.0).
    Questo semplifica la logica in `flux_blocks.py`.

### `generate.py`
Entrypoint CLI specifico per i Concept Sliders. Sostituisce
`LoRAShop-main/main.py`. Differenze di sostanza:
  * Accetta in input sia `.pt` (auto-convert on-the-fly e cacha in
    `_peft_cache/`) sia `.safetensors` (load diretto).
  * Nuovo parametro `--lora_scales` (lista di float, uno per slider):
    la scale continua del Concept Slider. Viene propagata alla pipeline
    in `target_lora_scales` (vedi sotto).
  * Non supporta HuggingFace Hub LoRA loading (i Concept Sliders sono
    locali). Tolto per semplicita'; l'originale lo supportava.
  * Tolto il supporto `--cuda_device`; uso la stessa convenzione
    `CUDA_VISIBLE_DEVICES` via slurm.

### `jobs/generate_shop_concept.slurm`
Template slurm per HPC Bocconi. Non ha corrispondente in LoRAShop-main
(la repo originale ha `example_multi_lora.sh` per uso bash locale).
Questo job:
  * Attiva il venv `flux-sliders` e le env HF_HUB_OFFLINE esistenti.
  * Chiama `python -m shop_concept.generate` con array bash per
    `SLIDERS`, `TARGETS`, `SCALES`.

### `sweep.py`
Wrapper CLI per generare una griglia (seed x scale) su un SINGOLO
Concept Slider caricando Flux UNA volta sola. Non ha corrispondente in
LoRAShop-main.

Motivazione: lanciare `generate.py` in un loop bash (come fanno i
`jobs/sweep_*_couple.slurm` v1) paga ~2-3 min di model load per ogni
immagine, perche' Flux.1-dev (~24GB) viene ricaricato da disco a ogni
invocazione del processo Python. Per uno sweep di 8 immagini quello fa
~24 min di overhead inutile su ~5 min di denoise reale. `sweep.py`
carica la pipeline una volta, applica lo slider una volta, e itera
internamente sui (seed, scale) pairs modificando solo
`target_lora_scales` e `generator` tra una chiamata e l'altra.

Riutilizza `ensure_matching_lora_params` e `prepare_slider_as_safetensors`
da `generate.py` per non duplicare logica.

Limite: single slider, single target. Multi-slider sweep e'
N-dimensionale e richiederebbe un design ad hoc (tutti gli slider
scalano insieme? scale indipendenti per ogni slider? ogni combinazione
incrociata?), quindi per ora se serve multi-slider si usa `generate.py`
diretto. Gli slurm che usano questo wrapper sono
`jobs/sweep_*_couple_v2.slurm`.

### `README.md`
Documentazione di utilizzo. Non ha corrispondente diretto (la repo
originale ha un README focalizzato su LoRAShop, non sui Concept Sliders).

### `CHANGES.md`
Questo file. Tracciamento delta vs LoRAShop-main.

## File derivati (copia con modifiche)

### `utils.py`
COPIA IDENTICA di `LoRAShop-main/utils.py`. Nessuna modifica funzionale.
Riportato qui solo per self-containment del package.

### `flux_blocks.py`
Derivato da `LoRAShop-main/flux_blocks.py` con le seguenti modifiche:

  1. **`set_adapter` estesa con parametro `target_lora_scales`**
     (in entrambe le classi `TransformerBlock` e `SingleTransformerBlock`).
     Nuova firma:
     ```python
     def set_adapter(self, module_to_set, adapter_idx, target_lora_scales=None)
     ```
     Se `target_lora_scales` e' passato, oltre a selezionare l'adapter
     PEFT `default_{adapter_idx}` imposta anche
     `module.scaling[adapter_name] = target_lora_scales[adapter_idx]`.
     In `LoRAShop-main/flux_blocks.py` originale, `set_adapter` si
     limitava a `module.set_adapter(f"default_{adapter_idx}")` senza
     toccare la scaling: la scaling restava quella di default PEFT
     (derivata da alpha del safetensors), cioe' 1.0 per LoRA tipici
     da HuggingFace Hub. Per i Concept Sliders con lora_scale continuo
     (0.5, 1.5, ...) serve questa modifica.

  2. **Factoring out del helper `_set_adapter_with_scale`**
     (funzione a livello modulo). Nell'originale il body di
     `set_adapter` era duplicato identico nelle due classi
     `TransformerBlock` e `SingleTransformerBlock`. Qui il codice e'
     condiviso via helper funzione.

  3. **Nelle chiamate interne a `set_adapter(...)` in
     `forward_blend_block`**, passiamo sempre
     `target_lora_scales = joint_attention_kwargs.get("target_lora_scales")`.
     Se la key non e' in kwargs (uso alla LoRAShop "vanilla"), torna
     `None` e il comportamento e' identico all'originale.

  4. **Compat signatures per diffusers >= 0.36**
     (hotfix aggiunto dopo test su HPC Bocconi).
       * `TransformerBlock.forward(..., **kwargs)`: aggiunto `**kwargs`
         finale per assorbire eventuali kwargs addizionali che diffusers
         0.36 propaga ai block (es. attention_mask, controlnet_*).
       * `SingleTransformerBlock.forward`: aggiunto `encoder_hidden_states`
         come kwarg + `**kwargs`, e cambiato il return. In diffusers 0.31
         i single block ricevevano una singola `hidden_states` gia'
         concatenata [enc; img] dal caller e restituivano un singolo
         tensor. In 0.36 ricevono `encoder_hidden_states` separato e
         devono restituire tuple `(encoder_hidden_states_out, hidden_states_out)`.
         Implementato come shim: se `encoder_hidden_states is not None`,
         concatena all'inizio (preservando la logica interna delle
         `forward_*` invariate), split e ritorno tuple alla fine. Se e'
         `None` (uso legacy / prior-extract da pipeline nostra), return
         single tensor come prima — backward compatible.

Nulla altro e' stato modificato: `enable_lora_all`, `disable_lora_all`,
`calc_attention`, `calc_attention_mask`, `forward_prior_extract`,
`forward_block`, `forward_block_invert` restano byte-per-byte identici
salvo whitespace. Il dispatcher `forward` ha solo assorbito i nuovi kwargs
e, per il single block, i concat/split di compat.

### `flux_real_pipeline.py`
Derivato da `LoRAShop-main/flux_real_pipeline.py` con le seguenti
modifiche:

  1. **Import relativi**: `from utils import ...` diventa
     `from .utils import ...`; `from flux_blocks import ...` diventa
     `from .flux_blocks import ...`. Necessario per eseguire sia come
     modulo (`python -m shop_concept.generate`) sia per integrazione
     da altri pacchetti.

  2. **Nuovo kwarg `target_lora_scales` in `RealGenerationPipeline.__call__`**.
     Ricevuta una lista di float (una scale per concept slider), viene
     validata contro il numero di `target_prompt` e scritta in
     `joint_attention_kwargs["target_lora_scales"]`. Da li' i
     `TransformerBlock` / `SingleTransformerBlock` la leggono nel loop
     per-target. Se il kwarg e' `None` (default) la chiave non viene
     scritta e il comportamento e' identico a LoRAShop originale
     (scaling=1.0 per tutti gli slider).

Nulla altro e' stato modificato: tutta la logica di encode_prompt,
prepare_latents, register_transformer_blocks, get_substring_tokens,
one_homogeneous_blob, loop denoising, prior-extraction (primi 5
timestep), inversione e visualizzazione rimane byte-per-byte identica.

## File NON portati da LoRAShop-main

Scelta deliberata per mantenere `shop_concept/` minimale e focalizzato.

### `demo.ipynb`
Scelta esplicita (richiesta dell'utente: "il notebook non lo voglio").

### `main.py`
Sostituito da `generate.py` (piu' focalizzato sui Concept Sliders).

### `example.py` / `example_multi_lora.sh`
Sostituiti da `jobs/generate_shop_concept.slurm`.

### `requirements.txt`
Non replicato. `shop_concept` usa gli stessi pacchetti gia' presenti
nel venv `flux-sliders` (diffusers, peft, safetensors, opencv-python,
scipy, matplotlib). Se serve un requirements dedicato in futuro, e'
una copia con eventualmente versioni pinnate.

### `README.md` (la versione originale)
Sostituito da un README proprio focalizzato sull'integrazione Concept
Sliders.

### `assets/`, `LICENSE`, `.gitignore`
Non portati: assets e' documentazione della repo originale; LICENSE
si applica al codice derivato ma va gestito a livello di repo principale
se serve (MIT nell'originale).
