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

## Multi-mask + multi-LoRA per maschera

`03_masked_edit.py` supporta:

  * **Legacy single-mask single-slider** (sweep o singola scale) — modalita'
    storica, comportamento invariato:
    ```
    --slider_path s.pt --mask_name mask.png --slider_scale 2.0
    --slider_path s.pt --mask_name mask.png --slider_scales -2 -1 0 1 2  # sweep
    ```

  * **Multi-mask + multi-LoRA per mask** (composizione paper-style):
    ```
    --slider_paths smile.pt age.pt vangogh.pt
    --mask_names mask_man.png mask_woman.png
    --slider_to_mask 0 0 1            # smile,age -> man ; vangogh -> woman
    --slider_scales 1.0 2.0 0.8       # 1 scale per slider
    ```

Il loop di denoising fa **1 forward base + N forward per-mask** per step
(N = numero di maschere). Per ogni mask attiva i suoi slider come adapter
PEFT contemporaneamente: PEFT somma additivamente le delta nel forward
LoRA (`out = W x + Σ s_i B_i A_i x`) — equivalente al Metodo 2 di Concept
Sliders / ExitStack su LoRANetwork. Il blend velocity finale e'
`v_pred = (1 − Σ mask_i) v_base + Σ (mask_i v_styled_i)`, assumendo
maschere disgiunte (warning se overlap).

Se ti serve applicare lo STESSO concept (es. smile) a regioni diverse con
scale diverse, basta passare lo stesso file slider piu' volte: PEFT lo
carica come adapter distinti (`default_0`, `default_1`, ...) grazie ad
`adapter_name` esplicito in `pipe.load_lora_weights`.

## Workflow multi-mask (3 maschere esempio)

```bash
# 1) Phase 1: genera base.png (su HPC)
sbatch flux/tasks/masked_lora/jobs/new_slurm/phase1_two_subjects.slurm

# 2) Pull su Mac
./tools/pull_from_hpc.sh

# 3) Phase 2: SAM N volte (sul Mac, interattivo)
RUN=flux/tasks/masked_lora/outputs/<RUN_ID>
for name in man woman; do
    python mask_SAM/segment_with_sam.py \
        --run_dir "$RUN" \
        --sam_checkpoint mask_SAM/checkpoints/sam_vit_h_4b8939.pth \
        --image_name base.png --mode interactive \
        --output_name "mask_${name}.png"
done

# 4) Push maschere su HPC (vanno via git: i .png in outputs/ sono tracciati)
git add "$RUN/mask_*.png" && git commit -m "masks for $RUN" && git push
ssh hpc 'cd FERT_PROJECT/local-concept-sliders && git pull'

# 5) Phase 3 multi-compose
sbatch flux/tasks/masked_lora/jobs/new_slurm/phase3_two_subjects_compose.slurm
```

## Relazione con shop_concept/

Riusa:
- `shop_concept/convert_slider_to_peft.py` per caricare gli slider .pt Flux
- lo stesso cache `_peft_cache/` per i .safetensors convertiti
- la stessa convenzione `default_{i}` per gli adapter PEFT (per coerenza
  con `flux_blocks._set_adapter_with_scale`, anche se qui usiamo l'API
  pipeline-level `pipe.set_adapters` invece di toccare i singoli moduli)

Non riusa (volutamente):
- l'attention hooking di `shop_concept/flux_blocks.py` (LoRAShop-style):
  qui vogliamo una strada completamente diversa per confrontare. Le
  maschere SAM sono PIXEL-level e venute da un human-in-the-loop, mentre
  shop_concept estrae le maschere dall'attention dei target_prompt.
