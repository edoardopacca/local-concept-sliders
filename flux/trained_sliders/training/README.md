# `flux/trained_sliders/training/`

Training framework for **Flux.1-dev** Concept Sliders. The sliders
trained here are the inputs to every Flux evaluation in the paper:

- The five concepts evaluated under the mask-guided pipeline of §4 on
  Flux: ``age``, ``smile``, ``curly hair``, ``fur length``, ``painterly``
  (the appendix table `tab:appendix:results-masked-flux`).
- The four concepts used to produce the qualitative LoRAShop-style
  demos in `flux/tasks/shop_concept/` (Appendix D).

The training stage is **not** a contribution of this work. We use the
upstream Concept Sliders training recipe (Gandikota et al., 2023)
unchanged; the paper's contribution is the inference-time mask-guided
localisation built on top of these adapters (§4).

## Origin

The training script `scripts/train_flux_slider.py` is a linearised port
of the upstream `flux-sliders/train-flux-concept-sliders.ipynb` notebook
(part of [rohitgandikota/sliders](https://github.com/rohitgandikota/sliders),
MIT licensed), with three additions to make the run fit on a single
~40 GB GPU:

1. **T5 + CLIP offload after embedding**: the text encoders are evicted
   from GPU as soon as the (target, positive, negative) embeddings are
   computed; the `pipe(...)` call inside the training loop is invoked
   with `prompt_embeds=` and `pooled_prompt_embeds=` so the encoders are
   not needed again.
2. **Gradient checkpointing on the transformer**, enabled by default.
3. **CLI overrides** for `max_train_steps`, `output_dir`, `slider_name`,
   `target/positive/negative_prompt`, `rank`, ..., and a YAML-driven
   multi-prompt mode (`--prompts_yaml`) that mirrors the SDXL trainer.

The LoRA stack lives in [`flux/core/lora.py`](../../core/lora.py) and
is also adapted from the upstream Concept Sliders code.

## Layout

```
flux/trained_sliders/training/
├── scripts/
│   └── train_flux_slider.py        # main training entrypoint
├── prompts/
│   ├── new_prompt/                 # prompt sets used for the paper runs
│   └── old_prompt/                 # earlier prompt iterations and upstream copies
├── jobs/
│   └── new_slurm/                  # SLURM templates used for the paper
├── setup_flux_venv.sh              # bootstrap a Python venv with all dependencies
├── setup_sliders_flux_conda.sh     # bootstrap a conda env (alternative)
├── flux-requirements.txt           # minimal pip requirements
├── requirements-flux.lock          # exact pinned snapshot (for reproducibility)
└── README.md                       # this file
```

The trained slider checkpoints live under
[`flux/trained_sliders/sliders/`](../sliders/). The final sliders used
by the paper experiments are committed under `general/` via a whitelist
in `.gitignore`; any other checkpoint produced by training is
git-ignored and can be regenerated. Downstream tasks
(`flux/tasks/masked_lora/`, `flux/tasks/shop_concept/`) convert each
`.pt` to the PEFT safetensors format on the fly via
`flux/tasks/shop_concept/scripts/convert_slider_to_peft.py`.

## How a slider is defined

A Flux slider can be specified in one of two ways:

1. **Single-triple CLI**: three prompts passed directly on the command
   line via `--target_prompt`, `--positive_prompt`, `--negative_prompt`.
   Used by the simplest training jobs (e.g. `train_smile_flux_v1_custom`).
2. **Multi-entry YAML**: a `--prompts_yaml <file>` pointing to a list of
   entries with the same four-field schema as the SDXL trainer
   (``target / positive / unconditional / neutral / guidance_scale``).
   Used to mix several training prompts (e.g. multiple ethnicities).

Either way, the LoRA learns the direction
``(positive - unconditional)`` applied as a shift from ``neutral``, and
active only when conditioned on ``target``. The four-field semantics
are identical to those documented in
[`sdxl/trained_sliders/training/README.md`](../../../sdxl/trained_sliders/training/README.md).

## Environment

The script targets Python 3.10+, PyTorch 2.4.1 with CUDA 12.4 and
diffusers 0.31+. Two bootstrap scripts are provided:

```bash
# Plain venv:
bash flux/trained_sliders/training/setup_flux_venv.sh

# Or conda (e.g. on H100 / H200):
bash flux/trained_sliders/training/setup_sliders_flux_conda.sh
```

Both scripts write a lock file (`requirements-flux.lock` or
`requirements-sliders-flux-conda.lock`) capturing the exact installed
versions for later reproducibility.

The training script installs a torch 2.4 / diffusers 0.36 compatibility
shim on `F.scaled_dot_product_attention` (it drops the `enable_gqa`
kwarg that newer diffusers passes unconditionally but only exists on
torch >= 2.5). The shim is a no-op on the Flux attention layout.

## Usage

From the repository root:

```bash
# Train one slider directly:
python flux/trained_sliders/training/scripts/train_flux_slider.py \
       --max_train_steps 500 \
       --output_dir flux/trained_sliders/training/outputs/<your_slider> \
       --slider_name <your_slider> \
       --target_prompt   "picture of a person" \
       --positive_prompt "photo of a person, smiling, happy" \
       --negative_prompt "photo of a person, frowning" \
       --rank 16 --alpha 1 --train_method xattn \
       --lr 0.002 --lr_warmup_steps 200 \
       --save_every 100

# Or via the bundled SLURM template:
sbatch flux/trained_sliders/training/jobs/new_slurm/train_smile_flux_v1_custom.slurm
```

The output directory will contain `slider_0.pt` (final checkpoint),
plus intermediate `step{100,200,300,400}/slider_0.pt` checkpoints when
`--save_every` is set.

A typical Flux training run with `rank 16`, `train_method xattn` and
500 steps takes ~45-70 min on a single A100-class GPU.

## Useful parameters

| Flag | Default | Meaning |
|---|---|---|
| `--max_train_steps` | 500 | Number of training iterations. |
| `--rank` | 16 | LoRA rank. |
| `--alpha` | 1.0 | LoRA alpha. |
| `--train_method` | `xattn` | Which layers receive the LoRA (`xattn`, `noxattn`, `full`). |
| `--eta` | 2.0 | Concept Sliders objective boost weight. |
| `--lr` | 0.002 | AdamW learning rate. |
| `--lr_warmup_steps` | 200 | Warmup steps for the LR schedule. |
| `--save_every` | 0 | Save an intermediate checkpoint every N steps (0 = final only). |
| `--prompts_yaml` | none | If set, overrides the three CLI prompts and enables multi-prompt sampling. |
| `--guidance_scale` | 3.5 | Flux distilled guidance used during the on-the-fly base generation inside the loss. |

## Adding a new slider

The minimum recipe:

1. Decide whether the slider needs the single-triple or multi-entry
   prompt mode.
2. (Multi-entry only) Create `prompts/new_prompt/<exp>.yaml` with the
   list of entries.
3. Create `jobs/new_slurm/train_<exp>.slurm` by copying an existing
   template and editing the slider name, the output dir, the prompt
   arguments (or the `--prompts_yaml` path) and the LoRA
   hyperparameters as needed.
4. Run the smoke test first, then submit the real job.

No script-level code changes are required for a new slider.
