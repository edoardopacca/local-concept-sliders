# CONTEXT DOCUMENT — Spatial Concept Slider Training Setup

> **Purpose of this document:** provide an AI assistant with all the context needed to help implement the training of spatially-aware Concept Sliders. This document must be read in full before starting any work.

> **LANGUAGE**: all code, comments, config files, READMEs, and any project output MUST be in **English**. Chat communication with the user can happen in Italian, but everything that ends up in the repository or files must be in English.

> **YOUR ROLE**: you are not just an executor. You must **guide** the user through the process, **explain** what you are doing and why, **ask** for confirmation on important choices before proceeding. The user is an AI master's student who understands the concepts, but wants to be walked through step-by-step. If you are unsure about something (e.g., a hyperparameter, a prompt choice, a file structure), ask. If you are about to make an architectural decision, explain the alternatives and ask which one they prefer. Do not take anything for granted.

> **APPROACH**: proceed calmly. Do NOT rush into creating files and folders. Before building anything inside `training_local_concept_sliders/`, you must **explain and discuss** with the user:
> - How Concept Slider training works in detail (the 4 prompts, the loss, the training loop)
> - How many prompts are needed for training and why (how many YAML entries, whether few or many are needed, what effect the number of prompts has on slider quality)
> - What kind of prompts are needed (what changes between target/positive/unconditional/neutral, how to write them, what makes a prompt good or bad for training)
> - The fundamental hyperparameters (rank, alpha, iterations, guidance_scale) and what they control
> - How to test whether a slider works
>
> The user wants to **understand** before doing. You will build `training_local_concept_sliders/` together, piece by piece, starting from the first simple experiment and only proceeding when the previous step is clear and validated.

> **LIVING DOCUMENT**: this document is NOT static. As the conversation progresses and you (AI) learn new things — from the user, from errors, from results — you must **update this document** accordingly. If you discover that information here is wrong, correct it. If the user tells you something important that is missing, add it. If a section becomes obsolete, remove or rewrite it. This document must always reflect the current state of knowledge about the project, so that if the conversation is interrupted and a new one starts with this file as context, the new AI has up-to-date and correct information.

---

## 1. WHO WE ARE AND WHAT WE DO

We are the FERT group (Federico, Edoardo, Rebecca, Tommaso), master's students in AI at Bocconi University, Milan. We are working on a Computer Vision project based on the paper **"Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models"** by Gandikota et al. (ECCV 2024).

Original repository: `https://github.com/rohitgandikota/sliders`

---

## 2. THE PROJECT IDEA (read carefully — this is the most important part)

### The problem
The original Concept Sliders have a **global** effect: if you apply the "smile" slider to an image with two people, both of them smile. This is because training uses prompts anchored to a generic "person" (e.g., "person, smiling" / "person, not smiling"), so the semantic direction learned by the LoRA does not distinguish between different subjects in the image.

### Our solution
Create Concept Sliders **anchored to specific subjects** (e.g., "man", "woman", "sky", "fish") by modifying **only the training prompts**, without changing the loss, the architecture, or the inference procedure.

### The key intuition
The original smile slider uses:
```
positive: "person, smiling"     →  negative pole: "person, not smiling"
```
This slider acts on ANY person in the image because the subject is generic ("person").

Our smile slider for men uses:
```
positive: "man, smiling"        →  negative pole: "man, not smiling"
```
The learned direction is anchored to "man". At inference, the slider is applied globally (as in the original paper) but the effect is naturally localized to men in the image, because the semantic direction only concerns that subject.

**There is NO need to describe context** (clothing, locations, scenes). The slider must work WHEREVER the subject appears. If you train with "man, smiling" / "man, not smiling", the slider will work on a man at the beach, in an office, in a group of people — it is not tied to a specific scene.

### Concrete use case (to understand the vision)
A user generates an image with three people: two men and a woman. They like the image, but the woman's expression is not right. With standard sliders they can't do anything — if they apply the "smile" slider, all three people smile. With **our** sliders, the user has:
- Slider "smile woman" → makes only the woman smile
- Slider "smile man" → makes only the men smile
- Slider "age woman" → ages/de-ages only the woman
- etc.

An informed user can **compose** these sliders to locally modify what they want, without masks, without segmentation, without touching the rest of the image.

### How it differs from the original slider (this comparison is the core of the project)

| | Original slider (baseline) | Our slider (improvement) |
|---|---|---|
| Smile | `"person, smiling"` / `"person, not smiling"` | `"man, smiling"` / `"man, not smiling"` |
| Age | `"person, very old"` / `"person, very young"` | `"woman, very old"` / `"woman, very young"` |
| Subject | Generic ("person") | Specific ("man", "woman", "sky", "fish", etc.) |
| Effect | Global on all similar subjects | Localized to the specified subject |
| Context | Not specified | **Not specified** (same!) |

**NOTE**: our prompts do NOT describe contexts, clothing, locations. They are as simple as the original ones — the only difference is that the subject is specific rather than generic.

### Possible FUTURE evolutions (not now!)

In the future, once the base method is validated, we may explore even more specific sliders:
- Slider "smile man with black hair" (even more localized — discriminates between different men)
- Slider "cooked fish" vs "raw fish" (instead of generic "cooked food")
- Sliders with explicit context for complex scenes (compositional prompts)
- Multi-subject sliders with specific scenes

**But all of this is for LATER.** Right now we start with the simplest case: slider anchored to a base subject (man, woman) with no additional context.

### STARTING STRATEGY — calmly, from simple to complex

**Do NOT start creating files immediately.** Before building anything inside `training_local_concept_sliders/`, you (the AI) must explain to the user how training works, discuss prompt choices together, and make sure the user understands every piece. The folder will be built **together**, piece by piece.

**Specifically, BEFORE creating files the AI must explain and discuss:**
- How many prompts are needed for training? (in the original paper the age slider uses only 2 YAML entries — male person and female person. We might need more or fewer. How many are really needed? What effect does having few vs many have?)
- How to write the 4 fields (target, positive, negative_pole, anchor — see naming note below) correctly? What happens if you write them wrong?
- What hyperparameters to use and why? (iterations, guidance_scale, rank, alpha)
- How to verify that training went well?

**First experiment: "smile man" slider**
- Goal: the slider controls smile ONLY on men, not on women
- Simple prompts, no context — the slider must work in any scene
- Example (in the code's YAML format):
  - target: `"man"`
  - positive: `"man, smiling"`
  - unconditional: `"man, not smiling"`
  - neutral: `"man"`
- Comparison: also train the classic baseline slider `"person, smiling"` / `"person, not smiling"` to demonstrate that ours is local and theirs is global

**Order of experiments:**
1. Understand well how training and prompts work (discussion) ← **START HERE**
2. Simple "smile man" slider (method validation)
3. Classic "smile person" baseline (comparison)
4. "smile woman" slider (second validation)
5. "age man" / "age woman" sliders (another concept)
6. Only after: more specific evolutions (context, multi-subject, etc.)

### Note on prompt naming in the code

The paper calls the roles: target, positive concept, negative concept, neutral/anchor. But in the code YAML the fields are called:
- `target` = subject the LoRA acts on (LoRA is active with this prompt)
- `positive` = positive pole of the concept (e.g., "man, smiling")
- `unconditional` = negative pole of the concept (e.g., "man, not smiling") — **WARNING: the name "unconditional" in the code is NOT the empty CFG unconditional! It is the negative pole of the concept.** The naming is unfortunate but that's how it is in the code and we don't change it.
- `neutral` = anchor/base point (typically equal to `target`)

If it helps to reason, think of these fields as: `(target, positive_pole, negative_pole, anchor)`. But in the YAML files use the code names: `target`, `positive`, `unconditional`, `neutral`.

---

## 3. HPC INFRASTRUCTURE AND WORKFLOW

### HPC machine structure
```
~/Linux4HPC/
├── venvs/
│   └── sliders/                 # Python virtualenv (used by SLURM jobs)
├── sliders_demo/
│   ├── hf_cache/                # HuggingFace model cache
│   │   └── hub/                 # SDXL, etc. — $HF_HUB_CACHE points here
│   └── local-concept-sliders/   # ← OUR REPOSITORY (see below)
```

### Repository structure (local-concept-sliders)
```
local-concept-sliders/
├── .git/
├── trainscripts/
│   ├── textsliders/             # ← original concept slider training scripts
│   │   ├── train_lora_xl.py     # main SDXL training script
│   │   ├── train_lora.py        # training script for SD 1.x/2.x
│   │   ├── prompt_util.py       # PromptSettings, PromptEmbedsPair, loss classes
│   │   ├── config_util.py       # RootConfig classes, YAML parsing
│   │   ├── model_util.py        # model loading (load_models_xl)
│   │   ├── train_util.py        # utilities: encode_prompts_xl, diffusion_xl, predict_noise_xl
│   │   ├── lora.py              # LoRANetwork, target modules
│   │   ├── debug_util.py        # check_requires_grad, check_training_mode
│   │   ├── generate_images_xl.py # image generation with slider
│   │   └── data/                # example config and prompt YAMLs
│   │       ├── config-xl.yaml
│   │       ├── prompts-xl.yaml
│   │       └── prompts-person_age_slider_GPT.yaml
│   └── imagesliders/            # image-based training (we don't use this)
├── maskedLORA/                  # previous work (masked editing, legacy)
│   ├── experiments/
│   ├── jobs/                    # .slurm files for masked editing
│   ├── logs/
│   └── runs/
├── real_editing/                # previous work (real image editing)
│   ├── archive/
│   ├── editing/
│   ├── inversion/
│   ├── io/
│   ├── jobs/                    # .slurm files for real editing
│   ├── logs/
│   ├── models/
│   └── runs/
├── exp_generation/              # generation scripts with trained sliders
│   ├── generate_with_sliders.py
│   ├── run_generation.sh        # SLURM job for generation
│   └── logs/
└── training_local_concept_sliders/  # ← TO BE CREATED (see section 5)
```

### How the workflow works (READ CAREFULLY)

**You (AI) work ONLY locally, in the `training_local_concept_sliders/` folder.** You do not have direct access to HPC. The user is the intermediary between you and HPC.

The work cycle is:

1. **You (AI) modify files locally** inside `training_local_concept_sliders/`
2. **You give the user git commands** to push changes to HPC:
   ```bash
   cd ~/path/to/local-concept-sliders
   git add training_local_concept_sliders/
   git commit -m "description of changes"
   git push
   ```
3. **You give the user HPC terminal commands** to launch jobs:
   ```bash
   ssh hpc
   cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
   git pull
   sbatch training_local_concept_sliders/SDXL_train/jobs/train_all.slurm
   ```
4. **The user sends you logs/errors** if something goes wrong
5. **You fix locally** and give new git + terminal commands to retry

### FUNDAMENTAL OPERATIONAL RULES

1. **Do NOT touch ANYTHING outside `training_local_concept_sliders/`.** Do not modify `trainscripts/`, `maskedLORA/`, `real_editing/`, or any other file in the repository. Your scope is EXCLUSIVELY `training_local_concept_sliders/`.

2. **ALWAYS provide terminal commands.** Every time you make a change or create a file, you must also tell the user:
   - The **git** commands to push changes (add, commit, push)
   - The **HPC terminal** commands to execute (ssh, cd, git pull, sbatch, etc.)
   - The commands to **check job status** (squeue, cat logs, etc.)

3. **When the user sends you an error:**
   - Analyze the error
   - Fix the file locally (inside `training_local_concept_sliders/`)
   - Give git commands to push the fix
   - Give HPC commands to retry

4. **SLURM files must be ready.** The .slurm jobs in the `jobs/` folder must be complete and functional. The user just does `sbatch` and everything must start.

### SLURM notes (from real existing .slurm files)
Standard parameters we use:
```bash
#SBATCH --account=3226571
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
```

Environment setup:
```bash
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
source ~/Linux4HPC/venvs/sliders/bin/activate

export HF_HOME=/home/<your-username>/FERT_PROJECT/Caches_and_venvs/hf_cache
export HF_HUB_CACHE=/home/<your-username>/FERT_PROJECT/Caches_and_venvs/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The local path of the SDXL model on the HPC machine is:
```
/home/3226571/Linux4HPC/sliders_demo/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
```

---

## 4. CONCEPT SLIDER TRAINING — HOW THE ORIGINAL WORKS (from real code)

### Main files and their roles

**`trainscripts/textsliders/train_lora_xl.py`** — SDXL training script. Accepts these arguments:
```
--config_file     (required)  Config YAML
--prompts_file    (optional)  Override for prompts file
--alpha           (optional)  LoRA alpha weight
--rank            (optional)  LoRA rank
--device          (optional)  CUDA device index (default 0)
--name            (optional)  Name prefix for saving
--attributes      (optional)  Attributes to prepend to prompts (comma-separated)
```

**`trainscripts/textsliders/prompt_util.py`** — Defines:
- `PromptSettings`: Pydantic class with fields `target`, `positive`, `unconditional`, `neutral`, `action`, `guidance_scale`, `resolution`, `dynamic_resolution`, `batch_size`, `dynamic_crops`
- `PromptEmbedsPair`: contains the 4 embeddings and computes the loss
- `load_prompts_from_yaml(path, attributes)`: loads prompts from YAML

**`trainscripts/textsliders/config_util.py`** — Config with sections: `prompts_file`, `pretrained_model`, `network`, `train`, `save`, `logging`, `other`

### EXACT prompt YAML format (from real file `prompts-xl.yaml`)
```yaml
# Each entry is a dictionary with these fields:
- target: "male person"                    # subject the LoRA acts on
  positive: "male person, very old"        # positive pole (direction to amplify)
  unconditional: "male person, very young" # negative pole (to compute the difference)
  neutral: "male person"                   # conditioning base point
  action: "enhance"                        # "enhance" or "erase"
  guidance_scale: 4                        # direction scale
  resolution: 1024                         # latent resolution
  dynamic_resolution: false
  batch_size: 1
```

### Paper formulation — Eq. 7 and Eq. 8 (from Gandikota et al., ECCV 2024)

The paper defines the Concept Slider objective in two stages:

**Eq. 7 — Basic objective (single prompt pair):**
```
ε_θ*(X, c_t, t) = ε_θ(X, c_t, t) + η (ε_θ(X, c_+, t) − ε_θ(X, c_−, t))
```
Where c_t is the target concept, c_+ is the positive attribute, c_- is the negative attribute, and η is the guidance scale. The LoRA learns a direction that shifts c_t toward more c_+ and less c_-.

**Problem**: a single prompt pair can learn an **entangled** direction (e.g., "old" might also shift race/gender).

**Eq. 8 — Disentangled objective (multiple preservation prompts):**
```
ε_θ*(X, c_t, t) = ε_θ(X, c_t, t) + η Σ_{p∈P} (ε_θ(X, (c_+, p), t) − ε_θ(X, (c_−, p), t))
```
Where **P is a set of preservation prompts** — different variants (e.g., different races, genders) over which the direction is averaged. This way, the LoRA learns ONLY the concept direction, while subject-specific biases cancel out.

**Figure 12 (supplementary)** illustrates this: "Old person" and "Young person" define an entangled direction. By adding preservation prompts ("Old asian person"/"Young asian person", "Old black person"/"Young black person", etc.), the resulting direction is disentangled — it captures only age.

### How Eq. 8 maps to the code

In the code, the Σ over P is realized **implicitly across training iterations**:
- Each YAML entry is a separate `prompt_pair` (one element of the sum)
- At each iteration, ONE prompt_pair is **randomly selected** (line 170-172 of `train_lora_xl.py`)
- Over 1000 iterations, the LoRA learns the AVERAGE direction across all entries
- This is stochastic gradient descent over the sum in Eq. 8

**Example from real files**: the age slider in `prompts-person_age_slider_GPT.yaml` has 10 entries:
```
male white person    → old / young
male black person    → old / young
male indian person   → old / young
male asian person    → old / young
male hispanic person → old / young
female white person  → old / young
female black person  → old / young
...etc (10 total)
```
Each entry provides the same concept direction (old-young) on a different subject variant. The LoRA learns the COMMON direction (age) while race/gender-specific parts cancel out across iterations.

**The `--attributes` flag** automates this: if you pass `--attributes "male, female"`, the code prepends each attribute to ALL 4 fields of EACH entry, multiplying the number of entries. This is a shortcut for creating disentanglement variants.

### IMPORTANT — Semantics of the 4 prompts (ALL 4 ARE REQUIRED)

Each YAML entry has **exactly 4 prompt fields**. This maps to the paper as follows:

| Code YAML field | Paper notation | Role | LoRA active? |
|---|---|---|---|
| `target` | c_t | Subject the LoRA modifies | YES |
| `positive` | c_+ (or (c_+, p)) | Positive pole of the concept | no |
| `unconditional` | c_- (or (c_-, p)) | Negative pole of the concept | no |
| `neutral` | (base point) | Anchor for the loss | no |

**WARNING about naming**: the field `unconditional` in the code has a **double role**:
1. It is the **negative pole c_-** of the concept direction (e.g., "male person, very young")
2. It is ALSO used as the **CFG negative prompt** for all noise predictions in the training loop

This double role happens because in `train_lora_xl.py`, the `unconditional` embedding is always the first argument to `concat_embeddings()` for the CFG pair. However, since positive/neutral/unconditional noise predictions all use `guidance_scale=1`, the CFG negative doesn't affect the result (with scale=1, CFG reduces to just the conditional prediction). Only the initial denoising step uses `guidance_scale=3` where this matters.

**Default values** (from `prompt_util.py`):
- If `positive` is not specified → defaults to `target`
- If `unconditional` is not specified → defaults to `""` (empty string)
- If `neutral` is not specified → defaults to `unconditional`

**How they interact in the training loop:**
1. A partially denoised latent is generated using `target` (with LoRA active, guidance_scale=3)
2. 3 noise predictions are computed **without LoRA** (guidance_scale=1): with `positive`, `neutral`, `unconditional`
3. 1 noise prediction is computed **with LoRA** (guidance_scale=1): with `target` → `target_latents`
4. The loss pushes `target_latents` toward `neutral + guidance_scale * (positive - unconditional)`

So the **learned semantic direction** is `positive - unconditional`, and it is applied starting from `neutral`. The `target` is what the LoRA learns to modify.

### Loss function (from `prompt_util.py`)

**action="enhance":**
```python
loss = MSE(target_latents, neutral_latents + guidance_scale * (positive_latents - unconditional_latents))
```

**action="erase":**
```python
loss = MSE(target_latents, neutral_latents - guidance_scale * (positive_latents - unconditional_latents))
```

Where:
- `target_latents` = noise prediction **with LoRA active**, conditioned on `target`
- `positive_latents` = noise prediction **without LoRA**, conditioned on `positive`
- `unconditional_latents` = noise prediction **without LoRA**, conditioned on `unconditional`
- `neutral_latents` = noise prediction **without LoRA**, conditioned on `neutral`

### Training loop (from `train_lora_xl.py`)
1. Load SDXL (tokenizers, text_encoders, unet, noise_scheduler) — all **frozen**
2. Create `LoRANetwork` with specified rank/alpha, type "c3lier", training_method "noxattn"
3. Pre-compute and cache all prompt embeddings (text_embeds + pooled_embeds for SDXL)
4. For each iteration (default 1000):
   a. **Randomly select ONE prompt_pair** from the list (this is how Eq. 8's Σ is approximated via SGD)
   b. Generate random timestep (1 to max_denoising_steps-1, default 1-49)
   c. Create initial noisy latents at the resolution specified in the prompt_pair
   d. **With LoRA active**: perform partial denoising from pure noise using (unconditional, target) embeddings with guidance_scale=3
   e. **Without LoRA** (guidance_scale=1): predict noise conditioned on `positive`, `neutral`, `unconditional` separately
   f. **With LoRA active** (guidance_scale=1): predict noise conditioned on `target` → `target_latents`
   g. Compute enhance/erase loss
   h. Backprop and optimize (single LoRA update)
5. Save checkpoint every `per_steps` steps and at the end

### Two styles of prompt files in the original repo

The original repo contains TWO styles of prompt YAML files:

**Style 1 — Minimal (hand-written by authors):**
- File: `prompts-xl.yaml`, `prompts.yaml`
- Few entries (typically 2: male person + female person)
- Short prompts: `"male person, very old"` / `"male person, very young"`
- Minimal disentanglement (only male/female)
- Example concepts: age, smile, beard, glasses, hair, muscular, makeup, cartoon, etc.
- Some concepts use **multiple synonyms** in positive/negative poles: e.g., muscular uses `"muscular, strong, biceps, greek god physique, body builder"` vs `"lean, thin, weak, slender, skinny, scrawny"`

**Style 2 — Expanded (GPT-generated for disentanglement):**
- Files: `prompts-person_age_slider_GPT.yaml`, `prompts-smile_slider_GPT.yaml`, etc.
- Many entries (typically 10: 5 ethnicities × 2 genders)
- More descriptive prompts with synonyms: `"smiling, happy face, big smile"` / `"frowning, grumpy, sad"`
- Full disentanglement across race and gender
- This is the implementation of Eq. 8's preservation set P

### Example training command (reconstructed from code)
```bash
python trainscripts/textsliders/train_lora_xl.py \
    --config_file trainscripts/textsliders/data/config-xl.yaml \
    --name "age_slider" \
    --rank 4 \
    --alpha 1.0 \
    --device 0
```
This trains the age slider using the 2-entry prompts in `prompts-xl.yaml`. Output: `./models/age_slider_alpha1.0_rank4_noxattn/age_slider_alpha1.0_rank4_noxattn_last.safetensors`

### Evaluation prompts (in `archive/prompts/`)

The repo also contains CSV files for **evaluation** (not training). These have hundreds of entries with format `case_number, prompt, evaluation_seed, concept` — e.g., 100 variants of "image of a person" / "photo of a person" / "portrait of a person" with fixed seeds. These are used to generate reproducible test images with and without the slider, to measure its effect quantitatively (CLIP score, LPIPS, interference).

### Model: SDXL (Stable Diffusion XL)

**We train EXCLUSIVELY on SDXL** (stabilityai/stable-diffusion-xl-base-1.0). Not on SD 1.5 or SD 2.x. This is important because:
- SDXL uses **two text encoders** (CLIP ViT-L + OpenCLIP ViT-bigG) → richer embeddings
- SDXL has native resolution **1024×1024** → latent resolution 128×128, more spatial detail
- The script `train_lora_xl.py` is SDXL-specific (uses `PromptEmbedsXL` with text_embeds + pooled_embeds)
- The LoRA type `c3lier` includes convolutional layers (`UNET_TARGET_REPLACE_MODULE_CONV`), XL-specific

**Prompts must use `resolution: 1024`** (SDXL native resolution), NOT 512.

### Standard SDXL training config (from `config-xl.yaml`)
```yaml
prompts_file: "training_local_concept_sliders/SDXL_train/prompts/smile_man.yaml"
pretrained_model:
  name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
  v2: false
  v_pred: false
network:
  type: "c3lier"          # includes conv layers (XL-specific)
  rank: 4
  alpha: 1.0
  training_method: "noxattn"
train:
  precision: "bfloat16"
  noise_scheduler: "ddim"
  iterations: 1000
  lr: 0.0002
  optimizer: "AdamW"
  lr_scheduler: "constant"
  max_denoising_steps: 50
save:
  name: "smile_man"
  path: "training_local_concept_sliders/outputs"
  per_steps: 500
  precision: "bfloat16"
logging:
  use_wandb: false
  verbose: false
other:
  use_xformers: true
```

### FUNDAMENTAL RULE: the training is IDENTICAL to the original

**We use EXACTLY the same training as the original Concept Sliders.** Same script (`train_lora_xl.py`), same loss, same LoRA architecture, same hyperparameters, same optimizer. We do NOT modify ANYTHING in the training code.

The ONLY thing that changes is the **text content of the 4 prompts** (target, positive, unconditional, neutral) in the YAML files. The mechanism with all 4 prompts stays identical — we simply write different prompts (subject-specific instead of generic) that produce a semantic direction naturally localized to that subject.

What stays the same:
- The training script (`train_lora_xl.py`)
- The loss function (MSE with enhance/erase)
- The LoRA architecture (c3lier, rank 4, alpha 1.0, noxattn)
- The optimization procedure (AdamW, lr=0.0002, 1000 iter)
- The prompt YAML format (all 4 fields: target, positive, unconditional, neutral)
- The way LoRA weights are saved/loaded (.safetensors)
- **The only thing that changes is the TEXT of the prompts**

---

## 5. WHAT TO DO: build `training_local_concept_sliders/`

### Goal
Create a **fully self-contained** folder inside `local-concept-sliders/` containing everything needed to train our spatial Concept Sliders. Anyone opening this folder should immediately understand what's there and how to use it.

### Required structure
```
training_local_concept_sliders/
├── README.md                    # Clear explanation of what's here and how to use it
├── configs/                     # Config files for each experiment
├── prompts/                     # Training prompt sets (YAML)
├── scripts/                     # All code (self-contained)
│   ├── train.py                 # Training script (adapted copy of train_lora_xl.py)
│   ├── generate_test.py         # Generate test images with a trained slider
│   ├── visualize_direction.py   # Visualize direction heatmaps
│   └── [utility modules]        # lora.py, model_util.py, etc.
├── jobs/                        # SLURM job files
├── logs/                        # Job output (initially empty)
├── outputs/                     # Trained LoRA weights (initially empty)
└── results/                     # Generated test images (initially empty)
```

The exact folder/file structure will be decided together with the user as we build it step by step. Do not create the full structure upfront — build incrementally.

### Fundamental rules

1. **Self-contained**: everything needed for training must be inside this folder. The `train.py` script must work standalone. The only external dependencies are the virtualenv and HF model cache. Utility files (`lora.py`, `model_util.py`, `train_util.py`, `prompt_util.py`, `config_util.py`, `debug_util.py`) must be copied into `scripts/` to guarantee independence.

2. **Based on the original**: the training script must start from `trainscripts/textsliders/train_lora_xl.py`. Do NOT rewrite from scratch. Copy, adapt, and modify only what's needed (mainly import paths).

3. **Config-driven**: each experiment is defined by a config file. To run a different training, just change the config file, not the code. The config points to the prompt file via the `prompts_file` field.

4. **Prompts as data**: training prompts are in separate YAML files, in the exact original format (target, positive, unconditional, neutral, action, guidance_scale, resolution, batch_size). Do not hardcode prompts in the script.

5. **Intuitive structure**: anyone opening the folder should understand in 30 seconds what's there and how it works. The README must explain everything.

### How to proceed (step by step) — CALMLY

**IMPORTANT: do not create everything at once. Proceed one step at a time, explaining and asking the user for confirmation at each step.**

**PHASE 0 — Understand (BEFORE creating any file)**

Before touching any file, explain to the user and discuss together:

0a. **How Concept Slider training works**: explain the role of the 4 prompts, the loss, the training loop. Make sure the user has the mechanism clear.

0b. **How many prompts are needed and why**: in the original paper the age slider uses only 2 YAML entries (one for "male person", one for "female person"). Explain: are few or many prompts needed? What effect does the number of prompts have on slider quality? How many entries do we need for our first simple experiment?

0c. **How to write prompts correctly**: explain the rules for writing target/positive/unconditional/neutral effectively. What happens if prompts are too similar? Too different? If neutral is wrong?

0d. **Hyperparameters**: explain what rank, alpha, iterations, guidance_scale control, and whether we should change them from defaults for the first experiment.

0e. **How to test results**: how to verify a slider works? How to generate with a trained slider? How to compare baseline vs spatial?

**PHASE 1 — First simple experiment**

Only AFTER the user has understood everything, proceed with:

1a. Read the original training scripts in `trainscripts/textsliders/` (key files: `train_lora_xl.py`, `prompt_util.py`, `config_util.py`, `lora.py`, `model_util.py`, `train_util.py`).

1b. Read existing .slurm files in `maskedLORA/jobs/` and `real_editing/jobs/`.

1c. **Discuss with the user** the prompts for the first experiment (smile man). How many entries? What text? Ask for confirmation.

1d. Create ONLY what's needed for the first experiment: the basic folder structure, the training script, the prompt file, the config, the SLURM job.

1e. Give the user git + HPC commands to launch training.

1f. Wait for results/errors from the user.

**PHASE 2 — Iterate and expand**

Only AFTER the first experiment works:

2a. Create the baseline (classic smile slider) for comparison.

2b. Create more subject-specific sliders (smile woman, age man, age woman).

2c. Add visualization and testing scripts.

2d. Write the final README.

2e. Only later: explore more specific evolutions (context, multi-subject, etc.)

---

## 6. PROMPT DETAILS

### Initial concepts to train
- **smile** (smiling/not smiling) — first concept, simplest for validation
- **age** (old/young) — second concept

Initial subjects: **man**, **woman** (separate, not generic "person").

### Prompt format in the code (follow this format EXACTLY)

Each YAML entry has 4 prompt fields + parameters:
- `target` = the base subject (anchor — LoRA is conditioned on this)
- `positive` = the subject + positive pole of the concept
- `unconditional` = the subject + negative pole of the concept (WARNING: in the code it's called "unconditional" but it's the negative pole, NOT the empty CFG unconditional)
- `neutral` = equal to target (conditioning base point)

**The learned direction is: positive - unconditional. The target is pushed toward neutral + guidance_scale * (positive - unconditional).**

### Example — Original baseline slider (generic "person" subject — GLOBAL effect)
```yaml
- target: "person"
  positive: "person, smiling"
  unconditional: "person, not smiling"
  neutral: "person"
  action: "enhance"
  guidance_scale: 4
  resolution: 1024
  dynamic_resolution: false
  batch_size: 1
```
This slider will make EVERYONE in the image smile.

### Example — Our slider (specific "man" subject — LOCAL effect)
```yaml
- target: "man"
  positive: "man, smiling"
  unconditional: "man, not smiling"
  neutral: "man"
  action: "enhance"
  guidance_scale: 4
  resolution: 1024
  dynamic_resolution: false
  batch_size: 1
```
This slider should make only men smile. Simple prompt, no context — works in any scene.

### Prompt rules (CURRENT PHASE — simple sliders)
1. **Do NOT describe contexts, clothing, locations.** Prompts must be short and context-free. The slider must work wherever the subject appears.
2. The ONLY difference between `positive` and `unconditional` is the controlled attribute (smiling/not smiling, old/young).
3. `target` and `neutral` are identical — they represent the base subject.
4. The subject in the prompt is what makes the slider local: "man" instead of "person".

### How many prompts are needed? (answered by the paper — Eq. 8)

The paper (Eq. 8) and the code give a clear answer: **multiple YAML entries serve as preservation/disentanglement prompts**. Each entry provides the same concept direction (e.g., old→young) on a different subject variant (different race, gender, etc.). The LoRA learns the AVERAGE direction across all entries — the common part (the concept) is reinforced, while subject-specific biases cancel out.

**Two styles exist in the original repo:**

1. **Minimal (2 entries)**: `prompts-xl.yaml` uses just "male person" + "female person". This provides minimal disentanglement (only gender). Quick to train, but the learned direction may be slightly entangled with race or other attributes.

2. **Expanded (10 entries)**: `prompts-person_age_slider_GPT.yaml` uses 5 ethnicities × 2 genders = 10 entries. This provides full disentanglement across race and gender, as described in Eq. 8 and Figure 12 of the paper.

**For our project**, the question is different from the original: we are NOT trying to disentangle across race/gender (our slider is already anchored to a specific subject like "man"). Instead, we might want to disentangle across **appearance variants** of that subject to ensure the slider captures ONLY the concept (smile, age) and not a specific "type" of man. This is something to discuss and test experimentally.

**Starting strategy**: begin with a SINGLE entry (simplest case) and see if it works. If the slider shows unwanted entanglement, add more entries with subject variants. The `--attributes` flag can automate this — e.g., `--attributes "white, black, asian, hispanic, indian"` would multiply each entry by 5 variants.

### Future evolutions (NOT NOW)
After validating simple sliders, more specific prompts may be explored:
- `"man with black hair, smiling"` — discriminates between different men
- `"fish, cooked"` / `"fish, raw"` — instead of generic "food"
- Prompts with explicit context for complex scenes
- Multi-subject prompts with specific scenes

---

## 7. DIRECTION VISUALIZATION DETAILS

The `visualize_direction.py` script is fundamental for verifying the project hypothesis. It must:

1. Load the SDXL model
2. Generate an image with a test prompt (e.g., "a man and a woman at a cafe")
3. For each prompt pair (positive/unconditional) of the slider, compute:
   - `D(x,y) = |ε(x_t, positive) - ε(x_t, unconditional)|` for each spatial position
4. Visualize D as a heatmap overlaid on the image
5. Save visualizations for comparison

If our "smile man" slider works, the heatmap will be concentrated on the man. If the baseline "smile person" slider gives diffuse heatmaps over all people, we have confirmation of the hypothesis.

**Technical note**: the difference must be computed in latent space (resolution is ~128×128 for SDXL). To visualize it on the image, use bilinear upscaling. Use the functions already available in `train_util.py` (`encode_prompts_xl`, `predict_noise_xl`) to compute noise predictions.

---

## 8. DO NOT

- **Do NOT touch ANYTHING outside `training_local_concept_sliders/`.** Do not modify `trainscripts/`, `maskedLORA/`, `real_editing/`, `exp_generation/`, or any other file/folder in the repository. Your scope is EXCLUSIVELY `training_local_concept_sliders/`.
- **Do NOT make changes without providing terminal commands.** Every time you create or modify a file, you MUST also give the user git commands to push and HPC commands to execute.
- **Do NOT rewrite training from scratch.** Always start from Gandikota's original code. We use EXACTLY the same training.
- **Do NOT modify the loss function.** The only thing that changes is the prompt text.
- **Do NOT modify the 4-prompt mechanism.** Every entry MUST have target, positive, unconditional, neutral. All 4 are required and have specific roles in the training loop.
- **Do NOT create dependencies on files outside the folder** (except virtualenv and HF models).
- **Do NOT hardcode prompts in code.** They must be in YAML config files.
- **Do NOT create unnecessary files.** Every file must have a clear purpose.
- **Do NOT proceed without reading** the original scripts in `trainscripts/textsliders/` and existing .slurm files.
- **Do NOT change the prompt YAML format.** Use exactly the fields: target, positive, unconditional, neutral, action, guidance_scale, resolution, dynamic_resolution, batch_size.
- **Do NOT write code, comments, or documentation in Italian.** All work in the repository must be in ENGLISH. Only chat communication can be in Italian.
- **Do NOT forget to prepare SLURM files.** Jobs must be ready to use: the user does `sbatch` and that's it.
- **Do NOT rush.** Build incrementally, explain, ask, validate. One step at a time.
