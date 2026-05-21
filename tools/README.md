# `tools/`

Optional helper scripts that sync code and artefacts between a local
workstation and an HPC node. Not required to run any experiment — the
repository works fine if everything lives on the same machine. These
utilities exist because the development split for this project was
"edit on the laptop, train on the cluster", and rsync is faster than
git for transferring large output artefacts.

## Files

| File | Purpose | Tracked? |
|---|---|---|
| `set_slurms.sh.example` | Per-user HPC config template (repo path, HF cache, env activation functions) | yes (template) |
| `set_slurms.sh` | Active per-user HPC config; loaded by every SLURM template via `source` | **git-ignored** |
| `pull_config.sh.example` | Per-user local config template (HPC user / host / paths) | yes (template) |
| `pull_config.sh` | Active per-user local config; loaded by the sync scripts below | **git-ignored** |
| `push_to_hpc.sh` | local -> HPC: pushes `.slurm` + `configs/*.yaml` + `prompts/*/*.yaml` (incremental) | yes |
| `pull_from_hpc.sh` | HPC -> local: downloads trained sliders + image outputs (incremental) | yes |

## First-time setup

On HPC (for the SLURM jobs):

```bash
cd /path/to/local-concept-sliders
cp tools/set_slurms.sh.example tools/set_slurms.sh
nano tools/set_slurms.sh
# Set: FERT_REPO, FERT_HF_CACHE, activate_flux_env(), activate_sdxl_env()
```

On the local machine (for the sync scripts):

```bash
cd /path/to/local-concept-sliders
cp tools/pull_config.sh.example tools/pull_config.sh
nano tools/pull_config.sh
# Set: HPC_USER, HPC_HOST, HPC_REPO, LOCAL_REPO
```

The `set_slurms.sh.example` template ships with five environment-manager
recipes (venv / conda / mamba / mixed / Lmod module load); pick one and
adapt it to your cluster.

### SSH alias (recommended on the local machine)

To avoid typing `ssh user@hostname` every time, add to `~/.ssh/config`:

```
Host hpc
    HostName <your-hpc-hostname>
    User <your-username>
```

After that, `ssh hpc`, `scp file hpc:...` and the scripts in this
directory all work with `HPC_HOST="hpc"`.

Set up an SSH key once so you do not have to type the password every
time:

```bash
ssh-keygen -t ed25519
ssh-copy-id <your-username>@<your-hpc-hostname>
ssh hpc                              # should log in without password
```

## Usage

### local -> HPC (`push_to_hpc.sh`)

Pushes the project configuration files (SLURM templates + training
YAMLs) from the local machine to HPC. **Incremental**: only new or
modified files are transferred (size + mtime comparison via rsync).

```bash
./tools/push_to_hpc.sh           # default: everything (slurm + configs + prompts)
./tools/push_to_hpc.sh slurm     # only .slurm (any jobs/new_slurm/)
./tools/push_to_hpc.sh configs   # only configs/*.yaml
./tools/push_to_hpc.sh prompts   # only prompts/*/*.yaml
./tools/push_to_hpc.sh yaml      # configs + prompts (no slurm)
```

### HPC -> local (`pull_from_hpc.sh`)

Downloads trained slider weights and image outputs from HPC to the
local machine. **Incremental**.

```bash
./tools/pull_from_hpc.sh                  # default: download, keep on HPC
DRY_RUN=1 ./tools/pull_from_hpc.sh        # preview only (no transfers)
REMOVE_REMOTE=1 ./tools/pull_from_hpc.sh  # download AND free HPC
```

## What "incremental" means

Under the hood both scripts use `rsync` to compare size + mtime.
Identical files are skipped silently; only new or modified ones are
transferred. The output convention is rsync's standard:

- `>f+++++++++ x.slurm` -> new file (created on HPC)
- `>f.st...... x.slurm` -> modified file (different size or timestamp)
- (no line) -> identical, skipped

## Typical workflow

```bash
# 1. Edit / create new .slurm or training YAMLs on the local machine.
# 2. Push them to HPC.
./tools/push_to_hpc.sh new

# 3. Submit the job (on HPC).
ssh hpc
cd /path/to/local-concept-sliders
sbatch flux/tasks/<task>/jobs/new_slurm/myjob.slurm
exit

# 4. When the jobs finish, download everything back.
./tools/pull_from_hpc.sh
```

## Local <-> HPC path structure

Inside the repository the relative paths are identical on both sides —
only the absolute prefix differs:

```
local:  ~/Desktop/local-concept-sliders/                  flux/tasks/baseline/outputs/myrun/img.png
HPC:    /home/<user>/path/to/local-concept-sliders/       flux/tasks/baseline/outputs/myrun/img.png
                       ^ only difference (LOCAL_REPO vs HPC_REPO)
```

The two prefixes are configured in the per-user files:

- `LOCAL_REPO` in `tools/pull_config.sh` (local)
- `FERT_REPO`  in `tools/set_slurms.sh` (HPC)

The scripts build the full absolute paths dynamically from those
variables.
