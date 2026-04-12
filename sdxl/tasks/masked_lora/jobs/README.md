# Slurm (HPC Bocconi-style)

Gli script usano `account=3226571`, `partition=stud`, `qos=stud`, `gpus=1` come `exp_generation/run_generation.sh`.

Dalla **root del repo** sul cluster:

```bash
cd /home/<your-username>/FERT_PROJECT/local-concept-sliders
mkdir -p sdxl/tasks/masked_lora/logs
sbatch sdxl/tasks/masked_lora/jobs/old_slurm/run_phase1.slurm
```

Dopo SAM in locale e upload di `mask.png` in `sdxl/tasks/masked_lora/outputs/fert_test_001/`:

```bash
sbatch sdxl/tasks/masked_lora/jobs/old_slurm/run_phase3.slurm
```

Log: `sdxl/tasks/masked_lora/logs/fert_p1_<jobid>.out` (e `.err`).

**Prova preimpostata:** `RUN_ID=fert_test_001`, fase 3 usa `sdxl/trained_sliders/sliders/age_sdxl.pt` (come il baseline generation). Se il file non c'è sul cluster, cambia `SLIDER_PATH` in `run_phase3.slurm`.
