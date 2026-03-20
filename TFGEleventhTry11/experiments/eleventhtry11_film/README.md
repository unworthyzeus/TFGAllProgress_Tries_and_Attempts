# EleventhTry11 — variante FiLM

- **Config:** `eleventhtry11_film.yaml`
- **Checkpoints:** bajo `outputs/...eleventhtry11_film_*` — incompatibles con el yaml principal (canal).

Desde el repo (`TFGpractice/cluster/`), con el venv que tengas:

```bash
# Subir try 11 y lanzar FiLM (2 GPU); si ya subiste otro job, añade --no-clean-outputs
python upload_and_submit_eleventhtry11.py --film --gpus 2
python upload_and_submit_eleventhtry11.py --film --gpus 1
```

En el cluster, desde `TFGEleventhTry11/`:

```bash
sbatch cluster/run_eleventhtry11_film_2gpu.slurm
# o
sbatch cluster/run_eleventhtry11_film_1gpu.slurm
```

Logs: `logs_train_eleventhtry11_film_ddp_{1,2}gpu_<JOBID>.out`
