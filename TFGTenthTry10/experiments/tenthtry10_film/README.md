# TenthTry10 — variante FiLM (carpeta nueva)

| YAML | Split |
|------|--------|
| `tenthtry10_los_film.yaml` | LoS-dominant |
| `tenthtry10_nlos_film.yaml` | NLoS-dominant |

Salidas distintas del try 10 por **canal**; **no** mezclar checkpoints.

## Upload + ambas variantes FiLM 1 GPU (desde `TFGpractice/cluster/`)

Los scripts de upload convierten `*.slurm` a finales de línea Unix al subir (evita el error `DOS line breaks` de `sbatch` en Windows).

Una subida y dos `sbatch` (LoS + NLoS); limpia `outputs/` una vez (salvo `--no-clean-outputs`):

```bash
set SSH_PASSWORD=...
python upload_and_submit_tenthtry10.py --both-film-1gpu
```

Solo una variante con FiLM:

```bash
python upload_and_submit_tenthtry10.py --variant los --gpus 1 --film
python upload_and_submit_tenthtry10.py --variant nlos --gpus 1 --film --no-clean-outputs
```

## Slurm (desde `TFGTenthTry10/`)

```bash
sbatch cluster/run_tenthtry10_film_los_1gpu.slurm
sbatch cluster/run_tenthtry10_film_nlos_1gpu.slurm   # o usa --both-film-1gpu arriba
# 2 GPU:
sbatch cluster/run_tenthtry10_film_los_2gpu.slurm
sbatch cluster/run_tenthtry10_film_nlos_2gpu.slurm
```

Logs: `logs_train_tenthtry10_{los,nlos}_film_ddp_{1,2}gpu_<JOBID>.out`

Rutas `../Datasets/...` se anclan a la raíz del try (carpeta con `configs/`).
