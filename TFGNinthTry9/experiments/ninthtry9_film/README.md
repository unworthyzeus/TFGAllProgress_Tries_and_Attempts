# NinthTry9 — variante `ninthtry9_film`

- **YAML:** `ninthtry9_film.yaml` (nombre corto; FiLM + `output_dir` propio).
- **Checkpoints:** salen bajo `outputs/...ninthtry9_film/`; **no** son compatibles con el entrenamiento por **canal** del `configs/cgan_unet_*_ninthtry9.yaml` principal.

## Slurm (desde la raíz `TFGNinthTry9/` en el cluster)

```bash
sbatch cluster/run_ninthtry9_film_2gpu.slurm
# o 1 GPU:
sbatch cluster/run_ninthtry9_film_1gpu.slurm
```

## Logs

- 2 GPU: `logs_train_ninthtry9_film_ddp_2gpu_<JOBID>.out`
- 1 GPU: `logs_train_ninthtry9_film_ddp_1gpu_<JOBID>.out`

## Manual

```bash
export CONFIG_PATH=experiments/ninthtry9_film/ninthtry9_film.yaml
export OUTPUT_SUFFIX=ddp2_ninthtry9_film   # o ddp1_... para 1 GPU
sbatch cluster/run_ninthtry9_film_2gpu.slurm
```

Las rutas `../Datasets/...` del YAML se resuelven respecto a la **raíz del try** (carpeta que contiene `configs/`), gracias a `config_utils.anchor_data_paths_to_config_file`.
