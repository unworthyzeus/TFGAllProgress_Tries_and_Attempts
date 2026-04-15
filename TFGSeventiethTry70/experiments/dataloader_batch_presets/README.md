# DataLoader, batch y gradient accumulation (Try 70)

Experto principal: `experiments/seventieth_try70_experts/try70_expert_open_sparse_lowrise.yaml`.

| Bloque | Claves |
|--------|--------|
| `data:` | `num_workers`, `val_num_workers`, `persistent_workers`, `val_persistent_workers`, `prefetch_factor`, `val_batch_size` |
| `training:` | `batch_size`, `gradient_accumulation_steps` |

## Slurm en este try

| GPUs | Script | CPUs | Memoria |
|------|--------|------|---------|
| 1 | `cluster/run_seventieth_try70_1gpu.slurm` | 6 | 38G |
| 2 | *(no hay `run_seventieth_try70_2gpu.slurm`; usar preset 2gpu y un Slurm 2×GPU tomado de Try 68/69)* | — | — |
| 4 | `cluster/run_seventieth_try70_4gpu.slurm` | 32 | 120G |

## Presets (`presets/`)

| Archivo | Uso |
|---------|-----|
| `1gpu.yml` | Job 1 GPU |
| `2gpu.yml` | Job 2×GPU (valores alineados a nodo 2 GPU típico del proyecto) |
| `4gpu.yml` | Job 4×GPU |

Copia manualmente a `data:` / `training:` del YAML del experto.

## Baseline repo (1 GPU)

Alineado con `presets/1gpu.yml`.
