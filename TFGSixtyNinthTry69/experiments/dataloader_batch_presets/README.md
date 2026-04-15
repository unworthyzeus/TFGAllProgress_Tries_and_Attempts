# DataLoader, batch y gradient accumulation (Try 69)

Expertos: `experiments/sixtyninth_try69_experts/try69_expert_*.yaml`.

| Bloque | Claves |
|--------|--------|
| `data:` | `num_workers`, `val_num_workers`, `persistent_workers`, `val_persistent_workers`, `prefetch_factor`, `val_batch_size` |
| `training:` | `batch_size`, `gradient_accumulation_steps` |

## Cómo aplicar un preset

Copia desde `presets/1gpu.yml`, `2gpu.yml` o `4gpu.yml` a los bloques `data:` y `training:` del experto. Alinea el preset con el **Slurm** que vayas a lanzar.

## Slurm en este try

| GPUs | Script | CPUs | Memoria |
|------|--------|------|---------|
| 1 | `cluster/run_sixtyninth_try69_1gpu.slurm` | 6 | 38G |
| 2 | `cluster/run_sixtyninth_try69_2gpu.slurm` | 16 | 58G |
| 4 | `cluster/run_sixtyninth_try69_4gpu.slurm` | 32 | 120G |

## Presets (`presets/`)

| Archivo | Encaje típico |
|---------|----------------|
| `1gpu.yml` | Solo 1 GPU por job |
| `2gpu.yml` | 2×GPU, más RAM que 1 GPU |
| `4gpu.yml` | 4×GPU, máximo host RAM / workers |

## Baseline repo (1 GPU)

Coincide con `presets/1gpu.yml`: workers 0, `gradient_accumulation_steps: 16`.
