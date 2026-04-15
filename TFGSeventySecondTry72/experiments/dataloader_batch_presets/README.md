# DataLoader, batch y gradient accumulation (Try 72)

Los expertos están en `experiments/seventysecond_try72_experts/try72_expert_*.yaml`.

| Bloque YAML | Claves |
|-------------|--------|
| `data:` | `num_workers`, `val_num_workers`, `persistent_workers`, `val_persistent_workers`, `prefetch_factor`, `val_batch_size` |
| `training:` | `batch_size`, `gradient_accumulation_steps` |

## Cómo aplicar un preset

1. Elige **1 GPU**, **2 GPU**, **3 GPU (`medium_gpu`)** o **4 GPU** según el script Slurm que uses (abajo). En SERT, **3 GPUs con `big_gpu`** suele quedar en cola con **`QOSMinGRES`**: ese QoS exige un **mínimo** de GPUs por job (muchas veces **4**), y un job de 3 no cumple el mínimo. Con **`medium_gpu`** y 3 GPUs no aplica ese mínimo. Si con `medium_gpu` ves **`MaxGRESPerAccount`**, tu **cuenta** Slurm ya tiene tantas GPUs reservadas/en uso (todos tus jobs sumados) que no cabe otro bloque de 3; cancela jobs o espera.
2. Abre el preset en `presets/1gpu.yml`, `2gpu.yml`, `3gpu_medium.yml` o `4gpu.yml` y copia las claves al bloque `data:` y `training:` del experto.
3. Con `num_workers: 0` debes tener `persistent_workers: false` y `val_persistent_workers: false` (requisito de PyTorch).

## Slurm que tenemos en este try (referencia)

| GPUs | Script | CPUs | Memoria |
|------|--------|------|---------|
| 1 | `cluster/run_seventysecond_try72_1gpu.slurm` | 6 | 38G |
| 2 | `cluster/run_seventysecond_try72_2gpu.slurm` | 24 | 78G |
| 3 | `cluster/run_seventysecond_try72_3gpu_medium_gpu.slurm` (`qos=medium_gpu`) | 24 | 120G |
| 4 | `cluster/run_seventysecond_try72_4gpu.slurm` | 32 | 120G |

`torchrun` multi-GPU: el `batch_size` del YAML suele ser **por proceso**; `gradient_accumulation_steps` se puede bajar al subir GPUs si quieres un paso de optimizador de coste similar (ajusta a tu prueba).

## Presets YAML (`presets/`)

| Archivo | Uso |
|---------|-----|
| `1gpu.yml` | Job 1 GPU, poca RAM host / varios jobs en el mismo nodo |
| `2gpu.yml` | Job 2×RTX2080, más CPUs y RAM para DataLoader |
| `3gpu_medium.yml` | Misma receta que `4gpu.yml` para jobs 3×RTX2080 + `medium_gpu` (ver nota arriba sobre `big_gpu`) |
| `4gpu.yml` | Job 4×RTX2080, máximo paralelismo de workers |

Subida + cancelar todos tus jobs + encolar cadena 3-GPU: desde `TFGpractice/`, `python cluster/relaunch_try72_3gpu_medium_upload.py --cancel-all-user-jobs` (requiere `SSH_PASSWORD` o `--ssh-key`).

## Baseline actual en repo (expertos Try 72)

Pensado para cadenas **1 GPU** en paralelo con otros tries: ver `presets/1gpu.yml` (workers 0, `grad_accum` 16).
