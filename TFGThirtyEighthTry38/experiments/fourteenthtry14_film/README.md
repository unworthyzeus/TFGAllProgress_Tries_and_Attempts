# FourteenthTry14 — FiLM (configuración recomendada)

- **YAML:** `fourteenthtry14_los_film.yaml` · `fourteenthtry14_nlos_film.yaml`
- Igual filosofía que Try 13 FiLM: **180 dB**, **100 épocas** máx., FiLM **hidden 192**, early stopping, scheduler patience 4.
- **Entrenamiento previsto en cluster:** **2 GPUs** por variante (LoS y NLoS por separado).

## Enviar al cluster (2 GPUs cada job)

Desde la raíz **`TFGFourteenthTry14/`**:

```bash
cd /ruta/a/TFGFourteenthTry14

# LoS FiLM — 2 GPUs
sbatch cluster/run_fourteenthtry14_film_los_2gpu.slurm

# NLoS FiLM — 2 GPUs
sbatch cluster/run_fourteenthtry14_film_nlos_2gpu.slurm
```

Logs:

- `logs_train_fourteenthtry14_los_film_ddp_2gpu_<JOBID>.out`
- `logs_train_fourteenthtry14_nlos_film_ddp_2gpu_<JOBID>.out`

Sufijos por defecto: `ddp2_t14_los_film`, `ddp2_t14_nlos_film`.

### Variables opcionales

```bash
export HDF5_PATH=/ruta/a/CKM_Dataset_180326.h5
export SCALAR_CSV_PATH=/ruta/a/CKM_180326_antenna_height.csv
sbatch cluster/run_fourteenthtry14_film_los_2gpu.slurm
```

### Slurm 1 GPU (no usarlos por defecto)

Siguen en el repo por si los necesitas: `cluster/run_fourteenthtry14_film_los_1gpu.slurm` y `..._nlos_1gpu.slurm`.

---

## Subir desde Windows (primera vez / actualizar código)

Desde la raíz **`TFGpractice/`** (donde están `Datasets/` y `TFGFourteenthTry14/`):

1. Instala dependencia si hace falta: `pip install paramiko`
2. Define la contraseña SSH (no la guardes en el repo):

```powershell
$env:SSH_PASSWORD = "tu_contraseña"
```

3. **Solo subir** (sin lanzar jobs):

```powershell
cd C:\TFG\TFGpractice
python cluster\upload_and_submit_fourteenthtry14.py --upload-only
```

4. **Subir y lanzar LoS + NLoS con 2 GPU** (recomendado):

```powershell
python cluster\upload_and_submit_fourteenthtry14.py --both-film-2gpu
```

5. Si vas a subir **otra vez** mientras un job sigue escribiendo en `outputs/`, usa `--no-clean-outputs` para no borrar checkpoints:

```powershell
python cluster\upload_and_submit_fourteenthtry14.py --upload-only --no-clean-outputs
```

Los scripts convierten los `.slurm` a finales de línea Unix al subir (evita el error de `sbatch` con CRLF).

**Rutas en el script:** usuario `gmoreno`, host `sert.ac.upc.edu`, remoto `/scratch/nas/3/gmoreno/TFGpractice`. Ajusta `HOST` / `USER` / `REMOTE_BASE` en `cluster/upload_and_submit_fourteenthtry14.py` si tu cuenta difiere.
