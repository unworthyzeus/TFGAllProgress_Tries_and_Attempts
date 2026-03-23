# ThirteenthTry13 — FiLM, **un solo modelo**

- **Un checkpoint** entrenado con **todo el dataset** (no hay `los_only` / `nlos_only`).
- **YAML:** `thirteenthtry13_film.yaml`
- **Máscara:** `path_loss_saturation_db: 180`
- Hasta **100 épocas** con early stopping; FiLM `scalar_film_hidden: 192`.

> **Dos modelos** (uno dominante LoS y otro NLoS) → **`TFGFourteenthTry14`** (`fourteenthtry14_los_film.yaml` / `fourteenthtry14_nlos_film.yaml`).

## Cluster (1 GPU)

```bash
cd /ruta/a/TFGThirteenthTry13
sbatch cluster/run_thirteenthtry13_film_1gpu.slurm
```

Log: `logs_train_thirteenthtry13_film_ddp_1gpu_<JOBID>.out`  
Sufijo salida por defecto: `ddp1_t13_film`.

### Variables opcionales

```bash
export HDF5_PATH=/ruta/a/CKM_Dataset_180326.h5
export SCALAR_CSV_PATH=/ruta/a/CKM_180326_antenna_height.csv
export OUTPUT_SUFFIX=ddp1_t13_film_mi_run
sbatch cluster/run_thirteenthtry13_film_1gpu.slurm
```

---

## Subir desde Windows (`TFGpractice/`)

```powershell
pip install paramiko
$env:SSH_PASSWORD = "tu_contraseña"
cd C:\TFG\TFGpractice

python cluster\upload_and_submit_thirteenthtry13.py --upload-only   # solo archivos
python cluster\upload_and_submit_thirteenthtry13.py                 # sube + sbatch 1 job
```

`--no-clean-outputs` si re-subes sin borrar `outputs/` remotos.
