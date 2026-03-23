# Dataset: ¿todos los tries usan “10000 muestras” al completo?

**No.** En el repo **no** aparece un número fijo “10000”; depende del **HDF5** (`CKM_Dataset_180326.h5` u otro) y del **YAML**.

| Situación | Tries / configs |
|-----------|------------------|
| **Dataset pequeño/antiguo** | **Try 1–5** suelen apuntar a `CKM_Dataset_old_and_small.h5` (no es el grande 180326). |
| **Solo 100 muestras en entrenamiento** | **SixthTry6** (`subset_size: 100`), **SeventhTry7** y **EighthTry8** en sus YAML principales de path loss — *no* usan todo el catálogo. |
| **Train = todo el split HDF5** (sin `subset_size`, o `null`) | **NinthTry9** en adelante en los YAML típicos de path loss: se usa **todo el conjunto train** tras el split 70/15/15 (o el que marque `val_ratio` / `test_ratio`). El **número real de muestras** = lo que haya en el `.h5`, no “10000” salvo que tu fichero tenga exactamente eso. |
| **Subconjunto por tipo de escena** | **Try 10, 12, 14** (y similares con `los_only` / `nlos_only`): cada modelo entrena con **una parte** del dataset (muestras dominantes LoS o NLoS), no con las ~10000 a la vez en un solo checkpoint. |
| **Try 13 FiLM** | **Un modelo** con **todo** el train (sin filtro LoS/NLoS). |

Para saber el número exacto de muestras en tu máquina: inspecciona el HDF5 o mira el log al arrancar el entrenamiento (recuentos tras split/filtro).

---

# Comandos: subir y ejecutar (desde Windows)

Raíz: **`TFGpractice/`**. Instala una vez: `pip install paramiko`.

**Cuota scratch (SERT):** si `quota -s` muestra el uso de **scratch** por encima de la cuota (asterisco `*`), SFTP falla (`OSError: Failure`) al crear carpetas o subir `.h5`. Libera espacio (borra `outputs/` viejos, checkpoints duplicados, etc.) **antes** de subir. Un `.h5` a **0 bytes** en el servidor cuenta como “existe”: el script ahora lo detecta y lo vuelve a subir si el local tiene tamaño correcto.

```powershell
cd C:\TFG\TFGpractice
$env:SSH_PASSWORD = "tu_contraseña"
```

Ajusta usuario/host/ruta remota en cada script si no usas `gmoreno` @ `sert.ac.upc.edu` y `/scratch/nas/3/gmoreno/TFGpractice`.

**`--no-clean-outputs`** (donde exista): re-subir sin borrar `outputs/` remotos (útil con dos jobs en paralelo o reintentos).

---

## Try 6

```powershell
python cluster\upload_and_submit_sixthtry6.py
```

## Try 7

```powershell
python cluster\upload_and_submit_seventhtry7.py --gpus 2
# o
python cluster\upload_and_submit_seventhtry7.py --gpus 4
```

## Try 8

```powershell
python cluster\upload_and_submit_eighthtry8.py --gpus 2
python cluster\upload_and_submit_eighthtry8.py --gpus 4
```

## Try 9

```powershell
python cluster\upload_and_submit_ninthtry9.py --gpus 2
python cluster\upload_and_submit_ninthtry9.py --gpus 1
```

## Try 10 (incl. FiLM)

```powershell
# Un run: variante los o nlos, 1 o 2 GPU, --film opcional
python cluster\upload_and_submit_tenthtry10.py --variant los --gpus 2
python cluster\upload_and_submit_tenthtry10.py --variant nlos --gpus 2 --no-clean-outputs

# FiLM
python cluster\upload_and_submit_tenthtry10.py --variant los --gpus 1 --film
python cluster\upload_and_submit_tenthtry10.py --both-film-1gpu
```

## Try 11

```powershell
python cluster\upload_and_submit_eleventhtry11.py --gpus 2
python cluster\upload_and_submit_eleventhtry11.py --gpus 1 --film
```

## Try 12

```powershell
python cluster\upload_and_submit_twelfthtry12.py --variant los --gpus 2
python cluster\upload_and_submit_twelfthtry12.py --variant nlos --gpus 2 --no-clean-outputs
```

## Try 13 — **un modelo** FiLM (1 GPU)

```powershell
python cluster\upload_and_submit_thirteenthtry13.py --upload-only   # solo sincronizar
python cluster\upload_and_submit_thirteenthtry13.py                  # sube + sbatch 1 job
# opcional:
python cluster\upload_and_submit_thirteenthtry13.py --no-clean-outputs
```

**Dataset FiLM 13/14:** el script sube (si falta en el cluster) **`TFGpractice/Datasets/CKM_180326_antenna_height.h5`** (o `CKM_Dataset_180326_antenna_height.h5` local → remoto con nombre **`CKM_180326_antenna_height.h5`**). **No** usa `CKM_Dataset_180326.h5` ni CSV en estos uploads.

En el cluster (si ya subiste a mano):

```bash
cd .../TFGThirteenthTry13
sbatch cluster/run_thirteenthtry13_film_1gpu.slurm
```

## Try 14 — **dos modelos** FiLM LoS + NLoS (2 GPU recomendado)

```powershell
python cluster\upload_and_submit_fourteenthtry14.py --upload-only
python cluster\upload_and_submit_fourteenthtry14.py --both-film-2gpu
# opcional 1 GPU cada uno (scripts conservados):
python cluster\upload_and_submit_fourteenthtry14.py --both-film-1gpu
# un solo job:
python cluster\upload_and_submit_fourteenthtry14.py --variant los --gpus 2
```

En el cluster:

```bash
cd .../TFGFourteenthTry14
sbatch cluster/run_fourteenthtry14_film_los_2gpu.slurm
sbatch cluster/run_fourteenthtry14_film_nlos_2gpu.slurm
```

---

## Tries 1–5 / genérico

Si tienes `cluster/upload_and_submit.py` u otro legado, úsalo según su docstring. Los intentos **1–4** a menudo se lanzan en local o con Slurm antiguo; no todos tienen el mismo helper que 6+.

## Dispatcher consolidado

Si prefieres un único script para los tries actuales y los futuros experimentos de city-regime, usa:

```powershell
python cluster\upload_and_submit_experiments.py --preset ninth --gpus 1
python cluster\upload_and_submit_experiments.py --preset thirteenth --gpus 1
python cluster\upload_and_submit_experiments.py --local-dir TFGCityRegimeTry15 --slurm cluster/run_cityregime_try15_1gpu.slurm --gpus 1
```

La estrategia completa de los nuevos tries está en [NEXT_TRIES_CITY_REGIME.md](NEXT_TRIES_CITY_REGIME.md).

---

Más detalle FiLM 13/14:  
`TFGThirteenthTry13/experiments/thirteenthtry13_film/README.md` · `TFGFourteenthTry14/experiments/fourteenthtry14_film/README.md`
