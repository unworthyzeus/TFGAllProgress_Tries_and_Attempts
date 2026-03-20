# Export visual del dataset y predicciones

## Rutas de datos (antenna height)

Por defecto el script usa:

- **`Datasets/CKM_Dataset_180326_antenna_height.h5`** (incluye `uav_height` por muestra → **no hace falta CSV** para NinthTry9).
- Si usas el `.h5` sin `_antenna_height`, entonces suele hacer falta **`CKM_180326_antenna_height.csv`** (o `--scalar-csv`).

## Comandos guardados (referencia local `C:\TFG\TFGpractice`)

Variables comunes (ajusta `--dataset-out` y `--device` si quieres):

```powershell
$Root = "C:\TFG\TFGpractice"
$H5   = "$Root\Datasets\CKM_Dataset_180326_antenna_height.h5"
$CSV  = "$Root\Datasets\CKM_180326_antenna_height.csv"
$Out  = "D:\Dataset_Imagenes"
$Split = "test"   # o "all" si quieres todas las muestras del reparto
cd $Root
```

### 1) Export: **NinthTry9** (un solo modelo path loss)

Checkpoint único ya usado en el repo (`ninthtry9_ddp2_ninthtry9`):

```powershell
python scripts/export_dataset_and_predictions.py `
  --skip-dataset-export `
  --device directml `
  --hdf5 $H5 `
  --scalar-csv $CSV `
  --dataset-out $Out `
  --split $Split `
  --ninth-root "$Root\TFGNinthTry9" `
  --ninth-config "$Root\TFGNinthTry9\configs\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml" `
  --ninth-checkpoint "$Root\cluster_outputs\TFGNinthTry9\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9_ddp2_ninthtry9\best_cgan.pt" `
  --spread-try second `
  --spread-config "$Root\TFGSecondTry2\configs\cgan_unet_hdf5_amd_midvram.yaml" `
  --spread-checkpoint "$Root\TFGSecondTry2\outputs\cgan_unet_hdf5_amd_midvram\best_cgan.pt"
```

### 2) Export: **TenthTry10** LoS + NLoS (FiLM, dos checkpoints)

LoS: `t10_los_film` · NLoS: `t10_nlos_film` (mismas carpetas bajo `cluster_outputs\TFGTenthTry10\...`).

```powershell
python scripts/export_dataset_and_predictions.py `
  --skip-dataset-export `
  --device directml `
  --hdf5 $H5 `
  --scalar-csv $CSV `
  --dataset-out $Out `
  --split $Split `
  --path-loss-try-root "$Root\TFGTenthTry10" `
  --ninth-config "$Root\TFGTenthTry10\experiments\tenthtry10_film\tenthtry10_los_film.yaml" `
  --ninth-config-los "$Root\TFGTenthTry10\experiments\tenthtry10_film\tenthtry10_los_film.yaml" `
  --ninth-config-nlos "$Root\TFGTenthTry10\experiments\tenthtry10_film\tenthtry10_nlos_film.yaml" `
  --ninth-checkpoint "$Root\cluster_outputs\TFGTenthTry10\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_tenthtry10_los_film_ddp1_t10_los_film\best_cgan.pt" `
  --ninth-checkpoint-nlos "$Root\cluster_outputs\TFGTenthTry10\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_tenthtry10_nlos_film_ddp1_t10_nlos_film\best_cgan.pt" `
  --spread-try second `
  --spread-config "$Root\TFGSecondTry2\configs\cgan_unet_hdf5_amd_midvram.yaml" `
  --spread-checkpoint "$Root\TFGSecondTry2\outputs\cgan_unet_hdf5_amd_midvram\best_cgan.pt"
```

Salida path loss: `$Out\predictions_ninthtry9_path_loss\<split>\by_field\path_loss\...` (nombre histórico). En modo dual también `*_gt_nlos_los_combined.png`.

### 3) Panel 2×4: `build_alltogether_panel.py`

**Mismo `--split`** que en el export (`test`, `val`, `train` o `all`). **`--spread-label`** por defecto es **`auto`** (elige una carpeta `predictions_*_delay_angular` que exista para ese split). Si ves “Missing image” en predicciones, suele ser **`--split` distinto al export** o no haber corrido el export con **`--spread-checkpoint`**. Usa **`--verbose-paths`** en la primera muestra para imprimir rutas exactas.

El nombre del PNG `*_2x4_h56p6469m.png`: **`2x4`** = rejilla 2 filas × 4 columnas (no tamaño en píxeles del mapa); **`h56p6469m`** ≈ altura UAV **56,6469 m** leída del HDF5 (`uav_height`), no un mapa de altura dibujado.

```powershell
python scripts/build_alltogether_panel.py `
  --data-root $Out `
  --split $Split `
  --hdf5 $H5
```

Salida: `$Out\alltogether\<ciudad>\<muestra>_2x4_*.png`.

---

### Un comando ya listo (tus rutas)

En PowerShell (el `.ps1` intenta activar `TFGpractice\.venv` antes de llamar a Python):

```powershell
C:\TFG\TFGpractice\scripts\run_export_local_assets.ps1
```

O manualmente: activa el venv y usa el mismo comando `python scripts/export_dataset_and_predictions.py ...` que lleva el `.ps1`.

## Modelos

| Rol | Try | Config por defecto | Métricas en `cluster_outputs` |
|-----|-----|-------------------|-------------------------------|
| **Path loss** | **NinthTry9** | `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml` | `TFGNinthTry9/.../validate_metrics_cgan_best.json` → época **15**, `rmse_physical` path loss ~24 dB (en los JSON que tienes sincronizados). |
| **Delay + angular spread** | **auto** entre First / Second / Third | Si `--spread-try auto`, escanea `validate_metrics*.json` y elige el try con **menor** `delay_spread.rmse_physical + angular_spread.rmse_physical`. Si no hay ningún JSON con esas claves, usa **Third** (`cgan_unet_hdf5_cuda_max.yaml`). |

> En el repo actual **no** hay JSONs con `delay_spread`/`angular_spread` bajo `cluster_outputs` (solo path_loss). En ese caso `auto` hace fallback a **third**. Cuando sincronices evaluaciones multi-target, el escaneo las usará.

### Path loss: ¿Ninth (un modelo) o Tenth (LoS + NLoS)?

- **Un solo checkpoint** que entrena con **todo** el dataset (sin partir en `los_only` / `nlos_only`): suele ser **NinthTry9** → `--ninth-root` / **`--path-loss-try-root`** apuntando a `TFGNinthTry9`, un `--ninth-checkpoint`, **sin** `--ninth-checkpoint-nlos`.
- **Dos checkpoints** (uno entrenado solo en muestras LoS-dominantes y otro en NLoS): suele ser **TenthTry10** → mismo script, pero `--path-loss-try-root` (o `--ninth-root`) = `TFGTenthTry10`, `--ninth-checkpoint` = LoS, `--ninth-checkpoint-nlos` = NLoS, y YAMLs `-los` / `-nlos` si difieren.

Los flags siguen llamándose `--ninth-*` por histórico; **`--path-loss-try-root`** es alias de `--ninth-root`. La carpeta de salida sigue siendo `predictions_ninthtry9_path_loss/` aunque uses TenthTry10 (solo es el nombre de la carpeta).

### Ver pistas sin ejecutar inferencia

```powershell
cd C:\TFG\TFGpractice
python scripts/export_dataset_and_predictions.py --print-metric-hints
```

## Predecir **todo** el path loss (train + val + test)

Con HDF5, por defecto solo se infiere un split (`--split test`). Para generar PNG GT|Pred de **todas** las muestras del reparto (misma partición y filtros LoS que el entrenamiento, sin augmentación):

```powershell
python scripts/export_dataset_and_predictions.py ... --split all --ninth-checkpoint ...
```

Salida: `.../predictions_ninthtry9_path_loss/all/by_field/path_loss/<ciudad>/...`. Lo mismo aplica al modelo de delay/angular si pasas `--spread-checkpoint` (carpeta `predictions_<try>_delay_angular/all/`).

## Solo predicciones (dataset PNG ya exportado)

No volver a generar `raw_hdf5` desde el HDF5 (ahorra mucho tiempo):

```powershell
cd C:\TFG\TFGpractice
.\.venv\Scripts\Activate.ps1   # o tu venv
python scripts/export_dataset_and_predictions.py --skip-dataset-export `
  --hdf5 "C:\TFG\TFGpractice\Datasets\CKM_Dataset_180326_antenna_height.h5" `
  --dataset-out "D:\Dataset_Imagenes" --device cuda `
  --ninth-checkpoint "...\best_cgan.pt" `
  --spread-try second --spread-config "...\cgan_unet_hdf5_amd_midvram.yaml" --spread-checkpoint "...\best_cgan.pt"
```

O ejecuta **`scripts\run_export_predictions_only.ps1`** (mismas rutas que el export completo).

Sigue haciendo falta **`--hdf5`** porque el DataLoader lee el HDF5 para las mismas muestras/split; solo se omite escribir otra vez miles de PNG en `raw_hdf5/`.

### Comando completo (Ninth path loss + Second spread, Slurm parity)

Usa **`--scalar-csv`** con el mismo CSV que el cluster para que `scalar_feature_norms` coincidan con el entrenamiento (junto con `*_antenna_height.h5` sigue leyendo `uav_height` por muestra).

**CPU vs DirectML:** con las normas bien puestas, path loss en **DirectML** y **CPU** dan resultados casi iguales (como en `inspect_cgan_pathloss_checkpoint.py`). Usa **`--device directml`** en AMD para ir más rápido; **`--device cpu`** si algún día ves diferencias raras. **`--device auto`** elige CUDA → si no hay NVIDIA, DirectML si tienes `torch-directml`, si no CPU.

```powershell
cd C:\TFG\TFGpractice
python scripts/export_dataset_and_predictions.py `
  --skip-dataset-export `
  --device directml `
  --hdf5 "C:\TFG\TFGpractice\Datasets\CKM_Dataset_180326_antenna_height.h5" `
  --scalar-csv "C:\TFG\TFGpractice\Datasets\CKM_180326_antenna_height.csv" `
  --dataset-out "D:\Dataset_Imagenes" `
  --split all `
  --ninth-root "C:\TFG\TFGpractice\TFGNinthTry9" `
  --ninth-config "C:\TFG\TFGpractice\TFGNinthTry9\configs\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml" `
  --ninth-checkpoint "C:\TFG\TFGpractice\cluster_outputs\TFGNinthTry9\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9_ddp2_ninthtry9\best_cgan.pt" `
  --spread-try second `
  --spread-config "C:\TFG\TFGpractice\TFGSecondTry2\configs\cgan_unet_hdf5_amd_midvram.yaml" `
  --spread-checkpoint "C:\TFG\TFGpractice\TFGSecondTry2\outputs\cgan_unet_hdf5_amd_midvram\best_cgan.pt"
```

### Path loss: modelos separados LoS y NLoS (p. ej. TenthTry10 FiLM)

Si entrenaste dos checkpoints (`los_only` vs `nlos_only`), pasa el de **LoS** en `--ninth-checkpoint` y el de **NLoS** en `--ninth-checkpoint-nlos`, y los YAML que usaste en Slurm en `--ninth-config-los` / `--ninth-config-nlos` (si omites estos dos, se usa `--ninth-config` para ambos — solo si la arquitectura y `target_metadata` coinciden).

- El script **quita `los_sample_filter`** del dataset de inferencia e itera **todo** el split (train/val/test/`all`).
- Por cada muestra, clasifica con `los_mask` y el mismo criterio que el entrenamiento (`los_classify_field`, `los_classify_threshold`).
- PNG **`*_gt_pred.png`**: GT | predicción **combinada** (ruta LoS o NLoS según la muestra); el título del panel predicho indica `[LoS model]` o `[NLoS model]`.
- PNG adicional **`*_gt_nlos_los_combined.png`**: **cuatro** columnas en el orden **GT | NLoS | LoS | combinada** (misma escala dB que `--path-loss-viz-scale`: `gt` / `joint` / `independent`).

`--ninth-root` / `--path-loss-try-root` debe ser la carpeta del try que aporta `data_utils` (p. ej. `TFGTenthTry10` para FiLM).

```powershell
python scripts/export_dataset_and_predictions.py `
  --skip-dataset-export `
  --device directml `
  --hdf5 "C:\TFG\TFGpractice\Datasets\CKM_Dataset_180326_antenna_height.h5" `
  --scalar-csv "C:\TFG\TFGpractice\Datasets\CKM_180326_antenna_height.csv" `
  --dataset-out "D:\Dataset_Imagenes" `
  --split all `
  --ninth-root "C:\TFG\TFGpractice\TFGTenthTry10" `
  --ninth-config "C:\TFG\TFGpractice\TFGTenthTry10\experiments\tenthtry10_film\tenthtry10_los_film.yaml" `
  --ninth-config-los "C:\TFG\TFGpractice\TFGTenthTry10\experiments\tenthtry10_film\tenthtry10_los_film.yaml" `
  --ninth-config-nlos "C:\TFG\TFGpractice\TFGTenthTry10\experiments\tenthtry10_film\tenthtry10_nlos_film.yaml" `
  --ninth-checkpoint "C:\TFG\TFGpractice\cluster_outputs\TFGTenthTry10\...\t10_los_film\best_cgan.pt" `
  --ninth-checkpoint-nlos "C:\TFG\TFGpractice\cluster_outputs\TFGTenthTry10\...\t10_nlos_film\best_cgan.pt" `
  --spread-try second `
  --spread-config "C:\TFG\TFGpractice\TFGSecondTry2\configs\cgan_unet_hdf5_amd_midvram.yaml" `
  --spread-checkpoint "C:\TFG\TFGpractice\TFGSecondTry2\outputs\cgan_unet_hdf5_amd_midvram\best_cgan.pt"
```

Rutas literales TenthTry10 (t10_los_film / t10_nlos_film) y el bloque del panel: **§ Comandos guardados** al inicio de este README.

### Panel 2×4 (8 imágenes a la vez): `build_alltogether_panel.py`

Tras el export, junta por muestra: fila 1 (topology, LoS, path loss GT, delay GT) y fila 2 (angular GT + **path loss pred** = mitad derecha de `*_gt_pred.png`, delay pred, angular pred).

```powershell
cd C:\TFG\TFGpractice
python scripts/build_alltogether_panel.py `
  --data-root "D:\Dataset_Imagenes" `
  --split test `
  --hdf5 "C:\TFG\TFGpractice\Datasets\CKM_Dataset_180326_antenna_height.h5"
```

(`--split` debe coincidir con el export. `--spread-label auto` por defecto; `--verbose-paths` para depurar rutas.)

Los `*_gt_pred.png` llevan una **franja de texto** encima del mapa; el panel **recorta la mitad derecha y quita esos 26 px** para que path loss / delay / angular pred tengan el **mismo tamaño** que los GT de `raw_hdf5`.

**Altura UAV:** el CSV **`Datasets/CKM_180326_antenna_height.csv`** tiene columnas `city,sample,antenna_height_m` (como Nagpur / sample_09802 / 183.716…). Por defecto el panel **usa esos valores y pisan** el `uav_height` del HDF5 si existe la misma pareja ciudad/muestra (misma lógica que el cluster). Usa **`--prefer-hdf5-uav-height`** si quieres que el CSV solo rellene muestras que no estén en el HDF5.

Opcional: `--limit 20`, `--cell-w 384 --cell-h 384`. Salida: `Dataset_Imagenes/alltogether/<ciudad>/<muestra>_2x4_<altura>.png`.  
Mismo comando en **`scripts/run_build_alltogether_panel.ps1`** (ajusta `--data-root` y rutas).

## Path loss: `[0,1]` training targets vs dB on disk

The model does **not** output dB directly. Targets are normalized in the dataloader; export converts back to dB using `target_metadata.path_loss` from your YAML:

| `predict_linear` | Training target | Denormalize to dB |
|------------------|-----------------|-------------------|
| **false** (e.g. `…ninthtry9.yaml` tinygan lowresdisc) | `(dB - offset) / scale` (often `dB / 180`) | `pred * scale + offset` |
| **true** (many `path_loss_hybrid` / `path_loss_only` YAMLs) | log-linear encoding in `[0, 1]` | `path_loss_linear_normalized_to_db(pred)` |

**GT** in the side-by-side PNG uses the **same** YAML as the dataloader, so it can look correct even when **Pred** is wrong: if `--ninth-config` does not match the checkpoint’s training YAML (especially `predict_linear`), denormalization of the network output will be wrong. Pass the exact config file from that run (e.g. the one referenced in the Slurm job).

### Path loss preds in thousands of dB (scalar height norm)

If **CPU and DirectML** both show raw preds >>1 and dB in the thousands while GT is ~0–150 dB, weights usually load fine — the **antenna height plane** was wrong: with `scalar_table_csv` cleared (HDF5-only heights), norms defaulted to **1.0**, so the model saw **raw metres (~10–150)** instead of **÷ max like training with CSV**.

**Fix (automatic):** `build_dataset_splits_from_config` in NinthTry10/TenthTry9 `data_utils.py` now **infers `scalar_feature_norms` from HDF5 maxima** when there is no CSV and `hdf5_scalar_specs` is set (one-time log: `[INFO] scalar_feature_norms from HDF5 maxima`).

**Or** pass the same CSV as the cluster: `--scalar-csv Datasets\CKM_180326_antenna_height.csv`, or set `scalar_feature_norms.antenna_height_m` in YAML (e.g. `120.0`).

Sanity script (weights + forward):

```powershell
cd C:\TFG\TFGpractice
python scripts/inspect_cgan_pathloss_checkpoint.py `
  --config "TFGNinthTry9\configs\..." `
  --checkpoint "cluster_outputs\TFGNinthTry9\...\best_cgan.pt" `
  --hdf5 "Datasets\CKM_Dataset_180326_antenna_height.h5" `
  --split train --sample-index 0 `
  --compare-devices cpu,directml
```

If raw outputs are still bad only on DirectML, use **`--device cpu`** for export.

### Pred looks like “TV static” but GT is fine

Often this is **not** missing denorm: each PNG used **per-image min–max**. If predictions are almost **flat** (e.g. DirectML numerics or a weak constant output), tiny noise fills 0–255 and looks like static. The export script now defaults to **`--path-loss-viz-scale gt`**: both panels share the **GT’s** dB min/max so you see a fair comparison (flat pred → uniform gray). Use `--path-loss-viz-scale joint` to include pred in the range, or `independent` for the old behavior.

The first sample also prints **`sample 0 denorm dB`** (pred vs gt min/max/mean). If pred min≈max, the model/output device is suspect — try **`--device cpu`** to rule out DirectML.

## AMD GPU (e.g. RX 7800 XT) on Windows

PyTorch’s default pip wheel is often **CPU-only** (no NVIDIA CUDA). Options:

1. **`--device auto`** (default): uses **CPU** unless you install **`torch-directml`**, then it uses your AMD GPU via DirectML.
   ```powershell
   pip install torch-directml
   ```
2. **`--device directml`** after installing `torch-directml`.
3. **`--device cpu`** to force CPU (slow but always works).

Do **not** pass `--device cuda` without an NVIDIA stack; the script now falls back to CPU with a message.

## Venv (PyTorch)

Inference needs **`torch`** in the active Python. Either:

- Run **`.\scripts\run_export_local_assets.ps1`** (activates `TFGpractice\.venv` if `Scripts\Activate.ps1` exists), or
- Manually: `.\.venv\Scripts\Activate.ps1` then `python scripts\...`, or
- Set **`TFG_VENV_ROOT`** to another venv folder before the `.ps1` scripts.

If `torch` is missing, `export_dataset_and_predictions.py` exits with a short hint instead of a raw `ModuleNotFoundError`.

## Comando típico

```powershell
cd C:\TFG\TFGpractice
.\.venv\Scripts\Activate.ps1
python scripts/export_dataset_and_predictions.py `
  --dataset-out D:\Dataset_Imagenes `
  --ninth-checkpoint D:\ckpt\ninth\best_cgan.pt `
  --spread-checkpoint D:\ckpt\third\best_cgan.pt `
  --spread-try auto
```

- **Dataset** (`raw_hdf5/`):
  - `by_sample/<ciudad>/<muestra>/*.png` — todas las modalidades juntas por muestra.
  - `by_field/<tipo>/<ciudad>/<muestra>.png` — una carpeta por tipo (`topology_map`, `path_loss`, `delay_spread`, `angular_spread`, **`los`** para `los_mask`).
- **2×4 panel** (separate script):  
  `python scripts/build_alltogether_panel.py --data-root D:/Dataset_Imagenes`  
  → writes `alltogether/<city>/<sample>_2x4_<height>.png` (height from `uav_height` in HDF5, e.g. `h120p5m` = 120.5 m). Use `--hdf5` if needed. On-image labels are in **English**.

- **Predicciones** (misma idea `by_field/…`):
  - Ninth path loss: `predictions_ninthtry9_path_loss/<split>/by_field/path_loss/<ciudad>/<muestra>_gt_pred.png`
  - Spread: `predictions_*_delay_angular/<split>/by_field/<delay_spread|angular_spread|path_loss>/...`
- Por defecto **no** se guarda `path_loss` del modelo de 3 canales; añade `--spread-include-path-loss` si lo quieres.

Opciones: `--limit 50`, `--split val`, `--spread-try first|second|third`, `--skip-dataset-export`.
