# Versiones de experimentos (Try 1 → Try 14)

Este documento resume las carpetas **`TFG_FirstTry1`** … **`TFGFourteenthTry14`**, qué problema abordan y en qué difieren respecto a la versión anterior. Las rutas de código son copias evolutivas: cada try suele incluir `train_cgan.py`, `data_utils.py`, `configs/`, y a partir de cierto punto `cluster/` con Slurm.

> **Convención:** salvo **Try 13 y 14**, la mayoría de YAML usan `path_loss_saturation_db: **175**` para marcar píxeles saturados / no válidos en path loss. **Try 13 y 14** usan **`180`** (alineado con `target_metadata.path_loss.clip_max` y escala dB).

---

## Tabla rápida

| Try | Carpeta | Dataset principal | Objetivo(s) predichos | Notas clave |
|-----|---------|-------------------|------------------------|-------------|
| 1 | `TFG_FirstTry1` | `CKM_Dataset_old_and_small.h5` | path_loss + delay_spread + angular_spread (`out_channels: 3`) | Setup inicial HDF5, sin altura UAV, optimizadores RMSprop |
| 2 | `TFGSecondTry2` | `old_and_small` | 3 canales | Más capacidad (48ch), batch 2, `path_loss_saturation_db: 175` |
| 3 | `TFGThirdTry3` | `old_and_small` | 3 canales | Mayor UNet (96ch), `auto_batch_by_vram`, CUDA |
| 4 | `TFGFourthTry4` | `old_and_small` | **Solo path_loss + cabeza híbrida** (`out_channels: 2`) | `path_loss_hybrid.enabled`, tiny-GAN max112, batchnorm, discriminador baja resolución |
| 5 | `TFGFifthTry5` | `old_and_small` | Igual que 4 (2 salidas + hybrid) | LRs más bajos, `ReduceLROnPlateau` (patience 5); comentario en YAML orientado a MSE en dB |
| 6 | `TFGSixthTry6` | **`CKM_Dataset_180326.h5`** | Igual que 5 | Mismo esquema que 5 pero **dataset grande**; `subset_size: 100` (entrenamiento parcial) |
| 7 | `TFGSeventhTry7` | `180326` | **Solo path_loss** (`out_channels: 1`) | **Sin cabeza híbrida** (`lambda_gan: 0`); YAML principal con **`subset_size: 100`** (no todo el HDF5) |
| 8 | `TFGEighthTry8` | `180326` | Solo path_loss | **GAN activo**; **`subset_size: 100`** en YAML principal (como 7) |
| 9 | `TFGNinthTry9` | `180326` | Solo path_loss | **Altura antena**: `use_scalar_channels` + CSV/HDF5 `uav_height`; `path_loss_ignore_nonfinite`; variante **FiLM** opcional en `experiments/ninthtry9_film/` |
| 10 | `TFGTenthTry10` | `180326` | Solo path_loss | **Split LoS / NLoS** (`los_only` / `nlos_only`); postproceso simplificado vs 9; **FiLM LoS+NLoS** en `experiments/tenthtry10_film/` |
| 11 | `TFGEleventhTry11` | `180326` | Solo path_loss | **Normalización fija** de altura: `scalar_feature_norms.antenna_height_m: 120`; postpro mínimo; FiLM opcional `experiments/eleventhtry11_film/` |
| 12 | `TFGTwelfthTry12` | `180326` | Solo path_loss | Como 11 + **early stopping**; ramas **LoS / NLoS** en configs dedicados; **`use_scalar_film: true`** en YAML principales LoS/NLoS (FiLM en bottleneck) |
| 13 | `TFGThirteenthTry13` | `180326` | Solo path_loss | **Un modelo FiLM** (dataset completo, sin filtrar LoS/NLoS): `experiments/thirteenthtry13_film/thirteenthtry13_film.yaml`; **180 dB**; 1 GPU; predict PNG en dB |
| 14 | `TFGFourteenthTry14` | `180326` | Solo path_loss | **Dos modelos FiLM** (misma arquitectura/receta que 13, pero **`los_only`** y **`nlos_only`**): `fourteenthtry14_{los,nlos}_film.yaml`; Slurm **2 GPU** por job (1 GPU opcional en repo) |

---

## Detalle por versión

### Try 1 — `TFG_FirstTry1`
- **Enfoque:** CGAN UNet HDF5 multitarifa: path loss, delay spread y angular spread.
- **Datos:** dataset pequeño/antiguo; sin canal de saturación explícito en el fragmento típico del YAML principal.
- **Modelo:** canales base bajos (~24), sin escalares UAV.

### Try 2 — `TFGSecondTry2`
- **Cambio:** Mayor ancho de red, batch mayor, introduce `path_loss_saturation_db: 175` y `distance_map_channel` opcional en configs posteriores del try.
- **Uso típico hoy:** checkpoints **delay/angular** para export/visualización (p. ej. en `README_EXPORT_VISUALS.md`).

### Try 3 — `TFGThirdTry3`
- **Cambio:** Escala a ~96 canales base, entrenamiento CUDA con ajuste automático de batch por VRAM.

### Try 4 — `TFGFourthTry4`
- **Cambio:** Enfoque **path loss híbrido**: salida path loss + confianza (`out_channels: 2`, `path_loss_hybrid.enabled`).
- **Arquitectura:** `base_channels: 112`, discriminador pequeño, batch norm, augmentations estándar.
- **Postpro:** medianas, corrección LoS mezclada, mapas derivados de presupuesto de enlace.

### Try 5 — `TFGFifthTry5`
- **Cambio:** Misma familia híbrida que 4; **learning rates más conservadores** y scheduler con **patience 5** (vs 3 en runs posteriores).
- **YAML:** sigue llevando cabeza híbrida activa; el comentario del config describe el objetivo de entrenamiento en dB.

### Try 6 — `TFGSixthTry6`
- **Cambio:** **Mismo diseño que 5** pero entrenando sobre **`CKM_Dataset_180326.h5`**; `subset_size: 100` para iteración rápida en cluster.

### Try 7 — `TFGSeventhTry7`
- **Cambio:** **Una sola salida** path loss; **`lambda_gan: 0`** (sin adversario); sin altura UAV en el config principal.
- **Híbrido:** desactivado en la línea de “solo regresión”.

### Try 8 — `TFGEighthTry8`
- **Cambio:** Misma regresión única que 7 pero **reactiva GAN** (`lambda_gan: 0.01`) y **`lambda_recon: 20`** (recon menos dominante que en 7/9).

### Try 9 — `TFGNinthTry9`
- **Cambio:** **Condicionamiento por altura de antena** (`antenna_height_m`) como canal(es) extra; `use_scalar_film: false` en el YAML principal (plano espacial).
- **Datos:** `path_loss_ignore_nonfinite: true`; saturación **175**.
- **Extra:** `experiments/ninthtry9_film/` — FiLM (arquitectura incompatible con ckpt del YAML por canal).

### Try 10 — `TFGTenthTry10`
- **Cambio:** **Filtrado de muestras** `los_only` / `nlos_only` en configs dedicados; altura como en 9.
- **Postpro:** kernels medianos en 1, sin calibración heurística guardada (según YAML LoS principal).
- **Extra:** `experiments/tenthtry10_film/` — dos entrenamientos FiLM (LoS y NLoS), `lambda_gan: 0.01`, slurm `run_tenthtry10_film_*`.

### Try 11 — `TFGEleventhTry11`
- **Cambio:** **Normalización explícita** `scalar_feature_norms.antenna_height_m: 120` (entrada en [0,1] relativa a techo operativo).
- **Postpro:** alineado con experimentos FiLM de Tenth (kernels 1, sin export link budget en el YAML principal listado).

### Try 12 — `TFGTwelfthTry12`
- **Cambio:** Añade **early stopping** en training; mantiene **FiLM** (`use_scalar_film: true`) en configs **LoS / NLoS** dedicados; postpro tipo 11 (heurísticos mínimos en YAML principal).

### Try 13 — `TFGThirteenthTry13`
- **Definición:** **un único checkpoint FiLM** sobre **todo** el reparto train del HDF5 (**sin** filtros `los_only` ni `nlos_only`).
- **YAML / Slurm oficiales:** `experiments/thirteenthtry13_film/thirteenthtry13_film.yaml` · `cluster/run_thirteenthtry13_film_1gpu.slurm`.
- **Cluster manual:** `sbatch cluster/run_thirteenthtry13_film_1gpu.slurm` · **subida:** `python cluster/upload_and_submit_thirteenthtry13.py` (o `--upload-only`).
- **LoS + NLoS como dos entrenamientos separados** → eso es **`TFGFourteenthTry14`**, no el 13.
- **Inferencia:** `predict_cgan.py` → PNG path loss en **dB** denormalizado.
- *Nota:* en `configs/` y `cluster/` pueden quedar archivos copiados del fork con sufijos `_los` / `_nlos` en el nombre; **no** son el flujo definido del Try 13 (ignóralos o bórralos si confunden).

### Try 14 — `TFGFourteenthTry14`
- **Definición:** **dos checkpoints FiLM** — uno entrenado con **`los_only`** y otro con **`nlos_only`** (misma receta de modelo que el 13, distinto subconjunto de muestras).
- **FiLM recomendado:** `experiments/fourteenthtry14_film/fourteenthtry14_los_film.yaml` y `fourteenthtry14_nlos_film.yaml`.
- **Cluster:** por defecto `sbatch cluster/run_fourteenthtry14_film_los_2gpu.slurm` y `..._film_nlos_2gpu.slurm` (tiempo límite Slurm **2 días**). Los `*_film_*_1gpu.slurm` siguen en el repo por si hicieran falta.
- **Subida:** `python cluster/upload_and_submit_fourteenthtry14.py --both-film-2gpu` (o `--upload-only`).
- **Configs legacy** bajo `configs/` siguen existiendo para experimentación sin FiLM dedicado.

---

## Referencias útiles

| Tema | Dónde |
|------|--------|
| Export / paneles / rutas checkpoint | `scripts/README_EXPORT_VISUALS.md` |
| FiLM Tenth | `TFGTenthTry10/experiments/tenthtry10_film/README.md` |
| FiLM Ninth | `TFGNinthTry9/experiments/ninthtry9_film/README.md` |
| FiLM Eleventh | `TFGEleventhTry11/experiments/eleventhtry11_film/README.md` |
| FiLM Thirteenth (1 modelo, 1 GPU) | `TFGThirteenthTry13/experiments/thirteenthtry13_film/README.md` |
| FiLM Fourteenth (2 GPU por defecto; 1 GPU opcional) | `TFGFourteenthTry14/experiments/fourteenthtry14_film/README.md` |
| Subida / sbatch (todos los scripts) | `cluster/CLUSTER_COMMANDS.md` |
| Gráficas `validate_metrics_epoch_*` | `scripts/plot_cluster_validate_metrics.py` |

---

## Mantenimiento de este archivo

Al añadir **Try 15+**, duplica la fila de la tabla y un bloque en “Detalle”, indicando: dataset, `out_channels`, híbrido sí/no, escalares/FiLM, saturación dB, y slurm principal.
