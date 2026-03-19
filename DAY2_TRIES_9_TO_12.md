# IMPROVEMENTS day 2 — tries 9–12

## Dataset check (`CKM_Dataset_180326.h5`)

- Sample groups contain **only** the usual 2D maps: `topology_map`, `los_mask`, `path_loss`, `delay_spread`, `angular_spread`.
- **No HDF5 attributes** on city/sample groups in the inspected file → **no antenna height stored in the H5 yet**.
- **Per-sample antenna height** is supplied via `../Datasets/CKM_180326_antenna_height.csv` (column `antenna_height_m`), or later via HDF5 attrs/datasets with the same names used in `hdf5_scalar_specs`.

Regenerate the CSV from the file that contains **`uav_height`** (e.g. `CKM_Dataset_180326_antenna_height.h5`):

```text
python tools/gen_antenna_height_csv.py --hdf5 Datasets/CKM_Dataset_180326_antenna_height.h5 --out Datasets/CKM_180326_antenna_height.csv
```

Using only `CKM_Dataset_180326.h5` produces **all-zero** heights (that H5 has no height field). Upload scripts **always overwrite** the remote CSV so edits are not stuck behind “file exists”.

## Code changes (in each try folder’s `data_utils.py`)

- **`scalar_table_csv`** + **`hdf5_scalar_specs`**: extra input planes from CSV and/or HDF5 attrs/datasets.
- **`los_sample_filter`**: `los_only` | `nlos_only` — keeps samples where `mean(los_mask > 0.5)` is above/below **`los_classify_threshold`** (default `0.5`).
- **`path_loss_ignore_nonfinite`**: masks out non-finite path-loss pixels (and cleans values for targets); works with augment (mask is geo-augmented too).
- **`cluster/prepare_runtime_config.py`**: `--scalar-csv-path` for cluster absolute paths to the CSV.

## Folders and configs

| Try | Folder | Config(s) | Notes |
|-----|--------|-----------|--------|
| 9 | `TFGNinthTry9` | `..._ninthtry9.yaml` | Antenna height channel; `lambda_gan: 0` |
| 10 | `TFGTenthTry10` | `..._tenthtry10_los.yaml`, `..._tenthtry10_nlos.yaml` | Two separate trainings |
| 11 | `TFGEleventhTry11` | `..._eleventhtry11.yaml` | Height ÷ `scalar_feature_norms.antenna_height_m` (default **120** m) |
| 12 | `TFGTwelfthTry12` | `..._twelfthtry12_los.yaml`, `..._twelfthtry12_nlos.yaml` | LoS split + normalized height |

Train from **inside** each try directory (paths use `../Datasets/`).

## Slurm (cluster)

- Try 9 / 11: `run_*_1gpu.slurm`, `run_*_2gpu.slurm`
- Try 10 / 12: `run_*_los_1gpu.slurm`, `run_*_los_2gpu.slurm`, `run_*_nlos_1gpu.slurm`, `run_*_nlos_2gpu.slurm`

`HDF5_PATH` and `SCALAR_CSV_PATH` env vars point at shared files under `/scratch/.../TFGpractice/Datasets/`.

## Upload helpers (`TFGpractice/cluster/`)

- `upload_and_submit_ninthtry9.py --gpus 1|2`
- `upload_and_submit_tenthtry10.py --variant los|nlos --gpus 1|2`
- `upload_and_submit_eleventhtry11.py --gpus 1|2`
- `upload_and_submit_twelfthtry12.py --variant los|nlos --gpus 1|2`

Set `SSH_PASSWORD` before running.
