# HDF5 layout (`TFGpractice/Datasets`)

| File | Use |
|------|-----|
| **`CKM_Dataset_old_and_small.h5`** | **Tries 1–5** — small legacy subset in YAML as `../Datasets/CKM_Dataset_old_and_small.h5`. |
| **`CKM_Dataset_180326.h5`** | **Tries 6+** (sixth try onward) — larger export, **date in the filename**. |
| **`CKM_Dataset.h5`** | Classic full HDF5 **without** a date in the name; not the default for tries 1–5 in this project layout. |
| **`CKM_Dataset_180326_antenna_height.h5`** | Maps + `uav_height` for CSV generation. |

## Re-point YAML + prune checkpoints

```bash
python tools/repoint_hdf5_and_prune_checkpoints.py
python tools/repoint_hdf5_and_prune_checkpoints.py --dry-run
```

- **Prune** walks **all** `outputs/` and `cluster_outputs/` trees. It deletes `epoch_*.pt` only (keeps `best_*.pt`). **No `.json` files are deleted.**
