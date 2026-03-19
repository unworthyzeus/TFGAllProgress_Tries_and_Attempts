## Summary of experiment tries

### ThirdTry3 – original hybrid baseline

- **Dataset**: original `CKM_Dataset.h5`.
- **Loss**: reconstruction in normalized space + hybrid terms (heuristic fusion + confidence map), mainly optimizing **MSE in normalized space**, not physical RMSE in dB.
- **Architecture**: U-Net + cGAN with the original configuration (no fused-in-dB, no direct MSE in dB).
- **Issues**: very large `outputs/` (≈11 GB), no aggressive checkpoint pruning, weaker metrics (~22 dB path-loss RMSE).

### FourthTry4 – fused map in dB

- **Key idea**: train on the **fused map in dB** (prediction + heuristic) instead of the raw prediction map.
- **Changes**:
  - Added `compute_fused_path_loss_mse_differentiable`: blends prediction + heuristic using the confidence map, converts to dB, and optimizes **MSE of the fused map in dB**.
  - Still uses the old dataset but with a loss more aligned with the physical metric.
  - Slurm scripts for 2 GPUs (1 day 16 h) and pruning of `epoch_*_cgan.pt`.
- **Outcome**: better alignment with the selection metric and slightly better RMSE than ThirdTry3.

### FifthTry5 – direct MSE in dB, no fusion during training

- **Key idea**: simplify to **MSE(pred_db, target_db)** directly in dB, with no fusion during training.
- **Changes**:
  - Added `compute_path_loss_mse_db_direct`: compares `pred_db` vs `target_db` in dB, scaled by `1/scale²` for gradient stability.
  - Training no longer mixes heuristics into the loss; fusion is only used in postprocessing/inference.
  - Same architecture and dataset as FourthTry4, with better checkpoint cleanup.
- **Outcome**: more interpretable curve, RMSE around ~20.2 dB but somewhat oscillatory on the old dataset.

### SixthTry6 – new dataset + conservative LR scheduler

- **New dataset**: `CKM_Dataset_180326.h5`, centralized in `Datasets/` and referenced from configs (no per-try copies).
- **Key idea**:
  - Keep FifthTry5’s loss (direct MSE in dB, no fusion in training).
  - Use a **lower learning rate from the start** + **`ReduceLROnPlateau`** for both generator and discriminator.
  - Reduce per-epoch cost with `subset_size: 100` to iterate faster on the new dataset.
- **Concrete changes**:
  - `generator_lr = 5e-5`, `discriminator_lr = 2e-5`.
  - `lr_scheduler: reduce_on_plateau`, `patience: 3`, `min_lr: 5e-6`, applied to **opt_g** and **opt_d**, logging `lr_generator` and `lr_discriminator`.
  - Dedicated config `...sixthtry6.yaml` with `hdf5_path: CKM_Dataset_180326.h5` and a separate `output_dir`.
  - New Slurm scripts:
    - `run_sixthtry6_2gpu.slurm` (1 day 16 h, 2 GPUs).
    - `run_sixthtry6_4gpu_4h.slurm` (4 GPUs, 4 h).
- **Outcome (so far)**:
  - RMSE drops from ~21.6 dB to **~19.9 dB** around epoch 10 on the new dataset.
  - Occasional validation “spikes” (e.g., epoch 11), but the true best epoch stays around ~19.9 dB.

---

## Latest cluster results (best checkpoint RMSE in physical dB)

From the most recent `validate_metrics_cgan_best.json` downloaded into `c:\TFG\TFGpractice\cluster_outputs`:

- **ThirdTry3**: path loss `RMSE_physical = 19.790 dB` (hybrid_fused_metrics = `true`)
- **FourthTry4**: path loss `RMSE_physical = 23.115 dB` (hybrid_fused_metrics = `true`)
- **FifthTry5**: path loss `RMSE_physical = 19.747 dB` (hybrid_fused_metrics = `true`)
- **SixthTry6**: path loss `RMSE_physical = 19.676 dB` (hybrid_fused_metrics = `true`)

