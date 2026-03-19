# Changes For TFGFourthTry4

This file summarizes what changed in `TFGFourthTry4` compared with `TFGThirdTry3`.

## Structural Change

- Created `TFGFourthTry4` as a full copy of `TFGThirdTry3`
- FourthTry4 will implement: **optimize the fused path-loss map in dB** during training

## Planned Training Change

In ThirdTry3:
- Path-loss channel: optimized via MSE on raw normalized output
- Validation: path_loss_rmse_physical on fused map (blend + fallback) in dB

In FourthTry4 (planned):
- Path-loss channel: optimize via MSE on the **fused map in dB**
- Requires a differentiable version of the fusion pipeline for backprop
- Validation: unchanged (same fused metric)

## Status

- [x] Copy ThirdTry3 → FourthTry4
- [x] Implement differentiable fused-path-loss loss
- [x] Wire into train_cgan.py
- [ ] Run ablation vs ThirdTry3

## Implementation Details

- `compute_fused_path_loss_mse_differentiable`: builds fused map (blend of pred + heuristic) in dB, returns MSE scaled by `1/scale²` for gradient stability
- Heuristic uses `avg_pool2d` (differentiable) instead of median filter
- When `path_loss_hybrid` enabled, path-loss channel uses fused MSE; other channels use raw recon loss
