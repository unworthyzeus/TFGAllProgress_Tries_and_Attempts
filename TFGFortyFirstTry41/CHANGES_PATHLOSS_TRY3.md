# Changes For TFGThirdTry3

This file summarizes what changed in `TFGThirdTry3` compared with the previous tries.

Important base choice:

- `TFGThirdTry3` is based on `TFG_FirstTry1`
- `TFG_FirstTry1` currently has the best saved base path-loss checkpoint
- `TFGSecondTry2` is only used as a source for selected path-loss-specific improvements

## Structural Change

- Created a new experiment branch: `TFGThirdTry3`
- Used `TFG_FirstTry1` as the starting tree and main reference baseline
- Ported only path-loss-specific support from `TFGSecondTry2` into the new branch

## Path-Loss Data Handling

- Added linear path-loss supervision support (`predict_linear: true`)
- Added path-loss saturation masking via `data.path_loss_saturation_db`
- Added optional distance-map input support
- Added path-loss-aware physical and linear metric reporting

These were kept because they are directly about path loss, not because `TFGSecondTry2` is the main base.

## Model Changes

- Added hybrid generator mode in `model_unet.py`
- Hybrid mode now exposes:
  - dedicated coarse `path_loss` head
  - dedicated confidence head
- Kept the output flat so the rest of the pipeline remains compatible

## Training Changes

- Added confidence target generation from path-loss error during training
- Added masked confidence MSE loss with configurable weight
- Allowed `model.out_channels` to be larger than `len(target_columns)` when hybrid mode is enabled
- Limited the discriminator to the supervised target channels only
- Kept weighted metric selection for checkpointing

## Evaluation Changes

- Added fused path-loss evaluation for hybrid mode
- Path-loss physical metrics can now be computed on the final fallback-fused map
- Added hybrid summary fields such as confidence mean

## Prediction Changes

- Added export of:
  - `path_loss_confidence`
  - `path_loss_low_confidence_mask`
  - `path_loss_heuristic_prior`
  - `path_loss_coarse_physical`
  - final fused `path_loss`

## Config Changes

Added path-loss-only configs:

- `configs/cgan_unet_hdf5_pathloss_only_amd_midvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_midvram_nogan.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_midvram_bigger.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_lowvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_max.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_cuda_max.yaml`

Added hybrid configs:

- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_midvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_midvram_bigger.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_lowvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_max.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max.yaml`
