# CKM HDF5 Dataset Notes

This note documents what `CKM_Dataset.h5` contains and how this project has been adapted to train on it directly.

## What is inside `CKM_Dataset.h5`

The file contains `4915` samples grouped by city.

Cities found in the file:

- Abidjan
- Abu Dhabi
- Alexandria
- Athens
- Bangalore
- Bath
- Berlin
- Bilbao
- Bratislava
- Brugge
- Buenos Aires
- Canberra
- Carcassonne
- Chefchaouen
- Chiang Mai
- Cleveland
- Copenhagen
- Dalian
- Dhaka
- Dubrovnik
- Fortaleza

Each sample contains the same five 2D maps with shape `513 x 513`:

- `topology_map` as `float32`
- `path_loss` as `uint8`
- `delay_spread` as `uint16`
- `angular_spread` as `uint8`
- `los_mask` as `uint8`

The current HDF5 training configs keep the native spatial resolution and train at `513 x 513`.

Observed global ranges during inspection:

- `topology_map`: `0.0` to about `234.64`
- `path_loss`: `0.0` to `180.0`
- `delay_spread`: `0.0` to `910.0`
- `angular_spread`: `0.0` to `180.0`
- `los_mask`: `0.0` to `1.0`

There are no HDF5 attributes carrying antenna height, antenna power, bandwidth, or frequency.

## Mapping to the supervisor notes

- There is no antenna field in the HDF5.
- There is no bandwidth field in the HDF5.
- `path_loss` is present, which is consistent with using power or path loss later for SNR-related calculations.
- `delay_spread` is present without any stored bandwidth parameter.
- `los_mask` is present and can be trained directly as a binary LoS target.

## Antenna reference frame

The antenna is always centered at `(0, 0)` and the map spans `[-256, 256]` in each axis.

That means the direct HDF5 pipeline does not need an extra per-sample antenna-position input:

- the spatial reference is already fixed by construction
- every `513 x 513` sample is already aligned to the same antenna-centered coordinate frame
- adding synthetic `row` or `col` antenna features would be redundant unless the dataset format changes

## How the project was adapted

The training code now supports two dataset formats:

- `manifest`: the original CSV + PNG workflow
- `hdf5`: direct loading from `CKM_Dataset.h5`

For HDF5 training, the model uses:

- Input: `topology_map`
- Targets: `delay_spread`, `angular_spread`, `path_loss`, `los_mask`

This is intentionally different from the original proposal configs, which expected:

- `channel_power`
- `augmented_los`

Those two targets are not stored in the HDF5. Training on `path_loss` and `los_mask` is the most direct adaptation.

The loader resizes tensors to `data.image_size`, and the HDF5 configs now set `image_size: 513`, so training runs at the native dataset resolution rather than the earlier reduced `128 x 128` setup.

## Added configs

New configs were added for direct HDF5 training:

- `configs/baseline_hdf5.yaml`
- `configs/baseline_hdf5_cuda_max.yaml`
- `configs/proposal_regression_only_hdf5.yaml`
- `configs/cgan_unet_hdf5.yaml`
- `configs/baseline_hdf5_amd.yaml`
- `configs/cgan_unet_hdf5_amd.yaml`
- `configs/cgan_unet_hdf5_amd_midvram.yaml`
- `configs/cgan_unet_hdf5_amd_lowvram.yaml`
- `configs/cgan_unet_hdf5_cuda_max.yaml`

### `baseline_hdf5.yaml`

Use this for the plain U-Net with four outputs:

- `delay_spread`
- `angular_spread`
- `path_loss`
- `los_mask`

`los_mask` uses `bce` loss.

### `proposal_regression_only_hdf5.yaml`

Use this if you only want the three regression targets:

- `delay_spread`
- `angular_spread`
- `path_loss`

### `cgan_unet_hdf5.yaml`

Use this for the cGAN + U-Net setup with the same four HDF5-native targets as `baseline_hdf5.yaml`.

### `baseline_hdf5_amd.yaml`

Use this on a local AMD GPU setup such as an RX 7800 XT.

- Uses `runtime.device: directml`
- Disables AMP
- Lowers `batch_size` and `num_workers` to safer defaults for Windows + DirectML

### `cgan_unet_hdf5_amd.yaml`

Use this for the cGAN run on an AMD GPU with the same DirectML-oriented constraints.

### `cgan_unet_hdf5_amd_midvram.yaml`

Use this if you want a more aggressive local AMD run than the default config, but without jumping all the way to the old `amd_max` profile.

- `base_channels: 48`
- `disc_base_channels: 32`
- `batch_size: 2`
- `gradient_checkpointing: true`
- `generator_optimizer: rmsprop`
- `discriminator_optimizer: rmsprop`

### `cgan_unet_hdf5_amd_lowvram.yaml`

Use this if the RX 7800 XT starts spilling into shared memory and you want to stay closer to the 16 GB of dedicated VRAM.

- `base_channels: 24`
- `disc_base_channels: 12`
- `gradient_checkpointing: true`
- `batch_size: 1`
- `generator_optimizer: rmsprop`
- `discriminator_optimizer: rmsprop`

### `baseline_hdf5_cuda_max.yaml`

Use this for a larger CUDA U-Net run on the cluster when you want to push a single GPU harder than the default baseline config.

- `base_channels: 128`
- `runtime.device: cuda`
- `amp: true`
- VRAM-based dynamic batch sizing enabled with a `24 GB` reference point

### `cgan_unet_hdf5_cuda_max.yaml`

Use this for a larger CUDA cGAN run on the cluster when you want a wider generator and discriminator than the default HDF5 cGAN config.

- `base_channels: 96`
- `disc_base_channels: 96`
- `runtime.device: cuda`
- `amp: true`
- VRAM-based dynamic batch sizing enabled with a `24 GB` reference point

## Training commands

Run from `TFGpractice/TFG_FirstTry1`.

### U-Net

```bash
python train.py --config configs/baseline_hdf5.yaml
```

```bash
python train.py --config configs/baseline_hdf5_cuda_max.yaml
```

### U-Net, regression-only

```bash
python train.py --config configs/proposal_regression_only_hdf5.yaml
```

### cGAN + U-Net

```bash
python train_cgan.py --config configs/cgan_unet_hdf5.yaml
```

```bash
python train_cgan.py --config configs/cgan_unet_hdf5_cuda_max.yaml
```

## Remote cluster option

New SLURM scripts were added for remote execution:

- `cluster/run_train_hdf5.slurm`
- `cluster/run_train_cgan_hdf5.slurm`
- `cluster/run_eval_hdf5.slurm`
- `cluster/run_multi_gpu_hdf5.slurm`

Default usage:

```bash
sbatch cluster/run_train_hdf5.slurm
sbatch cluster/run_train_cgan_hdf5.slurm
sbatch cluster/run_eval_hdf5.slurm
```

To request more than one GPU with the current codebase, use the multi-GPU launcher:

```bash
sbatch cluster/run_multi_gpu_hdf5.slurm
```

These SLURM scripts now generate a temporary runtime config and adjust `training.batch_size` using the GPU VRAM detected at launch time.

Current scaling rule:

- uses the config batch size as the reference at `16 GB`
- scales linearly with detected VRAM
- applies a `0.9` safety factor
- clamps to per-config `min_batch_size` and `max_batch_size`

If VRAM autodetection fails, the original config batch size is used unchanged.

You can override the config or checkpoint without editing the script:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/proposal_regression_only_hdf5.yaml cluster/run_train_hdf5.slurm
sbatch --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5.yaml cluster/run_train_cgan_hdf5.slurm
sbatch --export=ALL,CONFIG_PATH=configs/baseline_hdf5.yaml,CHECKPOINT_PATH=outputs/baseline_hdf5_run/best.pt cluster/run_eval_hdf5.slurm
```

CUDA max examples:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/baseline_hdf5_cuda_max.yaml cluster/run_train_hdf5.slurm
sbatch --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_cuda_max.yaml cluster/run_train_cgan_hdf5.slurm
```

Important: the current training code is still single-GPU per process. The new multi-GPU launcher uses several GPUs in the same SLURM allocation by starting one independent training process per visible GPU, each with its own output directory suffix and seed offset. It is useful for parallel sweeps or repeated runs, but it is not `DistributedDataParallel`.

Examples:

```bash
sbatch --gres=gpu:4 -p medium_gpu --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_cuda_max.yaml cluster/run_multi_gpu_hdf5.slurm
sbatch --gres=gpu:8 -p big_gpu --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_cuda_max.yaml cluster/run_multi_gpu_hdf5.slurm
sbatch --gres=gpu:4 -p medium_gpu --export=ALL,CONFIG_PATH=configs/baseline_hdf5_cuda_max.yaml,TRAIN_SCRIPT=train.py cluster/run_multi_gpu_hdf5.slurm
```

Each worker writes to a different output directory derived from the base config, for example `outputs/cgan_unet_hdf5_cuda_max_gpu0`, `outputs/cgan_unet_hdf5_cuda_max_gpu1`, and so on.

If the cluster uses a different Python executable or virtual environment path, override:

- `PYTHON_BIN`
- `VENV_PATH`

Example:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/baseline_hdf5.yaml,PYTHON_BIN=python3,VENV_PATH=/path/to/venv/bin/activate cluster/run_train_hdf5.slurm
```

## AMD RX 7800 XT option

For an RX 7800 XT on Windows, the practical backend is DirectML.

Install the extra dependency in the same environment:

```bash
pip install torch-directml
```

Then run one of these configs:

```bash
python train.py --config configs/baseline_hdf5_amd.yaml
python train_cgan.py --config configs/cgan_unet_hdf5_amd.yaml
python train_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml
python train_cgan.py --config configs/cgan_unet_hdf5_amd_lowvram.yaml
```

Notes:

- These AMD configs force `runtime.device: directml`.
- AMP is disabled because the current training code only enables mixed precision on CUDA.
- The AMD configs use `mse` for `los_mask` to avoid `BCEWithLogitsLoss` operators that fall back to CPU on DirectML.
- The AMD cGAN configs also use `loss.adversarial_loss: mse` so the discriminator path avoids the same BCE fallback.
- If `loss.adversarial_loss` is omitted, cGAN now defaults to `bce` on CUDA and `mse` on non-CUDA backends such as DirectML.
- The training code also disables Adam and AdamW `foreach` optimizer kernels on non-CUDA backends, which avoids DirectML fallbacks like `aten::lerp.Scalar_out`.
- The AMD configs can enable `gradient_checkpointing` in the U-Net to reduce activation memory at the cost of extra compute time.
- If you are on Linux with ROCm instead of Windows DirectML, standard PyTorch ROCm usually appears as `cuda` inside PyTorch, so the non-AMD configs may already work.
- In the cGAN HDF5 configs, inference now exports both `los_mask_probabilities.npy` and `los_mask_binary.npy`, using `postprocess.los_mask_threshold` to threshold the binary export.
- In the cGAN HDF5 configs, inference also derives `channel_power_derived_dbm.npy` from predicted `path_loss` using a fixed link-budget assumption. By default the configs assume `tx_power_dbm: 46.0` with zero extra gains/losses, and optional SNR or link-availability maps can be enabled by setting `postprocess.link_budget.bandwidth_hz` or `postprocess.link_budget.reception_threshold_dbm`.

## Evaluation

The evaluation path also supports the new HDF5 format.

Example:

```bash
python evaluate.py --config configs/baseline_hdf5.yaml --checkpoint outputs/baseline_hdf5_run/best.pt
```

## Notes on normalization

The HDF5 loader normalizes each field using config metadata instead of assuming PNG inputs.

Current defaults are:

- `topology_map`: divided by `255.0`
- `delay_spread`: divided by `1000.0`
- `angular_spread`: divided by `180.0`
- `path_loss`: divided by `180.0`
- `los_mask`: kept in `[0, 1]`

Current HDF5 configs also use:

- `image_size: 513` for native-resolution training

If you later decide to predict `channel_power` instead of `path_loss`, do not just rename the field. Derive it explicitly from the physical relation you want to model and update the config metadata accordingly.