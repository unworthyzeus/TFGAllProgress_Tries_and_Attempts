# cGAN + U-Net implementation for `TFG_FirstTry1`

## Why this model
This implementation follows the direction suggested by the literature table and proposal:

- **U-Net / encoder-decoder with skip connections** for dense map prediction from spatial inputs.
- **cGAN / pix2pix-style training** to make predicted channel maps sharper and more structurally consistent than pure regression.
- **PatchGAN discriminator** to judge whether local spatial patterns in predicted CKM maps look realistic conditioned on the input height/LoS/scalar context.

This is aligned with the papers already tracked in the SOA table, especially:
- `U-Net: Convolutional Networks for Biomedical Image Segmentation`
- `Image-to-Image Translation with Conditional Adversarial Networks`
- CKM / radio-map papers in the Excel that motivate image-to-image style prediction

The implementation is intentionally lightweight and uses only the dependencies already present in [requirements.txt](requirements.txt).

## What was added

### 1. New architecture file
- [model_cgan.py](model_cgan.py)
  - `UNetGenerator`: wraps the existing `CKMUNet`
  - `PatchDiscriminator`: pix2pix-style discriminator over `(input, target)` pairs

### 2. New cGAN config
- [configs/cgan_unet.yaml](configs/cgan_unet.yaml)
  - Separate generator/discriminator learning rates
  - `lambda_gan` and `lambda_recon`
  - same target metadata and scalar input conventions as the baseline model

### 3. New training script
- [train_cgan.py](train_cgan.py)
  - trains discriminator on real vs fake conditioned outputs
  - trains generator with:
    - adversarial BCE loss
    - reconstruction loss over configured targets
  - supports mixed regression + BCE targets
  - saves `best_cgan.pt` and periodic `epoch_*_cgan.pt`

### 4. New inference script
- [predict_cgan.py](predict_cgan.py)
  - loads generator from cGAN checkpoint
  - exports preview PNGs
  - exports raw `.npy` predictions
  - exports denormalized physical-unit arrays for regression targets
  - exports probabilities for BCE targets like `augmented_los`

## Training objective
Generator loss:

$$
\mathcal{L}_G = \lambda_{gan} \mathcal{L}_{GAN} + \lambda_{recon} \mathcal{L}_{recon}
$$

Discriminator loss:

$$
\mathcal{L}_D = \frac{1}{2}(\mathcal{L}_{real} + \mathcal{L}_{fake})
$$

Where:
- `GAN` term uses `BCEWithLogitsLoss`
- reconstruction term is channel-wise and respects missing-target masks
- regression targets use `MSE` or `L1`
- `augmented_los` can use `BCE`

## Inputs supported
- height matrix (`input_path`)
- optional binary LoS input (`binary_los`)
- scalar inputs from manifest:
  - `antenna_height`
  - `antenna_power`
  - `bandwidth`
- constant scalar input:
  - `frequency_ghz = 7.125`

## Outputs supported
Default cGAN config predicts:
- `delay_spread`
- `angular_spread`
- `channel_power`
- `augmented_los`

`augmented_los` should be interpreted as a **soft wave-aware LoS/intensity map** rather than a purely binary label map.

This can still be reduced by editing `target_columns` and `model.out_channels` in the config.

## Current limitations
- discriminator is generic PatchGAN, not yet tuned per channel
- no spectral normalization yet
- no feature matching or perceptual losses yet
- physical target scales are still placeholder defaults and should be aligned to the real dataset encoding

## Suggested use
Train with:

```bash
python train_cgan.py --config configs/cgan_unet.yaml
```

Predict with:

```bash
python predict_cgan.py --config configs/cgan_unet.yaml --checkpoint outputs/cgan_unet_run/best_cgan.pt --input path/to/height.png --los-input path/to/binary_los.png --scalar-values antenna_height=120,antenna_power=46,bandwidth=100
```
