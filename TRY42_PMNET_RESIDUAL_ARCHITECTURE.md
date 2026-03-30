# Try 42: PMNet-style residual path-loss model

## Why Try 42 exists

`Try 41` showed that the calibrated physical prior is wired correctly, but the current U-Net-based family only improves modestly on top of that prior.

That suggests the bottleneck is no longer just the prior definition. It is also the backbone itself.

So `Try 42` changes the path-loss network family more radically:

- keep the current calibrated physical prior;
- drop the cGAN framing;
- stop using the U-Net backbone;
- predict only the residual over the prior.

In short:

- `final_prediction = calibrated_prior + learned_residual`

but now the residual is predicted by a different network family.

## Paper motivation from `TFG_Proto1`

The main architecture decision is motivated by the local paper review already collected in `TFG_Proto1`, especially:

- `TFG_Proto1/docs/markdown/2402.00878v1 (2)/2402.00878v1 (2).md`

That review highlights three important points:

1. Standard U-Nets are limited when long-range propagation needs to be modeled well.
2. Dilated convolutions help increase the effective receptive field.
3. PMNet is a stronger baseline than plain RadioUNet when longer-distance relationships matter.

The paper summary describes PMNet as:

- a relatively deep encoder with stacked ResNet-style layers;
- followed by several parallel convolutional branches with different dilation rates.

That is the main architectural idea used here.

## What Try 42 changes

### 1. Backbone change

`Try 42` uses a PMNet-inspired residual regressor:

- residual encoder blocks instead of a U-Net encoder-decoder;
- a multi-branch dilated context module after the encoder;
- top-down feature fusion closer to an FPN-style head than to a symmetric U-Net decoder;
- a direct regression head for the residual map.

The implementation lives in:

- `TFGFortySecondTry42/model_pmnet.py`

### 2. Training change

`Try 42` no longer trains through the cGAN path. It uses a dedicated residual-regression trainer:

- `TFGFortySecondTry42/train_pmnet_residual.py`

This means:

- no discriminator step;
- no adversarial loss;
- no generator/discriminator scheduling;
- only supervised residual learning over the calibrated prior.

### 3. Residual as the main target

The network predicts:

- `residual_pred`

and the final map is reconstructed as:

- `prior + residual_pred`

The loss then includes:

- final path-loss reconstruction loss on the reconstructed map;
- residual supervision loss on `(target - prior)`;
- multiscale path-loss loss on the reconstructed final map.

So the model is explicitly encouraged to learn the correction rather than to relearn the whole propagation law.

## Inputs kept from the current system

The model still uses the current calibrated prior system from `Try 41`, including:

- topology map;
- LoS mask;
- distance map;
- calibrated hybrid formula prior;
- antenna height as a scalar channel.

So `Try 42` changes the network family, not the prior definition.

## New validation metrics

`Try 42` also expands validation reporting beyond a single global RMSE.

Validation JSON now includes:

- global `path_loss` RMSE;
- global `prior-only` RMSE;
- RMSE by LoS and NLoS;
- RMSE by city type;
- RMSE by antenna-height bin;
- RMSE by the combined calibration regime:
  - `city_type x LoS/NLoS x antenna_bin`

These regime metrics are written into the validation JSON under:

- `_regimes`

and the plotting script was updated so that these new curves can be plotted directly.

## Why this is a better next step than just adding channels

The previous family already showed that:

- bigger models do not reliably help;
- extra losses on top of the same U-Net often only move details around;
- the calibrated prior already explains a large part of the easy structure.

So the next meaningful change is not “more channels”.

It is:

- a better long-range backbone;
- plus a cleaner residual-learning formulation.

That is exactly what `Try 42` is meant to test.
