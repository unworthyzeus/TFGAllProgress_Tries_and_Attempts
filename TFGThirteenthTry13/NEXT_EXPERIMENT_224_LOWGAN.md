# Next Experiment: 224 + Low-GAN

## Current status

The current clean cluster run is:

- config: `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max192_blend_db.yaml`
- mode: hybrid path-loss in dB
- fallback: `blend`
- GAN: disabled (`lambda_gan: 0.0`)
- world size: `5`

Early signal after epoch 1:

- `path_loss.rmse_physical ~= 22.94 dB`
- `path_loss.rmse_linear ~= 0.436`
- no invalid path-loss pixels/maps
- no OOM after the GPU cleanup

This makes the clean `192` dB run the current stable baseline for the next ablations.

## Updated execution decision

After the `224 + low-GAN` attempt failed with a clean early OOM, the next concrete run should be:

- `192` base channels
- `lambda_gan: 0.02`
- smaller discriminator
- `GroupNorm` in both generator and discriminator

Prepared execution config:

- `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max192_blend_db_lowgan_groupnorm.yaml`

Why this replaces the original `224` plan:

- `224` is too close to the VRAM limit once the discriminator is active
- the normalization ablation is still worth testing
- going back to `192` isolates the effect of low-GAN + GroupNorm with much lower OOM risk

## Prepared next config

Prepared config:

- `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max224_blend_db_lowgan.yaml`

Key choices:

- `base_channels: 224`
- `lambda_gan: 0.02`
- `disc_base_channels: 64`
- `discriminator_lr: 0.0001`
- path loss trained in `dB`
- hybrid confidence + heuristic blend kept unchanged

## Why 224 and not 256

`256` failed with a relatively clean per-rank OOM, even before the model reached validation:

- all ranks were around `10.28 GiB` used out of `10.75 GiB`
- the crash happened in `optimizer.step()`
- the missing chunk was about `576 MiB`

That does not look like a tiny margin problem. It looks like `256` is simply too close to the physical limit of the 11 GB RTX 2080 cards for this model family.

`224` is a better next step because:

- it increases generator capacity over `192`
- it stays meaningfully below `256`
- it leaves a little room to re-enable a small adversarial signal

## Why re-enable GAN only softly

The point is not to go back to a strong cGAN. The point is to test whether a weak adversarial regularizer helps the path-loss map look less over-smoothed without overwhelming the physical regression target.

So the next test should be intentionally conservative:

- small `lambda_gan`
- smaller discriminator
- slower discriminator learning rate

The goal is to avoid the old pattern where the discriminator becomes strong too early while `path_loss.rmse_physical` does not improve.

## Best structural change to try next

If we change one architectural thing, the best first candidate is:

- replace `BatchNorm2d` with `GroupNorm`

### Why this is a strong candidate

Right now both the generator and the discriminator use `BatchNorm2d`, while the effective batch size per rank is `1`.

That is a bad regime for batch normalization:

- batch statistics are noisy
- train/inference behavior can drift
- the discriminator can become unstable or overconfident
- path-loss regression can pick up unnecessary variance from normalization noise

This matters especially here because path loss is a dense physical regression problem, not a pure image synthesis task.

Relevant current code:

- `model_unet.py`: all `DoubleConv` blocks use `BatchNorm2d`
- `model_cgan.py`: `PatchDiscriminator` also uses `BatchNorm2d`

### Suggested structural variant

Create one normalization ablation:

- generator: replace `BatchNorm2d(C)` with `GroupNorm(num_groups=min(32, C // 8 or 1), num_channels=C)`
- discriminator: same idea, or keep the first discriminator block without norm and replace the later ones with `GroupNorm`

Expected benefit:

- more stable training at batch size `1`
- less sensitivity to per-rank statistics in DDP
- cleaner path-loss regression signal

Risk:

- training may become a bit slower
- the GAN branch may need a tiny LR retune afterward

## Second structural candidate

If GroupNorm helps or if we want a more path-loss-specific architectural change, the next best option is:

- predict residual path loss over a heuristic prior, not absolute path loss directly

That means:

1. build the heuristic path-loss prior first
2. feed or reuse it in the model path
3. train the network to predict a residual correction in dB
4. final prediction = `heuristic_prior_db + residual_db`

Why this may help:

- the model stops spending capacity on easy large-scale attenuation structure
- the network focuses on local deviations, shadowing, and geometry-driven corrections
- the confidence head becomes more interpretable because low confidence often corresponds to regions where the residual is hard

Risk:

- bigger code change
- requires careful clamping and metric handling in dB

## Third structural candidate

Another useful ablation would be:

- add deep supervision on an intermediate decoder scale for path loss only

Why:

- path loss has a strong large-scale spatial component
- an auxiliary coarse-resolution loss can help the decoder preserve global attenuation structure before the final high-resolution refinement

This is lower priority than GroupNorm because it changes the training graph more and is harder to isolate.

## Recommended order

1. Keep the current `192` dB no-GAN run as the reference until a few more epochs confirm the trend.
2. Run `224` with the prepared low-GAN config.
3. If `224` is promising but noisy, implement the `GroupNorm` ablation before trying larger capacity again.
4. Only if the previous steps help, try residual-over-heuristic path-loss prediction.

## Practical note

The prepared `224` low-GAN config is ready, but it should not replace the running `192` experiment unless we explicitly decide to stop that run.
