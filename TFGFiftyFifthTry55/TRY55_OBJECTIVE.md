# Try 55 Objective

`Try 55` keeps the partitioned-expert setup of `Try 54`, but changes the
generator objective to match the validation target as directly as possible.

## Generator objective

For each expert:

- the generator predicts a residual over the calibrated prior;
- the final prediction is still `prior + residual`;
- but the generator is optimized only with:
  - `RMSE(final_map, target_map)` over valid pixels
  - plus the auxiliary `no_data_loss`

There is no generator contribution from:

- residual-only reconstruction loss
- multiscale path-loss loss
- GAN loss

So the effective objective is:

```text
generator_loss = RMSE(final_prediction, target) + lambda_no_data * no_data_loss
```

## Why

In `Try 54`, the training loss could fall smoothly while validation
`path_loss.rmse_physical` improved only weakly or noisily. That meant the
optimization target was still not aligned enough with the metric we care about.

`Try 55` removes that mismatch by making the generator optimize the final-map
RMSE directly.

## Selection metric

Checkpoints are selected with:

- `path_loss.rmse_physical`

That means both:

- training objective
- best-model selection

are now aligned around the same physical metric family.
