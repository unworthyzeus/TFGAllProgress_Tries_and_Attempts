# Try 56 Training Dynamics Note

`Try 56` adopts the same optimizer hypothesis documented for `Try 55`:

- keep a constant learning rate;
- use stronger weight decay than usual;
- avoid early scheduler-driven cooling.

The full rationale lives in:

- [../TFGFiftyFifthTry55/GROKKING_STYLE_OPTIMIZATION_NOTE.md](../TFGFiftyFifthTry55/GROKKING_STYLE_OPTIMIZATION_NOTE.md)

For `Try 56`, the active profile is intentionally milder than `Try 55`:

- `generator_optimizer = adamw`
- `generator_lr = 3.0e-4`
- `weight_decay = 0.10`
- `ema_decay = 0.99`
- `lr_scheduler = none`

`Try 56` is now generator-only; the discriminator/GAN path was removed from the active trainer and configs.

This is still an experimental optimization choice, not a standard recipe.

## Rewind behavior

`Try 56` also uses early stopping with `rewind_to_best_model = true`.
When patience is exhausted, the trainer reloads `best_model.pt`, restores the
optimizer/scaler/scheduler state, resets the patience counter, and then
continues with the next epoch number.

That means the weights jump back to the best checkpoint, but the epoch index in
the logs and plots keeps increasing instead of restarting from the best epoch.
