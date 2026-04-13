# Try 60 Training-Dynamics Note

This note records the optimizer choice used by `Try 60` after the earlier
topology-expert experiments showed a familiar pattern: training kept improving
while validation peaked early.

## What We Changed

`Try 60` keeps the long-horizon setup and the grokking-style regularization we
wanted to test:

- `generator_optimizer = adamw`
- `generator_lr = 8.0e-4`
- `weight_decay = 0.10`
- `lr_scheduler = none`
- `ema_decay = 0.99`
- `early_stopping.enabled = false`
- `epochs = 10000`

## Why AdamW

Once weight decay is this strong, we want it explicit and decoupled instead of
hidden inside the gradient as classic Adam would do. That makes the experiment
easier to reason about.

The intent is not to force textbook grokking. The intent is to keep
optimization pressure high enough that the model can still move late in
training, while regularization prevents it from collapsing into an overly
brittle solution too early.

## What To Monitor

The main signals to watch are:

- whether validation keeps improving later into the run;
- whether the long schedule just overfits;
- whether the current weight decay is too aggressive for the smallest experts.

If the run collapses, the first rollback should be lowering `weight_decay`
before changing the rest of the setup.

## Early Stopping

Current `Try 60` disables early stopping on purpose, so rewind-to-best behavior
is inactive in this run. If we later compare against an early-stopped variant,
the trainer's rewind logic can still be re-enabled there.
