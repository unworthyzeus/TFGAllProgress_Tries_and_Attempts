# Try55 / Try56 LR and EMA Audit

This note records which learning-rate fields are actually consumed in the active Try55 and Try56 expert runs, and which ones are only metadata or obsolete for the current chain.

## Try55 active experts

Source files:
- `TFGFiftyFifthTry55/experiments/fiftyfifthtry55_partitioned_stage1/*.yaml`
- `TFGFiftyFifthTry55/scripts/generate_try55_configs.py`

Used by training:
- `training.learning_rate = 0.001`
- `training.weight_decay = 0.10`
- `training.ema_decay = 0.99`

Not used as optimizer inputs:
- `experiment.learning_rate` in validation JSON files is only a snapshot/metadata field.
- `runtime.learning_rate` is a runtime log of the optimizer state after resume, not a config value.

Resume behavior:
- The Try55 trainer now resets the optimizer LR and weight decay to the current `training.*` config after loading a checkpoint, and restores generator EMA state when the checkpoint has it.

Notes:
- All six active Try55 expert YAMLs share the same `learning_rate` and `weight_decay` values.
- The plot script reads `runtime.learning_rate`, which is the effective optimizer LR during resumed training.
- Validation and best-checkpoint selection now use the EMA weights when EMA is enabled.

## Try56 active experts

Source files:
- `TFGFiftySixthTry56/experiments/fiftysixthtry56_topology_experts/*.yaml`
- `TFGFiftySixthTry56/scripts/generate_try56_configs.py`

Used by training:
- `training.generator_lr = 0.001`
- `training.weight_decay = 0.10`
- `training.ema_decay = 0.99`

Resume behavior:
- The Try56 trainer now resets generator LR and weight decay to the current `training.*` config after loading a checkpoint, and restores generator EMA state when the checkpoint has it.

Not used in the emitted expert YAMLs:
- There is no standalone generic `learning_rate` field in the Try56 expert configs.
- The generator-only Try56 configs no longer emit discriminator optimizer settings, `lambda_gan`, or `adversarial_loss`.

Notes:
- All six active Try56 expert YAMLs now share the same generator LR, weight decay, and EMA decay.
- Validation and best-checkpoint selection use the EMA weights when EMA is enabled.

## What is used vs not used

Used in the active runs:
- Try55: `training.learning_rate`, `training.weight_decay`, `training.ema_decay`
- Try56: `training.generator_lr`, `training.weight_decay`, `training.ema_decay`

Not used for optimizer setup in the active runs:
- Try55: `experiment.learning_rate` metadata in validation JSON
- Try55: historical `runtime.learning_rate` values are logs, not config inputs
- Try56: a generic `learning_rate` field is not part of the active expert configs
- Try56: discriminator optimizer fields are no longer part of the active expert configs

## Scope note

The Try55 topology classifier has its own learning-rate settings and was not part of the active six-job expert chain. I left it unchanged.
