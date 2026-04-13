# Try 60

`Try 60` is the no-prior ablation of the `Try 57` topology-expert family.

We keep:

- the same 6 topology partitions
- the same PMHHNet expert family
- the same auxiliary `no_data` head
- the same topology classifier idea for routing

But we remove:

- the formula-prior input channel
- the prior-confidence input channel
- the dependency on `prior + residual` as the semantic target

## Main Goal

The point of this try is simple:

- predict `path_loss` directly from geometry and LoS context
- measure whether the calibrated prior was genuinely helping
- or whether it was anchoring the model into a bad low-frequency bias

## Inputs

Each `Try 60` expert receives:

- `topology_map`
- `los_mask`
- `distance_map`
- `antenna_height_m` through FiLM scalar conditioning

There is no formula-prior channel in this try.

## Outputs

Each expert outputs 2 channels:

- channel `0`: direct `path_loss`
- channel `1`: auxiliary `no_data` logit

Internally, the trainer reuses the residual-style implementation from `Try 57`,
but with a zero prior when `path_loss_formula_input.enabled = false`. That
makes the first output behave as a direct path-loss prediction.

## Current Config Shape

- `arch = pmhhnet`
- `base_channels = 10`
- `hf_channels = 8`
- `scalar_hidden_dim = 24`
- `out_channels = 2`
- `dropout = 0.08`
- `batch_size = 5`
- `epochs = 10000`
- `learning_rate = 8e-4`
- `weight_decay = 0.10`
- `early_stopping = false`

## Key Files

- `train_partitioned_pathloss_expert.py`
- `predict.py`
- `scripts/generate_try60_configs.py`
- `scripts/plot_try60_metrics.py`
- `TOPOLOGY_CLUSTERING_6_CLASSES.md`
