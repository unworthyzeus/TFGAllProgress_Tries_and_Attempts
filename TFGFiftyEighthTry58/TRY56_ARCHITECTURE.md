# Try 58 Architecture

## Base

`Try 58` reuses the same base model family as `Try 26`:
- same `UNetGenerator`
- same `CKMUNet` encoder-decoder structure
- same scalar FiLM conditioning for `antenna_height_m`

The architectural change is intentionally small:
- configurable dropout instead of hardcoded dropout
- one extra auxiliary output channel for `no_data`

## Why the topology mask is needed

In the newer HDF5 dataset, `delay_spread` and `angular_spread` are mostly zero on topology-occupied pixels. Those zeros behave like missing-data regions rather than useful regression targets.

So in `Try 58`:
- `topology_mask = topology_map > threshold`
- regression losses are applied on `1 - topology_mask`
- the model also predicts `no_data`, which is supervised with `BCE`

This keeps the regression problem focused on valid pixels while still teaching the network where the invalid regions are.

## Expert routing

`Try 58` uses six topology experts instead of one global model. The routing is deterministic and comes from `topology_map` statistics:
- building density
- mean occupied height

The six topology classes are:
- `open_sparse_lowrise`
- `open_sparse_vertical`
- `mixed_compact_lowrise`
- `mixed_compact_midrise`
- `dense_block_midrise`
- `dense_block_highrise`

## Dropout

`Try 26` used fixed dropout:
- `down3 = 0.10`
- `down4 = 0.20`
- `up1 = 0.10`

`Try 58` increases it to:
- `dropout_down3 = 0.18`
- `dropout_down4 = 0.30`
- `dropout_up1 = 0.16`

The goal is to improve generalization after splitting the dataset into smaller expert-specific subsets.
