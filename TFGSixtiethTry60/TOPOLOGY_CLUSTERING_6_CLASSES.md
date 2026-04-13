# Topology Clustering Into 6 Classes

This note documents how the dataset is partitioned into the 6 topology classes used by the expert models in the Try 55-60 family.

## Where It Is Defined

- Shared class list: `C:\TFG\TFGpractice\TFGFiftySeventhTry57\data_utils.py`
- Threshold resolution: `C:\TFG\TFGpractice\TFGFiftySeventhTry57\data_utils.py`
- Metadata computation per sample: `C:\TFG\TFGpractice\TFGFiftySeventhTry57\data_utils.py`
- Shared family config generator introduced in Try 57: `C:\TFG\TFGpractice\TFGFiftySeventhTry57\scripts\generate_try57_configs.py`
- Shared family expert registry: `C:\TFG\TFGpractice\TFGFiftySeventhTry57\experiments\fiftyseventhtry57_partitioned_stage1\fiftyseventhtry57_expert_registry.yaml`

## The 6 Classes

The ordered class list is:

1. `open_sparse_lowrise`
2. `open_sparse_vertical`
3. `mixed_compact_lowrise`
4. `mixed_compact_midrise`
5. `dense_block_midrise`
6. `dense_block_highrise`

## Features Used To Assign A Class

Each sample is assigned to one class using only two topology statistics computed from `topology_map`:

- `building_density`
  Definition: fraction of pixels where `topology_map != non_ground_threshold`
  In practice, with `non_ground_threshold = 0.0`, this is the fraction of non-zero topology pixels.

- `mean_height`
  Definition: mean of `topology_map` over only the non-ground pixels.

So the clustering is not learned by k-means or another unsupervised algorithm. It is a hand-defined rule system based on density plus mean height.

## Thresholds

The thresholds are resolved in this priority order:

1. `data.topology_partitioning` in the YAML
2. `city_type_thresholds` inside the prior calibration JSON
3. hardcoded defaults

For the shared family configs, the YAML explicitly sets:

- `density_q1 = 0.12`
- `density_q2 = 0.28`
- `height_q1 = 12.0`
- `height_q2 = 28.0`

These are the thresholds currently used by the expert configs in the Try 55-60 family.

## Decision Rules

The class assignment rule is:

```text
if density <= density_q1:
    if height <= height_q1:
        open_sparse_lowrise
    else:
        open_sparse_vertical

elif density >= density_q2:
    if height <= height_q2:
        dense_block_midrise
    else:
        dense_block_highrise

else:
    if height <= height_q1:
        mixed_compact_lowrise
    else:
        mixed_compact_midrise
```

With the current thresholds, that means:

- `open_sparse_lowrise`
  Sparse city fabric and low average height.
  Condition: `density <= 0.12` and `mean_height <= 12`

- `open_sparse_vertical`
  Sparse city fabric but taller structures.
  Condition: `density <= 0.12` and `mean_height > 12`

- `mixed_compact_lowrise`
  Intermediate density and low average height.
  Condition: `0.12 < density < 0.28` and `mean_height <= 12`

- `mixed_compact_midrise`
  Intermediate density and taller average height.
  Condition: `0.12 < density < 0.28` and `mean_height > 12`

- `dense_block_midrise`
  Dense fabric but not extremely tall on average.
  Condition: `density >= 0.28` and `mean_height <= 28`

- `dense_block_highrise`
  Dense fabric and tall on average.
  Condition: `density >= 0.28` and `mean_height > 28`

## Important Detail

The current 6-way partition does not use:

- LoS/NLoS
- path loss
- delay spread
- angular spread
- obstruction features
- antenna height

for the class decision itself.

`antenna_height_m` is still computed as metadata elsewhere, and there is support code for `antenna_bin`, but it is not part of the active 6-class routing used by these expert configs.

## How The Filter Is Applied

At dataset-build time:

1. all HDF5 samples are listed
2. `building_density` and `mean_height` are computed from each `topology_map`
3. the sample gets a `topology_class`
4. `partition_filter.topology_class` keeps only samples from the requested class
5. after filtering, the split is created with `split_mode = city_holdout`

So the pipeline is:

`sample -> topology stats -> one of 6 classes -> filter by expert -> split train/val/test by city`

## Shared Expert Mapping

The original Try 57 expert registry maps one model per class, and `Try 60`
reuses the same class names and split logic:

- `open_sparse_lowrise` -> `fiftyseventhtry57_expert_open_sparse_lowrise.yaml`
- `open_sparse_vertical` -> `fiftyseventhtry57_expert_open_sparse_vertical.yaml`
- `mixed_compact_lowrise` -> `fiftyseventhtry57_expert_mixed_compact_lowrise.yaml`
- `mixed_compact_midrise` -> `fiftyseventhtry57_expert_mixed_compact_midrise.yaml`
- `dense_block_midrise` -> `fiftyseventhtry57_expert_dense_block_midrise.yaml`
- `dense_block_highrise` -> `fiftyseventhtry57_expert_dense_block_highrise.yaml`

## Interpretation

This is best thought of as a manually designed routing heuristic:

- first split by how occupied the map is
- then split by how tall the occupied pixels are on average

It is simple, deterministic, and easy to reproduce, but it is not a learned clustering algorithm.
