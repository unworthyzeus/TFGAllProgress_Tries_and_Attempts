# TFGSeventySixthTry76

Try 76 is a configuration-only scaffold for a 12-expert topology-partitioned setup.

Scope:
- No Python files were copied.
- No model implementations were copied.
- No training scripts were copied.
- Only expert split scaffolding was created.

Split contract:
- The 12 experts use the same split definition as Try 75 (DO NOT use the splits definitions of try 68 or other previous 6 try experts, use the ones at try 75, the splitting code should produce the same results as try 75):
  - `data.split_mode: city_holdout`
  - `data.val_ratio: 0.15`
  - `data.test_ratio: 0.15`
  - `data.split_seed: 42`
- Each expert applies:
  - `partition_filter.topology_class`
  - `use_los_as_input: true`
  - `los_region_mask_mode: los_only` or `nlos_only`

Topology classes:
- `open_sparse_lowrise`
- `open_sparse_vertical`
- `mixed_compact_lowrise`
- `mixed_compact_midrise`
- `dense_block_midrise`
- `dense_block_highrise`

Expert family:
- `*_los`
- `*_nlos`

Files are under:
- `experiments/seventysixth_try76_experts/`

This scaffold is intentionally minimal. Model, loss, training, and cluster execution details must be filled in later where appropriate.
