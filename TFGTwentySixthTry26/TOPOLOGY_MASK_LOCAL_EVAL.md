# Try 26 Local Topology-Mask Validation

Local validation was run with:
- device: `DirectML` on the local AMD GPU
- checkpoint: `cluster_outputs/TFGTwentySixthTry26/twentysixthtry26_delay_angular_gradient_t26_delay_angular_gradient_2gpu/best_cgan.pt`
- dataset: `CKM_Dataset_270326.h5`
- split: `val`

Two runs were compared on the same dataset and split:
- baseline: no topology masking in the target loss/metrics
- topology-mask eval: regression metrics only on `1 - (topology_map > 0)`

Results:

- `delay_spread` RMSE: `22.9105 ns` -> `26.2772 ns`
- `delay_spread` delta: `+3.3667 ns`
- `angular_spread` RMSE: `8.7880 deg` -> `10.0798 deg`
- `angular_spread` delta: `+1.2918 deg`

Interpretation:
- the current `Try 26` checkpoint was trained without topology masking
- on the newer dataset, many topology-occupied pixels behave like missing-data zones for `delay_spread` and `angular_spread`
- once those pixels are masked out consistently, the current model looks worse
- that is consistent with the idea behind `Try 56`: train the experts with the topology mask baked into the objective and add an explicit `no_data` head
