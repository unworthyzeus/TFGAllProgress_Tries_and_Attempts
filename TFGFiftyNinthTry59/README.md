# Try 59

`Try 59` is a topology-partitioned continuation of `Try 26`.

Main idea:
- keep the same `Try 26` U-Net family and training code path
- split training into `6` topology experts
- add a binary `topology_mask` input channel
- mask regression losses on topology-occupied pixels
- add an auxiliary `no_data` BCE head so the model explicitly learns where targets are unavailable
- use stronger dropout than `Try 26`
- use a lighter throughput-oriented profile than the inherited `Try 26` defaults

The six experts are:
- `open_sparse_lowrise`
- `open_sparse_vertical`
- `mixed_compact_lowrise`
- `mixed_compact_midrise`
- `dense_block_midrise`
- `dense_block_highrise`

Current split sizes with `city_holdout` on `CKM_Dataset_270326.h5`:
- `open_sparse_lowrise`: train `1560`, val `400`, test `350`
- `open_sparse_vertical`: train `1070`, val `230`, test `230`
- `mixed_compact_lowrise`: train `1705`, val `475`, test `505`
- `mixed_compact_midrise`: train `2095`, val `965`, test `565`
- `dense_block_midrise`: train `3565`, val `925`, test `795`
- `dense_block_highrise`: train `405`, val `225`, test `115`

Generated configs live in:
- `experiments/fiftyninthtry59_topology_experts`

Active entrypoints:
- `train_topology_expert.py`
- `evaluate_topology_expert.py`
- `validate_topology_expert.py`
- `model_topology_expert.py`
- `topology_expert_heuristics.py`

Current throughput-oriented defaults:
- `base_channels = 72`
- `batch_size = 2`
- `gradient_checkpointing = false`
- `multiscale_targets.enabled = false`
- `gradient_targets.enabled = false`
- `generator_optimizer = adamw`
- `generator_lr = 3.0e-4`
- `weight_decay = 0.10`
- `ema_decay = 0.99`
- `lr_scheduler = none`
- `dropout_down3 = 0.18`
- `dropout_down4 = 0.30`
- `dropout_up1 = 0.16`

The active Try 59 experts are generator-only; discriminator and GAN settings are no longer part of the emitted configs.

`ema_decay` is the smoothing factor for the moving-average copy of the
generator weights. Values closer to `1.0` keep more past weights and update
more slowly; slightly lower values react faster. `0.99` keeps the validation
model stable without lagging too far behind training.

Architecture notes live in:
- `TRY56_ARCHITECTURE.md`
- `TRAINING_DYNAMICS_NOTE.md`
- `../TFGFiftyFifthTry55/OVERFITTING_MITIGATION_OPTIONS.md`

To regenerate the six YAMLs:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\TFGpractice\TFGFiftyNinthTry59\scripts\generate_try59_configs.py
```
