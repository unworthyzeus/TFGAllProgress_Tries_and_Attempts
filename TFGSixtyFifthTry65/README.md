# Try 65

`Try 65` is a deliberate grokking-style stress test.

The idea is to stop helping the optimizer:

- single stage only
- full `513x513`
- no early stopping
- no rewind to `best_model`
- no stage 2 refiner
- large constant learning rate
- high weight decay
- very long horizon (`10000` epochs)

## Main Idea

This try is intentionally simple and stubborn.

We want to see whether a long, regularized, high-LR training regime can
eventually discover a better solution after spending many epochs in a plateau,
instead of being cut short by early stopping or by a second refinement stage.

## Configuration

- `image_size = 513`
- `base_channels = 20`
- `batch_size = 1`
- `val_batch_size = 1`
- `epochs = 10000`
- `learning_rate = 1e-3`
- `weight_decay = 5e-2`
- `lr_scheduler = none`
- `early_stopping.enabled = false`
- `save_every = 25`

## Inputs

- formula prior disabled
- confidence channel disabled
- obstruction proxy channels disabled

## Expected Behaviour

This experiment may look "bad" for a long time.

That is part of the point: we are explicitly testing whether a long training
run with strong regularization eventually improves after a long delay.

The current variant is also intentionally lighter and faster than the initial
draft:

- no physical prior input
- no confidence channel
- no obstruction channels
- no gradient checkpointing

## Key Files

- `scripts/generate_try65_configs.py`
- `cluster/submit_try65_stage1_4gpu.py`
- `cluster/run_sixtyfifthtry65_4gpu.slurm`
- `train_partitioned_pathloss_expert.py`
