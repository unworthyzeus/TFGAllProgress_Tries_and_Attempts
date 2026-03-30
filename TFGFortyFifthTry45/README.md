# Try 42

This try is the first recent path-loss branch that:

- keeps the calibrated prior from `Try 41`,
- predicts only a learned residual,
- removes the discriminator path,
- and replaces the old U-Net family with a PMNet-inspired backbone.

Main files:

- `model_pmnet.py`
- `train_pmnet_residual.py`
- `predict.py`
- `evaluate.py`
- `experiments/fortysecondtry42_pmnet_prior_residual/fortysecondtry42_pmnet_prior_residual.yaml`

Cluster entry points:

- `cluster/run_fortysecondtry42_pmnet_prior_residual_1gpu.slurm`
- `cluster/run_fortysecondtry42_pmnet_prior_residual_2gpu.slurm`

The try still uses:

- the calibrated hybrid prior,
- building-mask exclusion,
- LoS input,
- distance map input,
- and antenna-height conditioning as an input channel.
