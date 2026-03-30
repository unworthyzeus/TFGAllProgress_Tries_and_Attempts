# Try 43

This try is the PMNet control branch without the physical prior.

It keeps:

- the PMNet-inspired backbone,
- the building-mask exclusion,
- LoS input,
- distance map input,
- antenna-height channel.

It removes:

- the calibrated prior input,
- the prior + residual reconstruction,
- and therefore predicts `path_loss` directly from the learned PMNet features.

Main files:

- `model_pmnet.py`
- `train_pmnet_direct.py`
- `predict.py`
- `evaluate.py`
- `experiments/fortythirdtry43_pmnet_no_prior/fortythirdtry43_pmnet_no_prior.yaml`

Cluster entry points:

- `cluster/run_fortythirdtry43_pmnet_no_prior_2gpu.slurm`
