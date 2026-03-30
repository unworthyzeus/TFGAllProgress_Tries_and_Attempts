# Try 44

This try is a more faithful PMNet-v3-style control branch without the physical prior.

It keeps:

- a PMNet-v3-style encoder/context/decoder closer to the official repo,
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
- `experiments/fortyfourthtry44_pmnet_v3_no_prior/fortyfourthtry44_pmnet_v3_no_prior.yaml`

Cluster entry points:

- `cluster/run_fortyfourthtry44_pmnet_v3_no_prior_1gpu.slurm`
- `cluster/run_fortyfourthtry44_pmnet_v3_no_prior_2gpu.slurm`
