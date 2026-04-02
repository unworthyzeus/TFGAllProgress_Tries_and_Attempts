# Try 46

`Try 46` is the first explicit `LoS / NLoS` branching path-loss model in the recent family.

It keeps:

- the calibrated physical prior
- the PMNet-style shared trunk
- the masked official metric
- the regime-level diagnostics

It changes the residual formulation:

- `LoS` uses a lightweight dedicated residual head
- `NLoS` uses a stronger mixture-of-experts residual head
- the final residual is blended using the explicit `LoS` map input

So the practical form is:

`path_loss = calibrated_prior + residual_LoS/NLoS`

where the residual head is no longer shared equally across both regimes.

Main files:

- `model_pmnet.py`
- `train_pmnet_residual.py`
- `predict.py`
- `evaluate.py`
- `experiments/fortysixthtry46_los_nlos_moe_prior/fortysixthtry46_los_nlos_moe_prior.yaml`

Cluster entry points:

- `cluster/run_fortysixthtry46_los_nlos_moe_prior_1gpu.slurm`
- `cluster/run_fortysixthtry46_los_nlos_moe_prior_2gpu.slurm`
