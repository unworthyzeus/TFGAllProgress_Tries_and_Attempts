# Try 50

`Try 50` became the prior-research sandbox around the `Try 49` / `Try 47`
family rather than a stable new training branch.

## Current status

The only prior in `Try 50` that is still considered operationally usable is the
copied calibration that `Try 49` already trusts:

- `prior_calibration/regime_obstruction_train_only_from_try47.json`

Everything else inside:

- `prior_calibration/worse_experiments/`

is archived research output from branches that did not beat the current usable
baseline.

## Main conclusion so far

- `LoS` is already reasonably strong.
- `NLoS` remains the real bottleneck.
- None of the recent formula-only or hybrid redesigns beat the copied
  `Try 47`-family baseline in a convincing way.

Best archived `NLoS` result so far:

- `nlos_delta_hgbr_specialist_2pct_results.json`
- `NLoS RMSE ~= 41.01 dB`

That is still far from the target regime.

## What was tried

Archived in:

- `prior_calibration/worse_experiments/`

Main failed / non-winning families:

- structured `NLoS` formula redesigns;
- strict hard-gated `LoS/NLoS` formula branches;
- minimal `delta_nlos` branches;
- `delta_nlos` calibration by regimes;
- `HGBoost` tabular `NLoS` delta models;
- `HGBoost + torch MLP` residual variants;
- heuristic obstruction-regime experts.

The current summary note for those branches is:

- [FAILED_PRIOR_EXPERIMENTS_SUMMARY.md](C:/TFG/TFGpractice/TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md)

## Code surface

Main files:

- `model_pmnet.py`
- `train_pmnet_prior_gan.py`
- `train_pmnet_tail_refiner.py`
- `predict.py`
- `evaluate.py`

Main prior research scripts:

- `scripts/analyze_formula_prior_generalization.py`
- `scripts/analyze_nlos_delta_hgbr.py`
- `scripts/analyze_nlos_regime_experts.py`

## Recent practical lesson

The modern/old-exact baseline comparison at larger sample fractions showed a
useful operational split:

- the `modern` branch can collapse badly under the same specialist recipe;
- the `old_exact` branch remains much more stable, but still does not push
  `NLoS` low enough.

So the next real leap likely requires a stronger change of paradigm in the
`NLoS` branch, not just another coefficient or small-regressor tweak.

## Supporting notes

- [PRIOR_NLOS_SOURCES.md](C:/TFG/TFGpractice/TFGFiftiethTry50/PRIOR_NLOS_SOURCES.md)
- [NLOS_PIPELINE_REDESIGN.md](C:/TFG/TFGpractice/TFGFiftiethTry50/NLOS_PIPELINE_REDESIGN.md)

Diagram:

- [try50_prior_system.mmd](C:/TFG/TFGpractice/diagram/try50/try50_prior_system.mmd)
