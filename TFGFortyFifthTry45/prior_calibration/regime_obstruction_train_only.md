# Try 45 prior-only obstruction-aware calibration

This calibration is fitted only on the training split and evaluated only on the validation split.

## Design

- Base formula: `hybrid_two_ray_cost231`.
- Regimes: `city_type × LoS/NLoS × antenna-height bin`.
- Extra NLoS-aware pixel features:
  - `log(1 + distance)`
  - local building occupancy density at a near scale
  - local building occupancy density at a broader scale
  - local mean building-height proxy at both scales
  - broad-scale occupancy-distance interaction
  - local NLoS support at near and broad scales

The goal is to strengthen the prior specifically where Try 42 still fails: NLoS, lower antennas, and denser urban morphologies.

## Validation RMSE

- `overall`: RMSE `23.6046 dB`, MAE `12.2790 dB`, count `485380287`
- `LoS`: RMSE `3.8149 dB`, MAE `3.0451 dB`, count `329805515`
- `NLoS`: RMSE `41.3219 dB`, MAE `31.8542 dB`, count `155574772`

## Notes

- The official metric mask ignores buildings (`topology != 0`).
- No city ID regression is used beyond the train-defined city-type grouping already present in the previous calibration.
- This keeps the prior more structured while still aiming to generalize to unseen cities and a new dataset.
