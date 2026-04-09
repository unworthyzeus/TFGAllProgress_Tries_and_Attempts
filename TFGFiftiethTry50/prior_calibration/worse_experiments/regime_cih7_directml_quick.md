# Try 45 prior-only obstruction-aware calibration

This calibration is fitted only on the training split and evaluated only on the validation split.

## Design

- Base formula: `hybrid_shadowed_ripple_two_ray_cih7_nlos`.
- Regimes: `city_type × LoS/NLoS × antenna-height bin`.
- Extra NLoS-aware pixel features:
  - `log(1 + distance)`
  - local building occupancy density at a near scale
  - local building occupancy density at a broader scale
  - local mean building-height proxy at both scales
  - broad-scale occupancy-distance interaction
  - local NLoS support at near and broad scales
  - `Eq. 12`-style shadow-sigma feature

The goal is to strengthen the prior specifically where Try 42 still fails: NLoS, lower antennas, and denser urban morphologies.

## Validation RMSE

- `overall`: RMSE `23.7172 dB`, MAE `12.1052 dB`, count `47726515`
- `LoS`: RMSE `3.3741 dB`, MAE `2.5562 dB`, count `31939056`
- `NLoS`: RMSE `40.9567 dB`, MAE `31.4234 dB`, count `15787459`

## Notes

- The official metric mask ignores buildings (`topology != 0`).
- No city ID regression is used beyond the train-defined city-type grouping already present in the previous calibration.
- This keeps the prior more structured while still aiming to generalize to unseen cities and a new dataset.
