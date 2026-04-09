# Try 45 prior-only obstruction-aware calibration

This calibration is fitted only on the training split and evaluated only on the validation split.

## Design

- Base formula: `hybrid_damped_coherent_two_ray_deepshadow_vinogradov_nlos`.
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

- `overall`: RMSE `23.7789 dB`, MAE `12.2533 dB`, count `48861408`
- `LoS`: RMSE `3.6156 dB`, MAE `2.8192 dB`, count `33048375`
- `NLoS`: RMSE `41.4711 dB`, MAE `31.9702 dB`, count `15813033`

## Notes

- The official metric mask ignores buildings (`topology != 0`).
- No city ID regression is used beyond the train-defined city-type grouping already present in the previous calibration.
- This keeps the prior more structured while still aiming to generalize to unseen cities and a new dataset.
