# Try 80 Vendored Calibrations

This directory vendors the frozen prior calibration JSON files that Try 80
needs at runtime so cluster uploads do not depend on sibling Try 78 / Try 79
folders being present.

Current files:

- `try78_los_two_ray_calibration.json`
- `try78_nlos_regime_calibration.json`
- `try79_calibration.json`

Source of truth:

- `TFGSeventyEighthTry78/final_calibrations/los_two_ray_calibration.json`
- `TFGSeventyEighthTry78/final_calibrations/nlos_regime_calibration.json`
- `TFGSeventyNinthTry79/calibration.json`
