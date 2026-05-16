# Try78 on the Try79/Try80 split

This folder records a same-split Try78 audit requested after the original
thesis tables had used Try78's older 70/30 standalone holdout.

What was run:

- Train/calibrate Try78 path-loss priors on the Try80 training cities:
  10840 samples from 37 cities.
- Evaluate the fitted prior on the Try80 validation+test held-out cities:
  2750 validation samples plus 2590 final-test samples.
- Keep a test-only summary as a sanity check against the frozen Try80 prior
  baseline used in the final residual-model tables.

Main outputs:

- `hybrid_eval_summary_try80_valtest.json`: validation+test standalone prior
  audit used for the updated thesis Try78 tables.
- `hybrid_eval_summary_try80_test.json`: final-test-only check.
- `try78_try80_valtest_grouped_metrics.json`: derived height and topology
  tables used by the thesis.
- `calibrations/`: fitted LoS and NLoS calibration JSONs for this split.

The headline numbers remained close to the previous thesis values:

- Validation+test: 1.9072 dB overall, 1.7379 dB LoS, 3.3591 dB NLoS.
- Test-only: 1.9273 dB overall, 1.7370 dB LoS, 3.5241 dB NLoS.

Try80 thesis-table convention:

- The final Try80 tables are intentionally left on the original frozen
  Try78 prior calibration that was used together with the trained Try80
  checkpoint. A later same-split sensitivity check changed the final-test
  Try80 path-loss RMSE by only about 0.011 dB, so the reported Try80 text and
  tables were not updated.
