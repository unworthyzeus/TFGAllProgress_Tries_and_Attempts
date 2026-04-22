# Try 80 - Joint prior-anchored mega-model

Try 80 is a single large model that predicts together:

- `path_loss`
- `delay_spread`
- `angular_spread`

using frozen non-DL priors as anchors:

- `Try 78` for path loss
- `Try 79` for spreads

The model keeps the `Try 76 / 77` philosophy:

- sinusoidal UAV-height conditioning
- shared encoder-decoder with GroupNorm
- distribution-first residual heads
- strict city-holdout split
- ground-only masking (`topology == 0`)

Main entry points:

- training: [train_try80.py](/c:/TFG/TFGpractice/TFGEightiethTry80/train_try80.py)
- evaluation: [evaluate_try80.py](/c:/TFG/TFGpractice/TFGEightiethTry80/evaluate_try80.py)
- default config: [try80_joint_big.yaml](/c:/TFG/TFGpractice/TFGEightiethTry80/experiments/try80_joint_big.yaml)
- prior precompute: [precompute_priors_hdf5.py](/c:/TFG/TFGpractice/TFGEightiethTry80/scripts/precompute_priors_hdf5.py)
- history plotting: [plot_history.py](/c:/TFG/TFGpractice/TFGEightiethTry80/scripts/plot_history.py)

Core technical notes:

- architecture and training rationale: [DESIGN_TRY80.md](/c:/TFG/TFGpractice/TFGEightiethTry80/DESIGN_TRY80.md)
- prior formulas and references: [PRIOR_FORMULAS_TRY80.md](/c:/TFG/TFGpractice/TFGEightiethTry80/docs/PRIOR_FORMULAS_TRY80.md)
