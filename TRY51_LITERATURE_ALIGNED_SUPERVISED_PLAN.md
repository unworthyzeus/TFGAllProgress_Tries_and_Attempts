# Try 51 Literature-Aligned Supervised Plan

This note explains what `Try 51` changes relative to `Try 49/50`, and why
those changes are more aligned with the current path-loss / radio-map
literature.

## What the literature says, briefly

The strongest recurring pattern across the reviewed papers is:

- supervised regression, not GAN-first;
- simulation or physics-guided supervision first;
- adaptation or fine-tuning second;
- realistic held-out validation by geography / site / city when possible;
- and explicit handling of different propagation regimes.

Relevant sources:

- RadioUNet: [arXiv](https://arxiv.org/abs/1911.09002)
- PMNet: [arXiv](https://arxiv.org/abs/2211.10527)
- PMNet transfer-learning extension: [arXiv](https://arxiv.org/abs/2312.03950)
- A2G RT + measurements: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0140366422003954)
- Outdoor challenge: [arXiv](https://arxiv.org/abs/2310.07658)
- Indoor challenge: [arXiv](https://arxiv.org/abs/2501.13698)
- Classification + regression for path loss: [MDPI](https://www.mdpi.com/1424-8220/24/17/5855)
- Full literature summary already collected in:
  - [PATH_LOSS_MODEL_TRAINING_PAPERS.md](C:/TFG/TFGpractice/PATH_LOSS_MODEL_TRAINING_PAPERS.md)

## What looked weak in Try 49 and Try 50

`Try 49` already had several good ideas:

- strong calibrated prior;
- residual learning;
- two-stage structure;
- explicit `LoS / NLoS` metrics.

But it still had two mismatches with the literature-driven target:

1. training was not using regime-aware importance strongly enough;
2. city morphology was present in calibration JSONs, but not promoted enough as
   an automatic generalization axis.

`Try 50` explored many `NLoS` prior variants, but the best `NLoS` remained
around `41 dB` RMSE, which strongly suggests that hand-tuning the formula alone
was not enough.

## The Try 51 bet

`Try 51` keeps the useful structure:

- strong prior;
- stage1 residual predictor;
- stage2 tail refiner;

but changes the emphasis:

1. **automatic city type over city-name lookup**
2. **regime-aware weighting during training**
3. **transfer from a strong existing model instead of fragile restart**
4. **supervised residual correction, not a more adversarial pipeline**
5. **city-holdout validation instead of random sample mixing**

## Automatic city type

The intended city-type axis is:

- `open_lowrise`
- `mixed_midrise`
- `dense_highrise`

It is inferred automatically from:

- obstacle/building density
- average non-ground obstacle height

Why this is better than `city -> class` as the main path:

- it can generalize to unseen cities;
- it matches the physical morphology more directly;
- it is closer to what a practical deployment would actually know about a new
  area.

## Stage 1 changes

Stage 1 in `Try 51` uses:

- the PMNet residual generator;
- `MSE`-dominant supervised loss (`1.0 * MSE + 0.25 * L1`);
- `Adam` with zero weight decay;
- multiscale supervision enabled;
- regime-aware reweighting:
  - extra weight on `NLoS`
  - extra weight on low antenna height
  - extra weight on dense-highrise morphology

This is a compromise between:

- the literature default, which is still largely `MSE/NMSE`;
- and the practical observation that a small `L1` term can stabilize residuals
  without fully switching to an MAE-dominant objective.

It also changes the validation logic:

- `data.split_mode = city_holdout`
- whole cities are held out together for val / test
- this is much closer to the generalization setup reported in the stronger
  radio-map papers and challenges than per-sample random splitting

Finally, `Try 51` no longer keeps a discriminator alive when `lambda_gan = 0`.
That means the active branch is not just "GAN disabled by weight", but
actually runs as a supervised regressor.

## Stage 2 changes

Stage 2 in `Try 51` keeps:

- teacher-on-the-fly residual refinement;
- simple residual correction;

and adds:

- automatic city-type regime reweighting;
- a slightly stronger emphasis on hard `NLoS` morphology.

The goal is to make stage 2 a **targeted residual corrector**, which is closer
to the masked/weighted correction logic seen in the literature than to a full
second model trying to relearn everything or to a highly engineered tail-only
auxiliary objective.

## What this still does not guarantee

Even if `Try 51` is more literature-aligned, it does **not** guarantee
`< 5 dB overall RMSE`.

That target is still extremely aggressive for a dense mixed `LoS + NLoS`
dataset. `Try 51` should be read as:

- a better-founded branch,
- a more generalizable branch,
- and a more defensible experimental direction,

not as proof that the final metric target is already solved.

## Where to look in code

- branch root:
  - [TFGFiftyFirstTry51](C:/TFG/TFGpractice/TFGFiftyFirstTry51)
- main README:
  - [README.md](C:/TFG/TFGpractice/TFGFiftyFirstTry51/README.md)
- stage 1 config:
  - [fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml](C:/TFG/TFGpractice/TFGFiftyFirstTry51/experiments/fiftyfirsttry51_pmnet_prior_gan_fastbatch/fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml)
- stage 1 resume config:
  - [fiftyfirsttry51_pmnet_prior_stage1_widen112_resume_literature.yaml](C:/TFG/TFGpractice/TFGFiftyFirstTry51/experiments/fiftyfirsttry51_pmnet_prior_gan_fastbatch/fiftyfirsttry51_pmnet_prior_stage1_widen112_resume_literature.yaml)
- stage 2 config:
  - [fiftyfirsttry51_pmnet_tail_refiner_stage2.yaml](C:/TFG/TFGpractice/TFGFiftyFirstTry51/experiments/fiftyfirsttry51_pmnet_tail_refiner_fastbatch/fiftyfirsttry51_pmnet_tail_refiner_stage2.yaml)
