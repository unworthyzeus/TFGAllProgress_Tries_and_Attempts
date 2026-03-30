# Experiment Versions (Try 1 -> Try 45)

This document summarizes the evolution from `TFG_FirstTry1` to `TFGFortyFirstTry41`, what each family predicts, and why each new branch was opened.

## Current status

- `Try 20` and `Try 21` were useful controlled tests, but `Try 22` became the stronger clean `path_loss` baseline.
- `Try 22` established the best recent path-loss base:
  - bilinear decoder
  - multiscale path-loss supervision
- `Try 23` reopened the `delay_spread + angular_spread` branch with the same structural recipe adapted to continuous regression.
- `Try 24` was prepared locally as a multitask branch, but intentionally kept out of the active cluster reading so that the single-task results remain interpretable.
- `Try 25`, `Try 27`, and `Try 28` tested additional `path_loss` ideas around global context and topology-aware regularization.
- `Try 29` tested explicit radial supervision for `path_loss`, but did not beat the stronger `Try 22` baseline.
- `Try 30` tested stronger amplitude-aware spread supervision, but did not beat the stronger `Try 26` baseline.
- `Try 31` and `Try 32` were useful paradigm-shift tests, but they did not beat `Try 22` and `Try 26`.
- `Try 33` reuses the strong `Try 22` path-loss recipe and changes only one thing:
  - building pixels are excluded from supervision and error accumulation
- `Try 34` is the physical-prior path-loss branch:
  - hybrid two-ray + COST231 formula map as conditioning input
- `Try 35` is the lighter 1-GPU spread-side branch with the same building-mask exclusion
- `Try 36` is the clean `Try 26` spread baseline with the same building-mask exclusion
- `Try 37-40` restart that family on the newer `CKM_Dataset_270326.h5` dataset:
  - `Try 37`: `Try 33` equivalent on the new dataset
  - `Try 38`: `Try 34` equivalent on the new dataset, with a larger model than the tiny debug rerun
  - `Try 39`: `Try 35` equivalent on the new dataset
  - `Try 40`: `Try 36` equivalent on the new dataset
- `Try 41` opens the new main path-loss direction on the harder dataset:
  - `prediction = physical_prior + learned_residual`
  - with the hybrid two-ray/COST231 formula map acting as the prior anchor
- `Try 42` keeps the calibrated prior from `Try 41` but replaces the old U-Net/cGAN family with a PMNet-inspired residual regressor.
- `Try 43` is the PMNet control branch without any physical prior:
  - same masked training target
  - same LoS / distance / antenna context
  - but direct `path_loss` prediction without `prior + residual`
- `Try 44` is the more faithful PMNet-v3-style control branch:
  - still no physical prior
  - but with an encoder / ASPP / decoder structure much closer to the original PMNet repository
- `Try 43` and `Try 44` now also report the same regime-level diagnostics used in `Try 42`:
  - `LoS / NLoS`
  - `city type`
  - `antenna-height bin`
  - combined calibration-style regimes
- `Try 45` is the next step after `Try 42`:
  - same PMNet-style residual path-loss formulation
  - same `Try 22`-style good practices that still matter here:
    - bilinear resizing/fusion
    - multiscale path-loss loss
    - group norm for `batch_size = 1`
  - stronger train-only prior calibration using:
    - urban regime
    - `LoS / NLoS`
    - antenna-height bin
    - multiscale obstruction features
    - local shadow-support proxies from the LOS map
  - lightweight spatial mixture-of-experts residual head on top of the calibrated prior
- Masking was re-verified in code:
  - building pixels are excluded from training loss,
  - excluded from validation/test metrics,
  - and exported as invalid regions in error-map style visualizations.

## Fast table

| Try | Folder | Main targets | Main idea |
|---|---|---|---|
| 1 | `TFG_FirstTry1` | `path_loss`, `delay_spread`, `angular_spread` | first working HDF5 multi-output pipeline |
| 2 | `TFGSecondTry2` | same three targets | stronger early multi-output baseline |
| 3 | `TFGThirdTry3` | same three targets | larger U-Net, more GPU-oriented |
| 4 | `TFGFourthTry4` | `path_loss` + confidence | first hybrid path-loss branch |
| 5 | `TFGFifthTry5` | same as 4 | more conservative optimization |
| 6 | `TFGSixthTry6` | same as 5 | switch to the larger HDF5 dataset |
| 7 | `TFGSeventhTry7` | `path_loss` | pure regression, no GAN |
| 8 | `TFGEighthTry8` | `path_loss` | reintroduce GAN |
| 9 | `TFGNinthTry9` | `path_loss` | antenna-height conditioning |
| 10 | `TFGTenthTry10` | `path_loss` | LoS / NLoS split |
| 11 | `TFGEleventhTry11` | `path_loss` | more formal height normalization |
| 12 | `TFGTwelfthTry12` | `path_loss` | early stopping + FiLM LoS/NLoS |
| 13 | `TFGThirteenthTry13` | `path_loss` | single FiLM model on the full dataset |
| 14 | `TFGFourteenthTry14` | `path_loss` | separate FiLM LoS and NLoS checkpoints |
| 15 | `TFGFifteenthTry15` | `path_loss` | first city-regime baseline |
| 16 | `TFGSixteenthTry16` | `path_loss` | conservative city-regime variant with more geometry |
| 17 | `TFGSeventeenthTry17` | `path_loss` | alternate city-regime balance |
| 18 | `TFGEighteenthTry18` | `path_loss` | additional city-regime regularization variant |
| 19 | `TFGNineteenthTry19` | `path_loss` | strongest city-regime base before opening new families |
| 20 | `TFGTwentiethTry20` | `path_loss` | bilinear decoder to reduce checkerboard artifacts |
| 21 | `TFGTwentyFirstTry21` | `path_loss` | multiscale path-loss loss |
| 22 | `TFGTwentySecondTry22` | `path_loss` | bilinear decoder + multiscale loss |
| 23 | `TFGTwentyThirdTry23` | `delay_spread`, `angular_spread` | bilinear decoder + multiscale regression loss |
| 24 | `TFGTwentyFourthTry24` | `path_loss`, `delay_spread`, `angular_spread` | local multitask branch, not launched |
| 25 | `TFGTwentyFifthTry25` | `path_loss` | `Try 22` base + lightweight bottleneck attention |
| 26 | `TFGTwentySixthTry26` | `delay_spread`, `angular_spread` | `Try 23` base + gradient-aware spread loss |
| 27 | `TFGTwentySeventhTry27` | `path_loss` | `Try 22` base + topology-edge path-loss weighting |
| 28 | `TFGTwentyEighthTry28` | `path_loss` | combine lightweight attention with topology-edge weighting |
| 29 | `TFGTwentyNinthTry29` | `path_loss` | `Try 22` base + radial profile loss + radial gradient loss |
| 30 | `TFGThirtiethTry30` | `delay_spread`, `angular_spread` | `Try 26` base + value-weighted spread loss + hotspot-focused spread loss |
| 31 | `TFGThirtyFirstTry31` | `path_loss` | physical prior + learned residual correction |
| 32 | `TFGThirtySecondTry32` | `delay_spread`, `angular_spread` | support map + amplitude map prediction |
| 33 | `TFGThirtyThirdTry33` | `path_loss` | `Try 22` recipe + building-mask exclusion only |
| 34 | `TFGThirtyFourthTry34` | `path_loss` | hybrid two-ray/COST231 formula input + building-mask exclusion |
| 35 | `TFGThirtyFifthTry35` | `delay_spread`, `angular_spread` | lighter 1-GPU spread-side branch with building-mask exclusion |
| 36 | `TFGThirtySixthTry36` | `delay_spread`, `angular_spread` | `Try 26` recipe + building-mask exclusion only |
| 37 | `TFGThirtySeventhTry37` | `path_loss` | `Try 33` rerun on `CKM_Dataset_270326.h5` |
| 38 | `TFGThirtyEighthTry38` | `path_loss` | `Try 34` rerun on `CKM_Dataset_270326.h5` with a larger model than the tiny debug variant |
| 39 | `TFGThirtyNinthTry39` | `delay_spread`, `angular_spread` | `Try 35` rerun on `CKM_Dataset_270326.h5` |
| 40 | `TFGFortiethTry40` | `delay_spread`, `angular_spread` | `Try 36` rerun on `CKM_Dataset_270326.h5` |
| 41 | `TFGFortyFirstTry41` | `path_loss` | `physical_prior + learned_residual` on `CKM_Dataset_270326.h5`, anchored to the hybrid formula map |
| 42 | `TFGFortySecondTry42` | `path_loss` | PMNet-inspired residual regressor over a calibrated physical prior |
| 43 | `TFGFortyThirdTry43` | `path_loss` | PMNet control branch without physical prior |
| 44 | `TFGFortyFourthTry44` | `path_loss` | more faithful PMNet-v3-style control branch without physical prior |
| 45 | `TFGFortyFifthTry45` | `path_loss` | `Try 42` + stronger train-only NLoS-aware prior + spatial MoE residual head |

## Main family transitions

## Tries 1-3: make the pipeline exist

The first three tries were about getting a working HDF5-based image-to-image pipeline for:

- `path_loss`
- `delay_spread`
- `angular_spread`

The main question at this stage was basic functionality and baseline stability, not yet strong physical specialization.

## Tries 4-14: path-loss specialization and physical conditioning

This phase progressively moved the project toward `path_loss` as the primary target.

Key additions across this stage:

- confidence-based hybrid path-loss prediction,
- larger and more stable HDF5 setup,
- pure-regression path-loss baselines,
- antenna-height conditioning,
- LoS / NLoS separation,
- FiLM-based scalar conditioning.

By `Try 14`, the project had a strong and stable path-loss-oriented base.

## Tries 15-19: city-regime consolidation

This family acted as a transition stage:

- cleaner dataset handling,
- conservative geometry-aware tuning,
- and a stronger baseline before opening more targeted architectural and loss-based branches.

The main value of this family was not radical novelty, but a cleaner reference point for future comparisons.

## Tries 20-22: establish the strongest modern path-loss base

- `Try 20` isolated the decoder hypothesis:
  - replace transposed convolutions with bilinear upsampling plus convolution.

- `Try 21` isolated the supervision hypothesis:
  - add multiscale path-loss loss.

- `Try 22` combined both.

This combination became the strongest clean recent path-loss branch and the base for later path-loss experiments.

## Tries 23 and 26: reopen and refine the spread branch

- `Try 23` transferred the structural improvements of `Try 22` to:
  - `delay_spread`
  - `angular_spread`

- `Try 26` added a gradient-aware spread loss because the outputs still looked too flat and blob-like.

These tries clarified that the spread branch needed not only a structural decoder/supervision update, but also better protection of local transitions.

## Tries 25, 27 and 28: extra path-loss hypotheses

- `Try 25` asked whether more global context was still missing, using lightweight bottleneck attention.
- `Try 27` asked whether path-loss errors should be weighted more strongly near topology edges and urban transitions.
- `Try 28` tested whether those two ideas were complementary when combined.

These branches were informative even when they did not clearly surpass `Try 22`, because they helped identify what the dominant remaining bottleneck was not.

## Try 29: radial path-loss supervision

After reviewing 20 composite diagnostic panels, the main path-loss conclusion was:

- the model still underlearns the transmitter-centered radial structure of the field.

That is why `Try 29` returns to the stronger `Try 22` base and adds:

- radial profile loss,
- radial gradient loss.

This is a more physically targeted follow-up than simply increasing model complexity again.

## Try 30: spread amplitude protection

The same visual review suggested that the spread branch often:

- gets the rough support approximately right,
- but underestimates sparse high-value responses.

That is why `Try 30` returns to the stronger `Try 26` spread base and adds:

- value-weighted spread regression loss,
- hotspot-focused spread loss.

This directly targets the amplitude imbalance visible in the `delay_spread` and `angular_spread` predictions.

## Try 31: physical prior + learned residual for path loss

`Try 31` is the first major formulation change for `path_loss`.

Instead of asking the model to predict the full map from scratch, it uses:

- a simple physical prior based on propagation distance and carrier frequency,
- plus a learned residual correction predicted by the network.

The idea is that the network should not have to rediscover the entire radial propagation law by itself. It should focus on learning how buildings, topology, and environment details deviate from that simpler baseline.

So far, the idea is conceptually strong but has not yet surpassed `Try 22`.

## Try 32: support + amplitude spread prediction

`Try 32` is the corresponding formulation change for:

- `delay_spread`
- `angular_spread`

Instead of directly predicting only the final value map, the network predicts:

- a support map telling where the response should exist,
- and an amplitude map telling how strong it should be there.

The final prediction is built from both. This is meant to help with the recurring spread failure mode where the approximate location is learned but the magnitude is underestimated.

So far, this idea is also promising in principle but has not yet surpassed `Try 26`.

## Tries 33-36: building-mask exclusion and new physical-prior branch

After reviewing the data interpretation more carefully, a simpler rule was introduced:

- pixels where `topology_map != 0` should not be treated as valid receiver locations;
- they should therefore be excluded from loss and error computation.

This produced two different experiment directions:

- `Try 33`:
  - keep the strong `Try 22` path-loss recipe,
  - but ignore building pixels during supervision and evaluation.

- `Try 34`:
  - open a separate path-loss branch with an explicit formula input,
  - using a hybrid two-ray / COST231 prior,
  - while also applying the same building-mask exclusion.

- `Try 35`:
  - keep a lighter 1-GPU spread-side branch active,
  - while still respecting the building-mask exclusion.

- `Try 36`:
  - return to the clean `Try 26` spread baseline,
  - apply the same building-mask exclusion,
  - and run it as the clearer 2-GPU comparison branch.

## Tries 37-40: same masked family, new dataset

After obtaining the newer dataset:

- `CKM_Dataset_270326.h5`

the masked-supervision family was restarted so that the dataset version and the masking policy could be tested together in a cleaner way.

- `Try 37`:
  - reruns the `Try 33` idea on the new dataset
  - `path_loss`
  - building-mask exclusion only

- `Try 38`:
  - reruns the `Try 34` idea on the new dataset
  - `path_loss`
  - hybrid `two_ray_ground + COST231` formula input
  - building-mask exclusion
  - larger than the tiny debug-sized `Try 34`

- `Try 39`:
  - reruns the lighter masked spread branch on the new dataset
  - `delay_spread`
  - `angular_spread`

- `Try 40`:
  - reruns the cleaner 2-GPU masked spread baseline on the new dataset
  - `delay_spread`
  - `angular_spread`

This restart also formalized a practical rule that was explicitly rechecked in code:

- building pixels are not only hidden in plots;
- they are removed from supervision;
- they are removed from metric accumulation;
- and they are exported as invalid (`NaN`) in error-map style visualizations.

## Try 41: make prior + residual the main path-loss formulation

The newer dataset appears harder than the old one, and the clean masked rerun (`Try 37`) dropped much more than expected.

That pushed the project toward a stronger formulation change:

- do not ask the model to predict the full `path_loss` map from scratch;
- instead ask it to predict a correction on top of a physically motivated prior.

`Try 41` therefore uses:

- `prediction = physical_prior + learned_residual`

with:

- the hybrid `two_ray_ground + COST231` formula map used as an explicit input channel,
- the same formula map also reused as the prior anchor for the residual target,
- building-mask exclusion still active.

The goal is to stop spending model capacity on rediscovering the basic radial propagation carrier and instead focus that capacity on the residual urban corrections.

An additional leakage-safe prior-only analysis was also added for this try:

- fit calibration on `train`,
- evaluate only on `val`,
- and score only on ground pixels where `topology == 0`.

That analysis showed:

- the raw prior is much too weak on its own (`~67.23 dB` RMSE on validation),
- but train-only regime-aware calibration improves it a lot,
- with the best prior-only variant so far reaching about `24.16 dB` on validation,
- using a quadratic calibration split by city type, LoS/NLoS, and antenna-height tertile.

This is still not enough to justify replacing the network with a hand-calibrated prior, but it does justify keeping the prior as a central scaffold for residual learning.

The rerun of `Try 41` therefore upgrades the prior from:

- raw hybrid formula input

to:

- train-only regime-aware quadratic calibration on top of that hybrid formula,
- split by city type, LoS/NLoS, and antenna-height tertile,
- still scored only on ground pixels where `topology == 0`.

The stored calibration and system description are documented in:

- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`

## Try 42: replace the U-Net/cGAN family with a PMNet-style residual regressor

`Try 42` is the first path-loss branch in this stage that changes the backbone family rather than just the prior usage or the loss balance.

It keeps the same calibrated physical prior used in `Try 41`, but changes the learning problem and the network:

- keep `prediction = calibrated_prior + learned_residual`;
- remove the discriminator path entirely;
- stop using the U-Net backbone;
- use a PMNet-inspired residual encoder plus dilated context module instead.

This decision is motivated by the local `TFG_Proto1` paper review, especially:

- `TFG_Proto1/docs/markdown/2402.00878v1 (2)/2402.00878v1 (2).md`

where PMNet is described as a stronger alternative than plain RadioUNet when longer-range propagation relationships matter.

`Try 42` also expands validation beyond the single global RMSE and now reports:

- global path-loss RMSE;
- global prior-only RMSE;
- RMSE by LoS and NLoS;
- RMSE by city type;
- RMSE by antenna-height bin;
- RMSE by the combined calibration regime.

So `Try 42` is not just “another try”.

It is the first branch in this family that tests whether a different path-loss backbone can exploit the calibrated prior better than the old U-Net/cGAN line.

The first received `Try 42` result showed an important pattern:

- the global RMSE remained poor (`~23.19 dB` at epoch 1),
- but the regime breakdown was highly asymmetric:
  - `LoS` already near `4.36 dB`
  - `NLoS` still around `40.46 dB`

This means the remaining difficulty is concentrated in urban correction regimes, not in the easy physical carrier itself.

## Try 43: PMNet control branch without prior

`Try 43` exists to answer a very simple control question:

- if the calibrated prior is removed entirely, how much of `Try 42` was due to the prior and how much was due to PMNet itself?

So `Try 43` keeps:

- PMNet-style path-loss regression,
- building-mask exclusion,
- LoS input,
- distance-map input,
- antenna-height conditioning.

But it removes:

- the physical prior input,
- the prior-residual decomposition,
- and the prior-only reporting block.

It predicts `path_loss` directly.

Even without the prior, it still reports the same regime-level metrics as `Try 42`, so failure can still be localized by:

- `LoS / NLoS`
- city type
- antenna-height bin
- combined calibration-style regimes

## Try 44: more faithful PMNet-v3-style control branch

`Try 44` is not just another PMNet control rerun.

It was opened because the first PMNet-inspired branch (`Try 42`) did not clearly outperform the prior-guided U-Net family, and it was still only PMNet-inspired rather than close to the official repository.

So `Try 44` keeps the same no-prior comparison setup as `Try 43`, but changes the backbone again:

- a more faithful PMNet-v3-style encoder,
- ASPP context aggregation closer to the original PMNet code,
- a decoder path closer to the original PMNet repository than the lightweight FPN-style fusion used in `Try 42`.

This makes `Try 44` the cleaner test of:

- whether PMNet itself is a better path-loss architecture for this project,
- independent of the physical prior.

## Practical rule for future tries

- If a change is mainly a data fix, execution fix, or cheap conditioning addition, it can be reused across a family.
- If a change alters the decoder, the loss, the target definition, the routing, or the main architecture, it should become a new try.

## Related documents

- `SUPERVISOR_SUMMARY_TRIES_20_TO_32.md`
- `PAPER_SOURCES_TRIES_20_TO_32.md`
- `TRY29_VISUAL_REVIEW_AND_RADIAL_PLAN.md`
- `TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md`
- `PATH_LOSS_PRIORITY_NEXT_STEPS.md`
- `TRIES_33_TO_36_PHYSICAL_PRIORS_AND_BUILDING_MASK.md`
- `TRIES_37_TO_40_NEW_DATASET_AND_MASK_VERIFICATION.md`
- `TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md`
- `analysis/formula_prior_generalization_try41.md`
- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`
- `FOR_GENIA_SUMMARY_TRIES_20_TO_42.md`
- `PAPER_SOURCES_TRIES_20_TO_42.md`
- `TRY42_SOURCES_AND_PMNET_SCHEMA.md`
- `TRY42_PMNET_RESIDUAL_ARCHITECTURE.md`
- `FOR_GENIA_SUMMARY_TRIES_20_TO_44.md`
- `PAPER_SOURCES_TRIES_20_TO_44.md`
- `TRY43_TRY44_PMNET_CONTROLS.md`
