# Experiment Versions (Try 1 -> Try 32)

This document summarizes the evolution from `TFG_FirstTry1` to `TFGThirtySecondTry32`, what each family predicts, and why each new branch was opened.

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
- `Try 31` is the new path-loss paradigm-shift branch:
  - physical prior + learned residual
- `Try 32` is the new spread paradigm-shift branch:
  - support + amplitude prediction
- Current cluster runs on 2026-03-26:
  - `Try 31` job `10014190`, `2 GPU`, `2 days`, `RUNNING`
  - `Try 32` job `10014189`, `2 GPU`, `2 days`, `RUNNING`

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

## Practical rule for future tries

- If a change is mainly a data fix, execution fix, or cheap conditioning addition, it can be reused across a family.
- If a change alters the decoder, the loss, the target definition, the routing, or the main architecture, it should become a new try.

## Related documents

- `SUPERVISOR_SUMMARY_TRIES_20_TO_32.md`
- `PAPER_SOURCES_TRIES_20_TO_32.md`
- `TRY29_VISUAL_REVIEW_AND_RADIAL_PLAN.md`
- `TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md`
- `PATH_LOSS_PRIORITY_NEXT_STEPS.md`
