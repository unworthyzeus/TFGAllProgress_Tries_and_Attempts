# Sources and Technical Motivation for Tries 20-44

This document extends the earlier source summary through the current PMNet control stage.

Important clarification:

- these tries are not exact reproductions of single papers;
- they are project-specific adaptations to this codebase, this dataset family, and the observed failure modes;
- the correct wording is:
  - inspired by,
  - motivated by,
  - or adapted from.

## High-level phases

The recent tries fall into five phases:

1. decoder and supervision cleanup:
   - `20-23`
2. targeted architecture and loss additions:
   - `24-30`
3. formulation change toward physically guided residuals:
   - `31-41`
4. first PMNet-style backbone replacement:
   - `42`
5. PMNet control and more faithful PMNet-v3-style comparison:
   - `43-44`

## Summary table

| Try | Main change | Main source or motivation |
|---|---|---|
| 20 | Bilinear decoder for `path_loss` | Odena et al. 2016 |
| 21 | Multiscale loss for `path_loss` | Multiscale supervision literature |
| 22 | Bilinear decoder + multiscale path-loss loss | Odena 2016 + multiscale radio-map literature |
| 23 | Extend the same structural idea to spreads | Same logic as Try 22, adapted to regression |
| 24 | Prepared multitask branch | Multitask learning literature |
| 25 | Lightweight bottleneck attention | RadioNet, RMTransformer, TransPathNet |
| 26 | Gradient-aware spread loss | Gradient-difference and edge-aware regression losses |
| 27 | Topology-edge path-loss regularization | RadioUNet, RadioDUN |
| 28 | Combine 25 and 27 | Context + local-physics emphasis |
| 29 | Radial profile and radial gradient losses | Propagation intuition + visual review |
| 30 | Value-weighted and hotspot-focused spread losses | Imbalance-aware dense regression intuition |
| 31 | Physical prior + learned residual | Residual learning over model-based priors |
| 32 | Support + amplitude spread prediction | Decomposed prediction for sparse maps |
| 33 | Building-mask path-loss supervision | Correct target-domain definition |
| 34 | Formula-input path-loss with building-mask | Physical prior + corrected supervision mask |
| 35 | Spread branch with building-mask | Correct target-domain definition for spreads |
| 36 | Clean spread baseline with building-mask | Cleaner comparison under corrected masking |
| 37 | Re-run masked path-loss on newer dataset | Dataset shift / transfer check |
| 38 | Re-run formula-input path-loss on newer dataset | Dataset shift + physics-guided robustness |
| 39 | Re-run lighter masked spread branch on newer dataset | Dataset shift check for spreads |
| 40 | Re-run stronger masked spread branch on newer dataset | Cleaner spread comparison on new dataset |
| 41 | Calibrated prior + learned residual | Regime-aware prior calibration + residual learning |
| 42 | PMNet-inspired residual regressor over calibrated prior | PMNet + long-range context + residual learning |
| 43 | PMNet direct control without prior | Architecture control / prior ablation |
| 44 | More faithful PMNet-v3-style direct control without prior | Official PMNet repository structure |

## Core source families

### Decoder artifacts and image-to-image supervision

1. Odena, Dumoulin, Olah, **"Deconvolution and Checkerboard Artifacts"** (Distill, 2016)  
   Link: [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)

2. Mathieu, Couprie, LeCun, **"Deep Multi-Scale Video Prediction Beyond Mean Square Error"** (ICLR, 2016)  
   Link: [https://arxiv.org/abs/1511.05440](https://arxiv.org/abs/1511.05440)

### Radio-map prediction and long-range context

3. Levie et al., **"RadioUNet: Fast Radio Map Estimation With Convolutional Neural Networks"**  
   Link: [https://arxiv.org/abs/1911.09002](https://arxiv.org/abs/1911.09002)

4. Tian et al., **"RadioNet: Transformer based Radio Map Prediction Model For Dense Urban Environments"**  
   Link: [https://arxiv.org/abs/2105.07158](https://arxiv.org/abs/2105.07158)

5. Zhang et al., **"RMTransformer: Accurate Radio Map Construction and Coverage Prediction"**  
   Link: [https://arxiv.org/abs/2501.05190](https://arxiv.org/abs/2501.05190)

6. Li et al., **"TransPathNet: A Novel Two-Stage Framework for Indoor Radio Map Prediction"**  
   Link: [https://arxiv.org/abs/2501.16023](https://arxiv.org/abs/2501.16023)

7. Lee et al., **"PMNet: Robust Pathloss Map Prediction via Supervised Learning"**  
   Link: [https://arxiv.org/abs/2211.10527](https://arxiv.org/abs/2211.10527)

8. Official PMNet code repository  
   Link: [https://github.com/abman23/large-scale-channel-prediction](https://github.com/abman23/large-scale-channel-prediction)

### Physics-aware guidance

9. COST231-Hata family and two-ray ground propagation formulas  
   Motivation: lightweight analytical propagation scaffolds used as prior maps.

10. Chen et al., **"RadioDUN"**  
    Link: [https://arxiv.org/abs/2506.08418](https://arxiv.org/abs/2506.08418)

### Edge-aware and structure-aware dense regression

11. Talker et al., **"Mind The Edge"**  
    Link: [https://openaccess.thecvf.com/content/CVPR2024/html/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.html)

12. Paul et al., **"Edge loss functions for deep-learning depth-map"**  
    Link: [https://www.sciencedirect.com/science/article/pii/S2666827021001092](https://www.sciencedirect.com/science/article/pii/S2666827021001092)

## Where the calibrated prior came from

The calibrated prior kept in `Try 41` and `Try 42` is not copied from one single paper.
It is a project-specific system built from:

1. a raw hybrid propagation map:
   - `two_ray_ground + COST231`
2. a train-only regime split:
   - `city type`
   - `LoS / NLoS`
   - `antenna-height tertile`
3. a quadratic empirical calibration inside each regime.

This came from project diagnosis rather than one paper:

- the raw analytical prior was far too inaccurate by itself;
- the model was visibly underlearning the basic radial carrier structure;
- the newer dataset appeared harder and benefited from stronger physical bias.

The relevant project documents are:

- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`
- `TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md`

## Why Try 42 exists

By the time `Try 41` was tested, the project had learned:

1. the calibrated prior is useful and should remain available;
2. the old U-Net/cGAN family still does not exploit it strongly enough.

That is why `Try 42` was opened:

- keep the calibrated prior,
- keep residual learning,
- but replace the old backbone with a PMNet-inspired one.

The first result did not immediately improve the global score, but it did reveal a useful pattern:

- `LoS` was already much easier,
- `NLoS` remained the dominant failure regime.

So `Try 42` was still informative, even without a clear win.

## Why Try 43 exists

`Try 43` removes the prior entirely while keeping the PMNet family.

Its purpose is methodological:

- to separate the effect of the prior
- from the effect of the PMNet-style backbone.

This is a standard and useful control:

- same task,
- same masking,
- same diagnostics,
- but no physical prior.

## Why Try 44 exists

`Try 44` was opened because `Try 42` was only PMNet-inspired, not close to the official PMNet-v3 implementation.

The official repository contains several structural ideas that our first PMNet branch simplified:

- bottleneck-style residual encoder;
- ASPP context block;
- a denser decoder path closer to the competition model.

So `Try 44` is motivated by the official PMNet code itself, not only by the paper-level description.

It is therefore the right next control if the question is:

- is PMNet failing because the family is wrong,
- or because the first PMNet branch was still too simplified?

## Is a future Try 45 with prior justified?

Yes, but only conditionally.

The clean logic is:

- first compare `Try 43` and `Try 44` as no-prior PMNet controls;
- if `Try 44` shows that the more faithful PMNet backbone is better than `Try 43`,
- then a future `Try 45 = Try 44 + calibrated prior` becomes well justified.

If `Try 44` does not improve, then adding the prior back would not answer the right question yet.
