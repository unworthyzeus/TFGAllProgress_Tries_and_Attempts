# Sources and Technical Motivation for Tries 20-42

This document extends the earlier sources summary and records the technical motivation behind the full recent chain from `Try 20` to `Try 42`.

Important clarification:

- these tries are not exact reproductions of single papers;
- they are practical adaptations to this codebase, these datasets, and the failure modes observed in the exported predictions;
- the correct wording is therefore:
  - inspired by,
  - motivated by,
  - or adapted from the intuition of.

## High-level phases

The recent tries fall into four phases:

1. decoder and supervision cleanup:
   - `20-23`
2. targeted architectural or loss additions:
   - `24-30`
3. formulation change toward physically guided residuals:
   - `31-41`
4. path-loss backbone replacement:
   - `42`

## Summary table

| Try | Main change | Main source or motivation |
|---|---|---|
| 20 | Bilinear decoder for `path_loss` | Odena et al. 2016 |
| 21 | Multiscale loss for `path_loss` | Multiscale supervision and radio-map transformer literature |
| 22 | Bilinear decoder + multiscale path-loss loss | Odena 2016 + multiscale radio-map literature |
| 23 | Extend the same idea to spreads | Same logic as Try 22, adapted to regression targets |
| 24 | Prepared multitask branch | Multitask learning literature |
| 25 | Lightweight bottleneck attention | RadioNet 2021, RMTransformer 2025, TransPathNet 2025 |
| 26 | Gradient-aware spread loss | Gradient-difference and edge-aware regression losses |
| 27 | Topology-edge path-loss regularization | RadioUNet 2019, RadioDUN 2025 |
| 28 | Combine 25 and 27 | Context + local physics emphasis |
| 29 | Radial profile and radial gradient losses | Path-loss propagation intuition + direct visual review |
| 30 | Value-weighted and hotspot-focused spread losses | Imbalance-aware dense regression intuition |
| 31 | Physical prior + learned residual | Residual learning over model-based priors |
| 32 | Support + amplitude spread prediction | Decomposed prediction for sparse maps |
| 33 | Building-mask path-loss supervision | Correct target-domain definition |
| 34 | Formula-input path-loss with building-mask | Physical prior + corrected supervision mask |
| 35 | Spread branch with building-mask | Correct target-domain definition for spread outputs |
| 36 | Clean spread baseline with building-mask | Cleaner comparison under corrected masking |
| 37 | Re-run masked path-loss on newer dataset | Dataset shift / transfer check |
| 38 | Re-run formula-input path-loss on newer dataset | Dataset shift + physics-guided robustness |
| 39 | Re-run lighter masked spread branch on newer dataset | Dataset shift check for spreads |
| 40 | Re-run stronger masked spread branch on newer dataset | Cleaner spread comparison on new dataset |
| 41 | Calibrated prior + learned residual | Regime-aware prior calibration + residual learning |
| 42 | PMNet-style residual regressor over calibrated prior | PMNet, long-range context, residual learning |

## Key source families

### Decoder artifacts and image-to-image supervision

1. Odena, Dumoulin, Olah, **"Deconvolution and Checkerboard Artifacts"** (Distill, 2016)  
   Link: [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)

2. Mathieu, Couprie, LeCun, **"Deep Multi-Scale Video Prediction Beyond Mean Square Error"** (ICLR, 2016)  
   Link: [https://arxiv.org/abs/1511.05440](https://arxiv.org/abs/1511.05440)

### Radio-map prediction and long-range context

3. Levie et al., **"RadioUNet: Fast Radio Map Estimation With Convolutional Neural Networks"** (IEEE TWC, 2021 / arXiv 2019)  
   Link: [https://arxiv.org/abs/1911.09002](https://arxiv.org/abs/1911.09002)

4. Tian et al., **"RadioNet: Transformer based Radio Map Prediction Model For Dense Urban Environments"** (arXiv, 2021)  
   Link: [https://arxiv.org/abs/2105.07158](https://arxiv.org/abs/2105.07158)

5. Zhang et al., **"RMTransformer: Accurate Radio Map Construction and Coverage Prediction"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2501.05190](https://arxiv.org/abs/2501.05190)

6. Li et al., **"TransPathNet: A Novel Two-Stage Framework for Indoor Radio Map Prediction"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2501.16023](https://arxiv.org/abs/2501.16023)

7. Fang et al., **"RadioFormer: A Multiple-Granularity Radio Map Estimation Transformer with 1/10000 Spatial Sampling"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2504.19161](https://arxiv.org/abs/2504.19161)

8. Lee et al., **"PMNet: Robust Pathloss Map Prediction via Supervised Learning"** (arXiv, 2023)  
   Link: [https://arxiv.org/abs/2211.10527](https://arxiv.org/abs/2211.10527)

### Physics-aware or obstacle-aware guidance

9. Chen et al., **"RadioDUN"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2506.08418](https://arxiv.org/abs/2506.08418)

10. COST231-Hata family and two-ray ground propagation formulas  
    Motivation: lightweight analytical propagation scaffolds used as prior maps.

### Edge-aware and structure-aware dense regression

11. Talker et al., **"Mind The Edge"** (CVPR, 2024)  
    Link: [https://openaccess.thecvf.com/content/CVPR2024/html/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.html)

12. Paul et al., **"Edge loss functions for deep-learning depth-map"** (2022)  
    Link: [https://www.sciencedirect.com/science/article/pii/S2666827021001092](https://www.sciencedirect.com/science/article/pii/S2666827021001092)

### Multitask and decomposed learning

13. Kendall, Gal, Cipolla, **"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"** (CVPR, 2018)  
    Link: [https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)

14. Chen et al., **"GradNorm"** (ICML, 2018)  
    Link: [https://arxiv.org/abs/1711.02257](https://arxiv.org/abs/1711.02257)

## Where the calibrated prior came from

The calibrated prior used in `Try 41` and kept in `Try 42` is not taken from one single paper. It is a project-specific system combining:

1. a raw physical path-loss scaffold:
   - hybrid `two_ray_ground + COST231`
2. a train-only calibration strategy:
   - split by `city type`, `LoS/NLoS`, and `antenna-height tertile`
3. a quadratic correction per regime.

This was motivated by three things:

- the raw analytical prior alone was far too inaccurate;
- visual inspection showed the model was still underlearning the radial carrier structure;
- the newer dataset required stronger physical bias to stabilize transfer.

The relevant project documents are:

- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`
- `TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md`

So the prior system is best described as:

- model-based propagation prior
- plus train-only regime-aware empirical calibration

rather than as a direct reproduction of one specific paper.

## Why Try 42 is the natural next step

By the time `Try 41` was tested, the project had learned two important things:

1. the calibrated prior is useful and should remain in the system;
2. the old U-Net/cGAN family does not seem to exploit that prior strongly enough.

That is why `Try 42` is motivated by PMNet-style ideas:

- stronger long-range context than a plain U-Net;
- residual encoder backbone;
- dilated context aggregation;
- direct regression instead of adversarial image translation.

This is the first recent path-loss branch whose main hypothesis is:

- the next dominant bottleneck is the backbone, not only the prior or the loss.
