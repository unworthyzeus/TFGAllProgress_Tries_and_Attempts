# Sources and Technical Motivation for Tries 20-32

This document explains where the ideas behind `Try 20` to `Try 32` come from and why they were reasonable to test in this TFG.

Important clarification:

- these tries are **not** exact reproductions of single papers;
- they are practical adaptations to this codebase, this dataset, and the failure modes observed in the exported predictions;
- the correct wording is therefore "inspired by", "motivated by", or "adapted from the intuition of".

## Quick summary table

| Try | Main change | Main inspiration |
|---|---|---|
| 20 | Bilinear decoder for `path_loss` | Odena et al. 2016 |
| 21 | Multiscale loss for `path_loss` | Multiscale supervision literature, RadioFormer 2025, RMTransformer 2025 |
| 22 | Bilinear decoder + multiscale path-loss loss | Odena 2016 + multiscale radio-map literature |
| 23 | Extend that recipe to `delay_spread` and `angular_spread` | Same ideas as Try 22, adapted to continuous regression targets |
| 24 | Prepared multitask `path_loss + delay + angular` branch | Multitask learning literature |
| 25 | Lightweight bottleneck attention | RadioNet 2021, RMTransformer 2025, TransPathNet 2025 |
| 26 | Gradient-aware spread loss | Gradient-difference and edge-aware losses for continuous maps |
| 27 | Topology-edge-weighted `path_loss` regularization | RadioUNet 2019, RadioDUN 2025 |
| 28 | Combine Try 25 and Try 27 | Transformer-style context + physics-aware local emphasis |
| 29 | Radial profile loss + radial gradient loss for `path_loss` | Path-loss propagation intuition + direct visual review |
| 30 | Value-weighted spread loss + hotspot-focused spread loss | Imbalance-aware dense regression intuition + direct visual review |
| 31 | Physical prior + learned residual for `path_loss` | Model-based residual learning, propagation-law prior intuition |
| 32 | Support + amplitude formulation for spread targets | Decomposed prediction for sparse structured maps |

## Try 20

### What was implemented

- replace `ConvTranspose2d` with bilinear upsampling plus convolution;
- keep the rest of the `path_loss` recipe as close as possible to the strongest previous baseline.

### Main source

1. Odena, Dumoulin, Olah, **"Deconvolution and Checkerboard Artifacts"** (Distill, 2016)  
   Link: [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)

### Why it made sense here

- checkerboard-like artifacts had been observed in the path-loss predictions;
- this made the decoder a plausible bottleneck;
- `Try 20` was the cleanest way to isolate that single hypothesis.

### Why it might fail

- reducing decoder artifacts does not automatically fix poor global field organization;
- the model can look cleaner while still missing important propagation structure.

## Try 21

### What was implemented

- keep the task focused on `path_loss`;
- add multiscale supervision so the model is penalized not only at full resolution but also at coarser spatial scales.

### Main sources

1. Fang et al., **"RadioFormer: A Multiple-Granularity Radio Map Estimation Transformer with 1/10000 Spatial Sampling"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2504.19161](https://arxiv.org/abs/2504.19161)

2. Zhang et al., **"RMTransformer: Accurate Radio Map Construction and Coverage Prediction"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2501.05190](https://arxiv.org/abs/2501.05190)

3. General multiscale supervision literature in image-to-image regression

### Why it made sense here

- some path-loss errors were global, not only local;
- the model seemed to need more pressure to respect large-scale field organization;
- `Try 21` tested whether the missing ingredient was supervision rather than architecture.

### Why it might fail

- multiscale loss can improve coarse structure while leaving visible local artifacts untouched;
- if the decoder itself is the main issue, this alone is not enough.

## Try 22

### What was implemented

- bilinear decoder;
- multiscale path-loss loss;
- `path_loss` as the only target.

### Main sources

1. Odena et al. 2016  
   Link: [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)

2. Fang et al. 2025  
   Link: [https://arxiv.org/abs/2504.19161](https://arxiv.org/abs/2504.19161)

3. Zhang et al. 2025  
   Link: [https://arxiv.org/abs/2501.05190](https://arxiv.org/abs/2501.05190)

### Why it made sense here

- the predictions simultaneously suggested decoder artifacts and weak field coherence;
- combining both fixes was the most natural next step.

### Why it might fail

- if the real limitation comes from missing physical bias or missing inputs, decoder and multiscale improvements can still plateau.

## Try 23

### What was implemented

- transfer the `Try 22` recipe to `delay_spread` and `angular_spread`;
- keep bilinear decoding;
- generalize multiscale supervision to continuous regression targets.

### Main sources

1. Same structural sources as `Try 22`

2. Kendall, Gal, Cipolla, **"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"** (CVPR, 2018)  
   Link: [https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)

### Why it made sense here

- the spread outputs also looked oversmoothed and structurally weak;
- it was sensible to test whether the strongest recent path-loss recipe transferred to them.

### Why it might fail

- `delay_spread` and `angular_spread` are not simply "path loss with different units";
- they may need more specialized supervision.

## Try 24

### What was implemented

- prepare a multitask branch with:
  - `path_loss`
  - `delay_spread`
  - `angular_spread`
- keep it local only.

### Main sources

1. Kendall et al. 2018  
   Link: [https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)

2. Chen et al., **"GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"** (ICML, 2018)  
   Link: [https://arxiv.org/abs/1711.02257](https://arxiv.org/abs/1711.02257)

### Why it made sense here

- all three outputs share environment geometry and propagation context;
- a shared encoder might help.

### Why it might fail

- `path_loss` can dominate optimization and hide whether the spread branch is really improving.

## Try 25

### What was implemented

- start from `Try 22`;
- add lightweight attention in the bottleneck.

### Main sources

1. Tian et al., **"RadioNet: Transformer based Radio Map Prediction Model For Dense Urban Environments"** (arXiv, 2021)  
   Link: [https://arxiv.org/abs/2105.07158](https://arxiv.org/abs/2105.07158)

2. Zhang et al. 2025  
   Link: [https://arxiv.org/abs/2501.05190](https://arxiv.org/abs/2501.05190)

3. Li et al., **"TransPathNet: A Novel Two-Stage Framework for Indoor Radio Map Prediction"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2501.16023](https://arxiv.org/abs/2501.16023)

### Why it made sense here

- after improving decoder behavior and coarse structure, the next plausible bottleneck was long-range context.

### Why it might fail

- more context does not help if the missing bias is physical rather than architectural.

## Try 26

### What was implemented

- start from `Try 23`;
- add gradient-aware regression loss for `delay_spread` and `angular_spread`.

### Main sources

1. Mathieu, Couprie, LeCun, **"Deep Multi-Scale Video Prediction Beyond Mean Square Error"** (ICLR, 2016)  
   Link: [https://arxiv.org/abs/1511.05440](https://arxiv.org/abs/1511.05440)

2. Talker et al., **"Mind The Edge"** (CVPR, 2024)  
   Link: [https://openaccess.thecvf.com/content/CVPR2024/html/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Talker_Mind_The_Edge_Refining_Depth_Edges_in_Sparsely-Supervised_Monocular_Depth_CVPR_2024_paper.html)

3. Paul et al., **"Edge loss functions for deep-learning depth-map"** (2022)  
   Link: [https://www.sciencedirect.com/science/article/pii/S2666827021001092](https://www.sciencedirect.com/science/article/pii/S2666827021001092)

### Why it made sense here

- the spread maps often looked too flat and too blob-like;
- a gradient loss penalizes exactly that failure mode.

### Why it might fail

- it can improve structure while still leaving strong responses too weak;
- it does not directly address amplitude imbalance.

## Try 27

### What was implemented

- start from `Try 22`;
- add a `path_loss` regularization weighted by topology edges.

### Main sources

1. Levie et al., **"RadioUNet"** (arXiv, 2019)  
   Link: [https://arxiv.org/abs/1911.09002](https://arxiv.org/abs/1911.09002)

2. Chen et al., **"RadioDUN"** (arXiv, 2025)  
   Link: [https://arxiv.org/abs/2506.08418](https://arxiv.org/abs/2506.08418)

### Why it made sense here

- many path-loss errors looked concentrated around obstacle-sensitive regions and urban transitions.

### Why it might fail

- topology edges are only a proxy for physically relevant discontinuities;
- overemphasizing them can bias the optimization away from the full map.

## Try 28

### What was implemented

- combine `Try 25` and `Try 27`;
- keep attention plus topology-edge regularization.

### Main sources

- same transformer-inspired sources as `Try 25`;
- same physics-aware sources as `Try 27`.

### Why it made sense here

- the two ideas target different weaknesses:
  - context,
  - and physically delicate local regions.

### Why it might fail

- helpful ideas can still interfere when combined;
- the resulting optimization can become more constrained without solving the dominant bottleneck.

## Try 29

### What was implemented

- start from `Try 22`;
- add radial profile loss;
- add radial gradient loss.

### Main sources

1. Physical propagation intuition from path-loss structure

2. Direct manual visual review of 20 composite diagnostic panels

### Why it made sense here

- the review showed that the model still underlearned the transmitter-centered radial carrier pattern;
- this suggested that a physically targeted radial supervision term was more appropriate than simply adding complexity again.

### Why it might fail

- radial auxiliary losses help only if the implementation of the prior radial structure aligns well with the data;
- they can also regularize the wrong aspect if the remaining error is not purely radial.

## Try 30

### What was implemented

- start from `Try 26`;
- add value-weighted spread regression loss;
- add hotspot-focused spread loss.

### Main sources

1. Imbalance-aware dense regression intuition

2. Direct manual visual review of 20 composite diagnostic panels

### Why it made sense here

- the review showed that strong spread responses were often underestimated and averaged away;
- this motivated losses that explicitly protect high-value regions.

### Why it might fail

- overemphasizing peaks can destabilize optimization;
- it can also make the model overfocus on rare regions while worsening average structure.

## Try 31

### What was implemented

- start from the stronger `Try 22` path-loss base;
- compute a simple physical path-loss prior from distance and carrier frequency;
- let the network predict a learned residual correction on top of that prior.

### Main sources

1. General residual-learning literature in model-based prediction

2. The path-loss propagation intuition that a large part of the field should already be explainable by distance-based attenuation

3. Direct visual evidence that the network was underlearning the radial carrier structure

### Why it made sense here

- the previous tries suggested that the network was using capacity to reconstruct a pattern that should partly come from a simple prior;
- residual learning is a natural way to say: let physics explain the easy part and let the network learn the deviations.

### Why it might fail

- if the prior is too crude or mismatched, the residual branch may spend capacity compensating for prior errors;
- it can also make optimization harder if the target decomposition is not well balanced.

## Try 32

### What was implemented

- keep the stronger `Try 26` spread base;
- predict a support map telling where the response should exist;
- predict an amplitude map telling how strong the response should be there;
- combine both to build the final spread prediction.

### Main sources

1. General decomposed-prediction intuition for sparse structured targets

2. The observation that spread predictions often get approximate location right but underestimate the amplitude

### Why it made sense here

- a single continuous output map may mix two different subproblems:
  - locating active regions,
  - and estimating their magnitude;
- splitting them can make the learning problem clearer.

### Why it might fail

- the support target is itself heuristic and may introduce noise;
- if support prediction is unstable, the final value map can become worse even when the decomposition is conceptually sound.
