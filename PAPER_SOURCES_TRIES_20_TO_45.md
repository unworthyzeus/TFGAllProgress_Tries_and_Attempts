# Paper Sources for Tries 20 to 46

This note tracks the main source ideas behind the recent tries and states clearly when a branch is only *inspired by* a paper rather than a faithful reproduction.

## Tries 20 to 22

Main source:

- Odena et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016

Why it matters:

- `Try 20` replaced transposed-convolution-heavy decoding with bilinear upsampling
- `Try 21` isolated multiscale supervision
- `Try 22` combined both

These were not paper reproductions, but they were motivated by known decoder-artifact behavior and multiscale supervision ideas.

## Try 41

Main idea source:

- physical-prior + residual-learning logic from model-based learning practice in inverse problems and scientific ML

What was adapted:

- use a deterministic propagation prior as the base prediction
- learn only the correction

## Try 42

Main source family:

- PMNet repository and PMNet-style radio-map prediction architecture

Repository:

- [PMNet GitHub repository](https://github.com/abman23/pmnet)

What was adapted:

- residual encoder
- multi-scale context aggregation
- FPN-like fusion
- direct residual prediction over a calibrated path-loss prior

This was a PMNet-inspired branch, not a line-by-line reproduction.

## Try 43

Purpose:

- no-prior PMNet control branch

This was not motivated by a separate paper, but by experimental logic:

- isolate whether the PMNet family is helping on its own

## Try 44

Main source:

- the official PMNet repository structure more directly than `Try 42`

What was changed:

- more faithful encoder/context/decoder organization
- still no prior

Purpose:

- isolate whether implementation fidelity, rather than the prior, was the missing factor

## Try 45

`Try 45` combines multiple source lines.

### 1. Deterministic propagation base

Source families:

- free-space path loss
- two-ray ground reflection
- COST231-Hata urban macro model

These provide the deterministic base map.

### 2. A2G LoS/NLoS excess-loss model

Main source:

- [2511.15412v1.md](C:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2511.15412v1/2511.15412v1.md)

Most relevant equations:

- Eq. (1): total large-scale loss decomposition
- Eq. (9): reference attenuation term
- Eq. (10): LoS excess-loss term
- Eq. (11): NLoS excess-loss term as a function of elevation angle
- Eq. (12): shadow-fading standard-deviation model

How it is used:

- not as a full standalone simulator
- but as an extra deterministic ingredient to strengthen the `NLoS` prior branch
- and, for `Eq. 12`, as an uncertainty-style side feature rather than direct additive mean path loss

Important nuance:

- this is not a verbatim reproduction of the paper
- it keeps the physical structure of the paper relevant to `NLoS`
- and then applies train-only calibration on top

### 3. LOS-map / shadow-region reasoning

Same paper, especially:

- Eq. (5): LOS map from the shadowed outdoor area
- Eq. (6): LOS/NLoS route segmentation

How it is used:

- the calibration adds local shadow-support proxies derived from the LOS map
- this is meant to help the prior know whether a pixel is only slightly blocked or deeply inside a shadowed region

### 4. Train-only calibration of compact physical parameters

This part is not taken from one single paper. It follows a general scientific-ML principle:

- keep the physical form compact and interpretable,
- fit only a small number of coefficients on `train`,
- choose the variant on `validation`,
- and keep `test` untouched for final reporting.

In `Try 45`, that means:

- the physical family is preserved,
- but some coefficients and regime mappings are allowed to adapt train-only,
- with explicit calibration artifacts saved as part of the experiment.
- a small parameter search is also allowed for a few global A2G coefficients, but only with:
  - search on `train`
  - comparison on `validation`
  - and `test` untouched for final reporting

### 5. Mixture-of-experts idea

This is not copied from a specific `TFG_Proto1` paper.

It is used here as an experimental engineering response to the regime diagnostics:

- `LoS` is relatively easy
- `NLoS` remains much harder
- lower antennas and denser city types are harder still

So the residual branch is specialized through a lightweight spatial MoE head rather than a single residual head.

## Current project conclusion

The current evidence says:

- PMNet without prior is not enough
- PMNet with prior is better
- but the dominant unresolved error is still `NLoS`

That is why `Try 45` is currently gated by prior quality:

- it should only be launched when the prior-only `NLoS` validation RMSE drops below `20 dB`

## Try 46

`Try 46` combines two source lines.

### 1. Explicit LoS/NLoS segmentation logic

- [Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections](https://arxiv.org/abs/2511.15412)
- [Spatially Consistent Air-to-Ground Channel Modeling with Probabilistic LOS/NLOS Segmentation](https://arxiv.org/abs/2506.12794)

This motivates the idea that:

- `LoS` and `NLoS` should be structurally distinct inside the system
- and visibility/shadow information should be used directly, not only as an implicit cue

### 2. Expert specialization for heterogeneous dense prediction

- [Multi-Task Dense Prediction via Mixture of Low-Rank Experts (MLoRE)](https://arxiv.org/abs/2403.17749)
- [Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution](https://arxiv.org/abs/2502.09654)

These motivate the engineering choice that:

- the `NLoS` branch should itself contain multiple experts,
- because the difficult regimes inside `NLoS` are likely heterogeneous.

## Important honesty note

For supervisor communication, the safest wording is:

- "inspired by"
- "motivated by"
- "adapted from"

instead of claiming exact reproduction unless the implementation is truly paper-faithful.
