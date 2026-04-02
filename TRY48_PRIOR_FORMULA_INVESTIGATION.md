# Try 48 Prior Formula Investigation

## Scope

This note summarizes what currently looks most promising while **building Try 48** to reduce **prior-only RMSE** and improve the physical realism of the `LoS` prior.

It is based on:

- the current repo implementation and calibration reports,
- the local `TFG_Proto1` paper markdown,
- and recent online papers from Vinogradov/Saboor plus related A2G sources.

## Main conclusion

The best next prior is **not**:

- another coefficient-only search,
- another smooth breakpoint two-ray variant,
- or another `NLoS = LoS-like formula + scalar excess-loss patch`.

The best next prior is:

1. a **damped coherent two-ray** law for `LoS`,
2. a **hard-segmented height-dependent NLoS log-distance** law for `NLoS`,
3. plus the existing train-only regime calibration on top.

In short:

$$
\Lambda_{\text{prior}} =
\begin{cases}
\text{Cal}_r\{\Lambda_{\text{LoS,coh-2ray-damped}}\}, & \text{LoS}\\
\text{Cal}_r\{\Lambda_{\text{NLoS,height-PLEx}} + \Delta_{\text{obs}}\}, & \text{NLoS}
\end{cases}
$$

where `r` is a regime such as `city_type x LoS/NLoS x antenna-height bin`.

## Exact Try 47 Port Note

After re-checking the local code, an important implementation mismatch showed up:

- `Try 48` was configured with `hybrid_two_ray_cost231_a2g_nlos`
- but `Try 48/data_utils.py` did not actually implement that exact mode name
- so the config could silently fall through to a different branch instead of reproducing the `Try 47` best prior

That has now been corrected in the `Try 48` codebase by porting the exact `hybrid_two_ray_cost231_a2g_nlos` branch from `Try 47`.

Also, `Try 48` now supports the same auxiliary obstruction-feature channels used in `Try 47`:

- shadow depth
- distance since LoS break
- max blocker height
- blocker count

When the precomputed HDF5 exists it is reused. If it does not exist, `Try 48` falls back to on-the-fly computation so local verification is still possible.

## Implemented in Try 48 code

The current implementation pass applied in the `Try 48` codebase now uses:

- formula mode: `hybrid_shadowed_ripple_two_ray_vinogradov_nlos`
- active config:
  - `TFGFortyEighthTry48/experiments/fortyeighthtry48_pmnet_prior_gan/fortyeighthtry48_pmnet_prior_gan.yaml`
- core implementation:
  - `TFGFortyEighthTry48/data_utils.py`

What was changed:

- `LoS` now keeps a **damped coherent two-ray envelope** plus an explicit **bounded ripple residual**, instead of relying only on a heavily smoothed two-ray trace
- `NLoS` now uses a **relaxed height-dependent Vinogradov/Saboor mean law**
- `NLoS` gets an additional **directional shadow-depth term** derived from:
  - resized topology,
  - local occupancy and local height proxy,
  - radial distance since the last `LoS` point along the same azimuth,
  - and a bounded survival of the coherent ripple inside shallow shadow
- the review/export script now explicitly excludes `topology > 0` pixels even if future dataset masks change

The goal of this implementation is to improve the prior structure before any new `Try 48` training launch.

## What the visual review changed

After exporting hard validation panels from `Try 48`, two failure modes became obvious:

1. the previous prior erased too much of the **concentric comb-filtering structure** visible in the target;
2. the previous `NLoS` branch applied a largely smooth penalty over whole blocked regions, making many maps too bright almost everywhere in `NLoS`.

That visual evidence changed the implementation priority:

- preserve a bounded oscillatory component from the coherent two-ray law,
- and drive `NLoS` severity more by **depth behind the LoS break** than by flat local occupancy alone.

## What the current repo says

### Best current prior-only result

The best current validation prior-only result still comes from the `Try 47` obstruction-aware calibrated system:

- overall RMSE: `23.5746 dB`
- LoS RMSE: `3.8138 dB`
- NLoS RMSE: `41.2686 dB`

So the project is still overwhelmingly bottlenecked by `NLoS`.

### Structural prior result that did not win

The newer structural `Try 48` prior-only calibration reached:

- best calibrated RMSE: `24.1777 dB`

So the first coherent-two-ray plus Vinogradov-style structural refresh is useful negative evidence, but not yet a better prior.

### Important implementation mismatch

There is a small but important mismatch between the `Try 48` notes and the active YAML:

- documentation discusses `hybrid_coherent_two_ray_vinogradov_nlos`
- active YAML uses `hybrid_fresnel_two_ray_cost231_a2g_nlos`

That should be resolved before the next formula comparison, otherwise results and notes can be compared against different priors by mistake.

## Formula ranking by expected impact on RMSE

### 1. Highest expected RMSE gain: replace the NLoS mean law

This is the highest priority.

The current active `Try 48` NLoS branch is still structurally weak because it is essentially:

- `max(COST231, angle-only A2G-NLoS)`

That means:

- no true height-dependent `NLoS` slope,
- no geometry-driven distance law beyond `COST231`,
- and no real separation between shallow-shadow and deep-shadow behavior.

The strongest next formula is therefore:

$$
\Lambda_{\text{NLoS}}(d,h) = \Lambda_0 + 10\,n_{\text{NLoS}}(h,c)\log_{10}(d/d_0) + b_c + \Delta_{\text{obs}}
$$

with

$$
n_{\text{NLoS}}(h,c) = n_{\infty,c} + (n_{0,c} - n_{\infty,c})\exp(-h/h_{0,c})
$$

and `c` a coarse city morphology class such as:

- `open_lowrise`
- `mixed_midrise`
- `dense_highrise`

Then add a deterministic obstruction term:

$$
\Delta_{\text{obs}} = \beta_1 \log(1 + d_{\text{since LoS break}}) + \beta_2 \,\text{shadow depth} + \beta_3 \,\text{blocker count} + \beta_4 \,\text{max blocker height}
$$

This is the closest match to:

- the Vinogradov/Saboor height-dependent PLE papers,
- your own `Try 47` observation that obstruction-aware features help,
- and the fact that `NLoS` remains the dominant RMSE term.

### 2. Second priority: use a better LoS coherent two-ray, but damp it properly

This is mainly for physical realism and for reducing what the network must invent.

The old `two_ray_ground` is only:

- `FSPL` before crossover,
- asymptotic `40 log10(d)` after crossover.

So it cannot create the concentric rings of a true interference model.

The best `LoS` family is therefore:

$$
\Lambda_{\text{LoS}} = -10\log_{10}\left[
\left(\frac{\lambda}{4\pi}\right)^2
\left|
\frac{e^{-jkd_{\text{LoS}}}}{d_{\text{LoS}}}
+
\Gamma(\theta)\,w(\theta,d)\,\frac{e^{-jkd_{\text{ref}}}}{d_{\text{ref}}}
\right|^2
\right]
$$

with:

- `dLoS` direct path length,
- `dref` reflected path length,
- `Gamma(theta)` Fresnel reflection coefficient,
- `w(theta,d)` a damping term.

The key is the damping term. A hard floor like:

- `max(coherent_db, fspl_direct_db - 6 dB)`

protects numerical stability but also suppresses part of the ring structure. A better choice is a smooth attenuation on the reflected component, for example:

$$
w(\theta,d) = \exp\!\left(-\left(\frac{4\pi \sigma_g \cos\theta}{\lambda}\right)^2\right)\cdot \exp(-d/\tau_d)
$$

or an equivalent smooth envelope.

This keeps the oscillatory structure while avoiding unrealistically deep notches.

For the current `Try 48` visual refinement, the practical engineering extension is:

- keep a smooth damped two-ray **envelope** for stability,
- but also preserve a **bounded residual ripple** relative to that envelope,
- and let some fraction of that ripple survive inside shallow shadow.

That last step is an inference from the visual review plus the source family below; it is not copied from a single paper verbatim.

### 3. Third priority: split NLoS into shallow-shadow and deep-shadow

This is likely better than adding more global coefficients.

Recent geometry-based A2G work suggests that `NLoS` is not one single regime. At minimum, the next prior should distinguish:

- recent LoS break / shallow shadow,
- persistent shadow / deep shadow.

So a practical version is:

$$
\Lambda_{\text{NLoS}} =
\begin{cases}
\Lambda_{\text{NLoS,height-PLEx}} + \Delta_{\text{mild}}, & \text{shallow shadow}\\
\Lambda_{\text{NLoS,height-PLEx}} + \Delta_{\text{strong}}, & \text{deep shadow}
\end{cases}
$$

using train-only thresholds on:

- distance since LoS break,
- blocker count,
- shadow depth,
- or local NLoS support.

This is cheaper than full ray tracing and much closer to the mechanisms described in the recent papers.

In the current code pass, this was specialized one step further into:

- a **radial shadow-depth proxy** computed from the provided `los_mask`,
- a milder obstruction term in shallow shadow,
- and extra loss only when shadow depth and blocker prominence both persist.

## What should probably be avoided

### Avoid spending more time on coefficient-only searches

The repo already shows that changing coefficients inside the old A2G excess-loss family does not solve the problem. The large gains must come from changing the **model family**, not from small numeric tuning.

### Avoid using the old asymptotic two-ray as the final LoS prior

It can give low `LoS` RMSE because it matches the broad radial slope, but it cannot reproduce the ring structure you actually see in the target maps.

### Avoid treating Vinogradov only as a soft PLoS blend

If you already have a pixelwise `los_mask`, the best use of Vinogradov is not mainly:

- replacing hard segmentation with a global probabilistic blend.

The best use is:

- taking the **height-dependent NLoS law**,
- and the **segmentation-first philosophy**.

In this project, hard segmentation is already available, so use it.

## Best next formula to test

If only one new prior family should be tested next, it should be:

### Candidate A

- `LoS`: damped coherent two-ray
- `NLoS`: height-dependent log-distance PLE
- `NLoS` support: obstruction-depth additive term
- calibration: keep train-only regime quadratic or obstruction-aware calibration

In pseudocode:

```python
if los_mask == 1:
    prior = cal_regime(los_damped_coherent_two_ray(d2d, h_tx, h_rx, f, eps_r, sigma_g, tau_d))
else:
    n_h = n_inf_city + (n0_city - n_inf_city) * exp(-h_tx / h0_city)
    mean_nlos = lambda0 + 10 * n_h * log10(d3d / d0)
    obs_term = (
        b0_city
        + b1 * log1p(distance_since_los_break)
        + b2 * shadow_depth
        + b3 * blocker_count
        + b4 * max_blocker_height
    )
    prior = cal_regime(mean_nlos + obs_term)
```

This is the most source-backed formula that still matches the project constraints.

## Practical recommendation for the next experiment

1. Freeze one baseline formula family in config and docs so they match.
2. Keep `Try 47` as the numeric baseline to beat: `23.5746 dB`.
3. Replace only the `NLoS` mean law first.
4. Re-run train-only calibration.
5. Only after that, refine the `LoS` two-ray damping to improve the ring structure.

If the goal is strictly **RMSE first**, step 3 matters more than step 5.

If the goal is **physical prior fidelity first**, step 5 matters more visually, but it is still less important numerically than `NLoS`.

## Sources used

### Online sources

- Ranchagoda et al., *Elevation-angle based two-ray path loss model for Air-to-Ground wireless channels*, Vehicular Communications, 2021.
  - https://www.sciencedirect.com/science/article/pii/S2214209621000620
- Vinogradov et al., *Spatially Consistent Air-to-Ground Channel Modeling with Probabilistic LOS/NLOS Segmentation*, arXiv:2506.12794.
  - https://arxiv.org/abs/2506.12794
- Saboor and Vinogradov, *Millimeter-Wave UAV Channel Model with Height-Dependent Path Loss and Shadowing in Urban Scenarios*, arXiv:2511.10763.
  - https://arxiv.org/abs/2511.10763
- Vinogradov et al., *Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections*, arXiv:2511.15412.
  - https://arxiv.org/abs/2511.15412
- Pang et al., *Geometry-Based Stochastic Probability Models for the LoS and NLoS Paths of A2G Channels under Urban Scenario*, arXiv:2205.09399.
  - https://arxiv.org/abs/2205.09399
- Saboor et al., *Path Loss Modelling for UAV Communications in Urban Scenarios with Random Obstacles*, arXiv:2501.14411.
  - https://arxiv.org/abs/2501.14411
- Feng et al., *Path loss models for air-to-ground radio channels in urban environments*, IEEE VTC 2006.
  - https://doi.org/10.1109/VETECS.2006.1683399
  - https://research-information.bris.ac.uk/en/publications/path-loss-models-for-air-to-ground-radio-channels-in-urban-enviro/
- Oestges, *Physical-Statistical Model for UAV-to-Ground Urban Radio Channels*, ISAP 2020.
  - https://dial.uclouvain.be/pr/boreal/object/boreal%3A251072/datastream/PDF_01/view

### Local repo sources

- `TRY48_PRIOR_RESEARCH_NOTES.md`
- `TRY48_TWO_RAY_AND_NLOS_REFINEMENT.md`
- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`
- `analysis/try45_prior_candidates.md`
- `analysis/try45_a2g_parameter_grid.md`
- `TFGFortySeventhTry47/prior_calibration/regime_obstruction_train_only.md`
- `TFGFortyEighthTry48/prior_calibration/regime_structural_train_only_results.md`
- `TFGFortySeventhTry47/data_utils.py`
- `TFGFortyEighthTry48/data_utils.py`
- `TFGFortyEighthTry48/experiments/fortyeighthtry48_pmnet_prior_gan/fortyeighthtry48_pmnet_prior_gan.yaml`

### Local TFG_Proto1 paper markdown used

- `TFG_Proto1/docs/markdown/2511.10763v1/2511.10763v1.md`
- `TFG_Proto1/docs/markdown/2511.15412v1/2511.15412v1.md`

### Notes

- The recommendation in this note is based on the sources above plus direct inspection of the current repo code/config.
- Where I propose a damped coherent two-ray envelope, ripple carry-over, or a directional shadow-depth term, that is an engineering synthesis from these sources plus the exported `Try 48` visual panels, not a verbatim formula copied from a single paper.
