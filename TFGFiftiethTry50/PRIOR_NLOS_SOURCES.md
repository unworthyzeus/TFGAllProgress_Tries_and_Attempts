# Try 50 Prior Sources and NLoS Rationale

## Current status

These sources were useful to guide the redesign work, but the resulting prior
branches are now mostly archived in:

- `prior_calibration/worse_experiments/`

The current practical baseline still reused by the `Try 49` family is:

- `prior_calibration/regime_obstruction_train_only_from_try47.json`

So this note should now be read mainly as source rationale for attempted `NLoS`
redesigns, not as proof that those redesigns replaced the working baseline.

This note documents the source family behind the current `Try 50` prior work,
with special focus on why the next `NLoS` candidate is changing shape instead of
just re-tuning coefficients.

## Problem observed

The recent quick `DirectML` prior-only run produced:

- overall RMSE: `23.7104 dB`
- `LoS` RMSE: `3.3741 dB`
- `NLoS` RMSE: `40.9450 dB`

This means the current prior is already strong on `LoS`, but the `NLoS` branch
is still too severe. The failure mode is not a lack of detail in the `LoS`
formula. It is that the current `NLoS` branch combines:

- a hard `max(...)` between a height-dependent mean and the supervisor-paper
  `Eq. 11` term,
- an explicit floor above the smoothed `LoS` path loss,
- and an additional obstruction penalty.

That stack tends to lift the entire `NLoS` map too high.

## Sources used

### 1. Supervisor paper: elevation-angle A2G excess and shadowing

Local copy:

- [2511.15412v1.md](C:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2511.15412v1/2511.15412v1.md)

Main role in the prior:

- `Eq. (9)` gives the reference-loss term.
- `Eq. (11)` gives an elevation-dependent `NLoS` excess term.
- `Eq. (12)` gives an elevation-dependent shadow-sigma term.

Why it is still useful:

- it captures the strong dependence of `NLoS` severity on elevation angle;
- it gives a structured, physics-motivated correction rather than a purely
  empirical urban penalty.

Why it should not dominate the whole `NLoS` branch:

- the formula is best treated as a soft elevation penalty, not as the entire
  mean `NLoS` law at all distances;
- when used with a hard `max(...)`, it can overinflate the whole map.

### 2. Height-dependent A2G path loss exponent model

Local copy:

- [2511.10763v1.md](C:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2511.10763v1/2511.10763v1.md)

Web source:

- [Millimeter-Wave UAV Channel Model with Height-Dependent Path Loss and Shadowing in Urban Scenarios](https://arxiv.org/abs/2511.10763)

Main role in the prior:

- `Eq. (14)` gives a height-dependent path-loss exponent:
  `n(h) = n_inf + (n0 - n_inf) exp(-h / h0)`
- the paper shows that `NLoS` PLE decreases toward roughly `2.5-3` at higher
  UAV altitudes.

Why this matters here:

- our UAV heights are not near-ground macro BS heights;
- we are around `7.125 GHz`, fixed frequency, but variable UAV height;
- so the prior should mostly change through the height-dependent large-scale
  slope, not through very precise coefficient tuning.

### 3. 7 GHz urban path-loss anchor

Web source:

- [Deep Learning for Path Loss Prediction at 7 GHz in Urban Environment](https://www.nist.gov/publications/deep-learning-path-loss-prediction-7-ghz-urban-environment)

Why it matters:

- even though it is not a UAV-specific A2G paper, it gives a useful urban
  `7 GHz` anchor showing that a close-in/log-distance `NLoS` mean around a
  `PLE` near `2.8-3.0` is plausible at this frequency;
- that supports moving away from the much harsher effective `NLoS` slope we were
  producing in the prior.

### 4. Elevation-angle based two-ray A2G model

Web source:

- [Elevation-angle based two-ray path loss model for Air-to-Ground wireless channels](https://www.sciencedirect.com/science/article/pii/S2214209621000620)

Main role in the prior:

- motivates using a two-ray family for the large-scale `LoS` branch;
- supports keeping the `LoS` branch coherent/damped-two-ray rather than falling
  back to a fully smooth radial-only law.

## New Try 50 NLoS candidate

The new candidate is now:

- `LoS`: keep the damped coherent two-ray branch that is already working well,
  unchanged;
- `NLoS`: stop trying to define one global mean law and instead model a
  structured extra attenuation on top of the `LoS` reference;
- use the elevation-angle `Eq. (11)` contribution only as a soft `LoS-break`
  cue;
- keep topology-driven urban severity as separate blocker/context terms instead
  of folding everything into one monolithic `NLoS` curve;
- keep ripple in `NLoS`, but weaker than in `LoS`.

In code, this new branch is:

- `hybrid_shadowed_ripple_two_ray_structured_nlos_strict_los`

Its shape is:

- `PL_NLoS = PL_LoS_reference + delta_break + delta_shadow_depth + delta_blocker + delta_context`
- final path:
  - `PL = PL_LoS_reference` on `LoS`
  - `PL = PL_NLoS` on `NLoS`

The practical implication is important:

- the `LoS` mask is treated as known input, not as a soft probability map;
- the `LoS` branch is preserved exactly after nearest-neighbor mask resizing;
- only the `NLoS` pixels are allowed to receive the new structured extra loss.

This is not a direct copy of one paper equation. It is a structured prior that
combines:

- the already-good two-ray `LoS` backbone;
- the supervisor-paper elevation/shadow cues;
- and the geometry intuition behind urban `7 GHz` close-in / height-dependent
  `NLoS` families.

## Why this is a better fit for the current evidence

What the recent quick results say:

- `LoS` is already good enough for a prior;
- `NLoS` is too high everywhere, not just in a few deep-shadow pockets.

So the next useful change is:

- soften the large-scale `NLoS` mean;
- do not force `Eq. 11` to dominate all `NLoS` pixels;
- let local obstruction explain local extra loss, rather than using the base
  law itself as a worst-case bound.

## Practical conclusion after the CIH7 quick run

The quick `DirectML` run with `hybrid_shadowed_ripple_two_ray_cih7_nlos` ended at:

- `overall`: `23.7172 dB`
- `LoS`: `3.3741 dB`
- `NLoS`: `40.9567 dB`

So the new branch still preserves the good `LoS` behavior, but it does not fix
the `NLoS` bottleneck.

That is evidence for a larger conclusion:

- the next improvement will not come from coefficient-level tuning of a single
  monolithic `NLoS` law;
- it should come from redesigning the `NLoS` prior as structured extra loss on
  top of the already-good `LoS` reference.

That is exactly what the new
`hybrid_shadowed_ripple_two_ray_structured_nlos_strict_los` mode is trying to
test.

See:

- [NLOS_PIPELINE_REDESIGN.md](C:/TFG/TFGpractice/TFGFiftiethTry50/NLOS_PIPELINE_REDESIGN.md)
